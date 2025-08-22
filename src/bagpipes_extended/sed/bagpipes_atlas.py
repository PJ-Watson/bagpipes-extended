"""
A module for fitting objects using a grid search sampling of bagpipes.

Parts of the documentation here are copied from
`ACCarnall/bagpipes <https://github.com/ACCarnall/bagpipes>`__, for
clarity, and any use of this module should cite
`Carnall+18 <https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.4379C>`__.
"""

import contextlib
import os
from copy import deepcopy
from functools import partial
from itertools import repeat
from multiprocessing import Manager, Pool, cpu_count, shared_memory
from multiprocessing.managers import SharedMemoryManager
from os import PathLike
from pathlib import Path
from typing import Callable

import bagpipes
import h5py
import numpy as np
from astropy.table import Table
from bagpipes.fitting.prior import dirichlet, prior
from bagpipes.models import model_galaxy
from numpy.random import Generator
from numpy.typing import ArrayLike
from tqdm import tqdm

from bagpipes_extended.c_utils import calc_chisq

__all__ = ["AtlasGenerator", "AtlasFitter"]


default_min_errs = {
    "continuity:massformed": 0.01,
    "continuity:metallicity": 0.01,
    "continuity:dsfr": 0.01,
}


@contextlib.contextmanager
def temp_chdir(path):
    """
    Temporarily change directory using a context manager.

    Parameters
    ----------
    path : PathLike
        The target directory.
    """
    _oldCWD = os.getcwd()
    os.chdir(os.path.abspath(path))

    try:
        yield
    finally:
        os.chdir(_oldCWD)


class AtlasGenerator:
    """
    An extension to ACCarnall/bagpipes, optimised for speed.

    The original sampling technique used by
    `bagpipes <https://bagpipes.readthedocs.io>`__ is the MultiNest code
    (`Feroz+08 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.384..449F>`__).
    Whilst accurate, it is computationally expensive. This code implements
    a Bayesian grid search method instead, trading off speed against
    storage space and accuracy. Rather than calculating or updating the
    model spectra on the fly, we pre-generate a large grid of model
    observables (i.e. flux density in a set of photometric filters). For
    each source, the sampling is thus reduced to a simple array
    evaluation. This method is best-suited to spatially-resolved studies,
    whereby a large number of sources are expected to lie in a similar
    region of the parameter space.

    Parameters
    ----------
    fit_instructions : dict
        A dictionary containing the details of the model, as well as any
        constraints and priors on the parameters.
    filt_list : list, optional
        A list of paths to filter curve files, which should contain a
        column of wavelengths in angstroms followed by a column of
        transmitted fraction values. Only needed for photometric data.
    spec_wavs : `ArrayLike`, optional
        An array of wavelengths at which spectral fluxes should be
        returned. Only required if spectroscopic output is desired.
    spec_units : str, optional
        The units the output spectrum will be returned in. The default
        is “ergscma” for ergs per second per centimetre squared per
        angstrom, but it can also be set to “mujy” for microjanskys.
    phot_units : str, optional
        The units of the input photometry, which defaults to microjanskys,
        ``"mujy"``. The photometry will be converted to ``"ergscma"`` by
        default within the class (see ``out_units``).
    index_list : list | None, optional
        A list of ``dict`` containining definitions for spectral indices,
        by default ``None``.

    Attributes
    ----------
    model_components : dict
        A dictionary containing information about the model to be
        generated.
    model_galaxy : `bagpipes.models.model_galaxy`
        A model galaxy object, which calculates the desired observables
        for a given set of input parameters.
    prior : `bagpipes.fitting.prior`
        An object containing the joint prior distribution of the
        model parameters.
    params : list
        The model parameters to be varied.
    limits : list
        The limits for varying parameters.
    pdfs : list
        The probability densities of parameters within their limits.
    hyper_params : list
        The hyperparameters of prior distributions.
    mirror_pars : dict
        Parameters which mirror another parameter.
    ndim : int
        The dimensionality of the fit, i.e. ``len(params)``.
    param_vectors : ArrayLike
        A 2D array containing N samples of ``ndim`` parameters.
    model_atlas : ArrayLike
        A 2D array, containing for each of the N samples the expected
        photometry for each of the filters in ``filt_list``.

    Warnings
    --------
    Currently, only photometric data is supported. Generating a model grid
    for spectroscopic data will not work.

    Notes
    -----
    For this method to provide an accurate estimate of the posterior
    distribution, the N-dimensional parameter space must be well-sampled.
    Increasing the sampling density comes with a corresponding increase in
    storage space, and can lead to memory errors if not careful.
    """

    def __init__(
        self,
        fit_instructions: dict,
        filt_list: list | None = None,
        spec_wavs: ArrayLike | None = None,
        spec_units: str = "ergscma",
        phot_units: str = "mujy",
        index_list: str | None = None,
    ):

        self.fit_instructions = deepcopy(fit_instructions)
        self.model_components = deepcopy(fit_instructions)

        # self._set_constants()
        self._process_fit_instructions()

        self.prior = prior(self.limits, self.pdfs, self.hyper_params)
        self.model_galaxy = None

        self._model_kwargs = {
            "filt_list": filt_list,
            "spec_wavs": spec_wavs,
            "index_list": index_list,
            "spec_units": spec_units,
            "phot_units": phot_units,
        }

        self.param_vectors = None
        self.model_atlas = None

    def _process_fit_instructions(self) -> None:
        """
        Load the fit instructions and populate the class attributes.

        Originally from `bagpipes.fitting.fitted_model`.
        """
        all_keys = []  # All keys in fit_instructions and subs
        all_vals = []  # All vals in fit_instructions and subs

        self.params = []  # Parameters to be fitted
        self.limits = []  # Limits for fitted parameter values
        self.pdfs = []  # Probability densities within lims
        self.hyper_params = []  # Hyperparameters of prior distributions
        self.mirror_pars = {}  # Params which mirror a fitted param

        # Flatten the input fit_instructions dictionary.
        for key in list(self.fit_instructions):
            if not isinstance(self.fit_instructions[key], dict):
                all_keys.append(key)
                all_vals.append(self.fit_instructions[key])

            else:
                for sub_key in list(self.fit_instructions[key]):
                    all_keys.append(key + ":" + sub_key)
                    all_vals.append(self.fit_instructions[key][sub_key])

        # Sort the resulting lists alphabetically by parameter name.
        indices = np.argsort(all_keys)
        all_vals = [all_vals[i] for i in indices]
        all_keys.sort()

        # Find parameters to be fitted and extract their priors.
        for i in range(len(all_vals)):
            # R_curve cannot be fitted and is either unset or must be a 2D numpy array
            if not all_keys[i] == "R_curve":
                if isinstance(all_vals[i], tuple):
                    self.params.append(all_keys[i])
                    self.limits.append(all_vals[i])  # Limits on prior.

                    # Prior probability densities between these limits.
                    prior_key = all_keys[i] + "_prior"
                    if prior_key in list(all_keys):
                        self.pdfs.append(all_vals[all_keys.index(prior_key)])

                    else:
                        self.pdfs.append("uniform")

                    # Any hyper-parameters of these prior distributions.
                    self.hyper_params.append({})
                    for i in range(len(all_keys)):
                        if all_keys[i].startswith(prior_key + "_"):
                            hyp_key = all_keys[i][len(prior_key) + 1 :]
                            self.hyper_params[-1][hyp_key] = all_vals[i]

                # Find any parameters which mirror the value of a fit param.
                if all_vals[i] in all_keys:
                    self.mirror_pars[all_keys[i]] = all_vals[i]

                if all_vals[i] == "dirichlet":
                    n = all_vals[all_keys.index(all_keys[i][:-6])]
                    comp = all_keys[i].split(":")[0]
                    for j in range(1, n):
                        self.params.append(comp + ":dirichletr" + str(j))
                        self.pdfs.append("uniform")
                        self.limits.append((0.0, 1.0))
                        self.hyper_params.append({})

        # Find the dimensionality of the fit
        self.ndim = len(self.params)

    def _update_model_components(
        self,
        param: ArrayLike,
    ) -> dict:
        """
        Generates a model object with the current parameters.

        Originally from `bagpipes.fitting.fitted_model`, modified to allow
        for running in parallel (no attributes overwritten).

        Returns
        -------
        new_components : dict
            The updated components for a model galaxy.
        """

        new_components = deepcopy(self.model_components)

        dirichlet_comps = []

        # Substitute values of fit params from param into model_comp.
        for i in range(len(self.params)):
            split = self.params[i].split(":")
            if len(split) == 1:
                new_components[self.params[i]] = param[i]

            elif len(split) == 2:
                if "dirichlet" in split[1]:
                    if split[0] not in dirichlet_comps:
                        dirichlet_comps.append(split[0])

                else:
                    new_components[split[0]][split[1]] = param[i]

        # Set any mirror params to the value of the relevant fit param.
        for key in list(self.mirror_pars):
            split_par = key.split(":")
            split_val = self.mirror_pars[key].split(":")
            fit_val = new_components[split_val[0]][split_val[1]]
            new_components[split_par[0]][split_par[1]] = fit_val

        # Deal with any Dirichlet distributed parameters.
        if len(dirichlet_comps) > 0:
            comp = dirichlet_comps[0]
            n_bins = 0
            for i in range(len(self.params)):
                split = self.params[i].split(":")
                if (split[0] == comp) and ("dirichlet" in split[1]):
                    n_bins += 1

            new_components[comp]["r"] = np.zeros(n_bins)

            j = 0
            for i in range(len(self.params)):
                split = self.params[i].split(":")
                if (split[0] == comp) and "dirichlet" in split[1]:
                    new_components[comp]["r"][j] = param[i]
                    j += 1

            tx = dirichlet(new_components[comp]["r"], new_components[comp]["alpha"])

            new_components[comp]["tx"] = tx

        return new_components

    def sample_from_params(
        self,
        x: ArrayLike,
        model_gal: model_galaxy | None = None,
    ) -> model_galaxy:
        """
        Generate an updated model from a position in parameter space.

        Parameters
        ----------
        x : ArrayLike
            A 1D array describing a position in parameter space,
            of equal length to `AtlasGenerator.ndim`.
        model_gal : `bagpipes.models.model_galaxy` | None, optional
            The model galaxy to be updated. By default `None`, which will
            result in a new object being returned.

        Returns
        -------
        `bagpipes.models.model_galaxy`
            The updated model galaxy.
        """

        new_comps = self._update_model_components(x)

        if model_gal is None:
            model_gal = model_galaxy(new_comps, **self._model_kwargs)

        model_gal.update(new_comps)

        return model_gal

    def store_photometry(
        self,
        model_gal: model_galaxy,
    ) -> ArrayLike:
        """
        A helper function to format photometric models for grid storage.

        Parameters
        ----------
        model_gal : `bagpipes.models.model_galaxy`
            The model galaxy to be stored.

        Returns
        -------
        ArrayLike
            The predicted photometry and a flag indicating whether the
            model star-formation history is unphysical.
        """
        return np.asarray([*model_gal.photometry, float(model_gal.sfh.unphysical)])

    def _get_pgb_pos(self, shared_list):
        # Acquire lock and get a progress bar slot
        for i in range(self.n_proc):
            if shared_list[i] == 0:
                shared_list[i] = 1
                return i

    def _release_pgb_pos(self, shared_list, slot):
        shared_list[slot] = 0

    def _shared_memory_worker(
        self,
        indices,
        shared_init,
        init_shape,
        shared_output,
        output_shape,
        store_fn,
        tqdm_list,
    ):
        # return
        # print(shared_init[indices].shape)
        # param_cube = np.zeros_like(cubes)
        # print (indices)
        pgb_pos = self._get_pgb_pos(tqdm_list)

        worker_id = indices[0]
        indices = indices[1]
        model_obj = None

        shm_init = shared_memory.SharedMemory(name=shared_init)
        init_arr = np.ndarray(init_shape, dtype=float, buffer=shm_init.buf)
        shm_output = shared_memory.SharedMemory(name=shared_output)
        output_arr = np.ndarray(output_shape, dtype=float, buffer=shm_output.buf)

        # for i, c in tqdm(enumerate(cubes), total=n_samples):
        try:
            for i, idx_i in tqdm(
                enumerate(indices),
                total=len(indices),
                position=pgb_pos + 1,
                desc=f"Worker {worker_id}",
                leave=False,
                mininterval=1,
            ):
                # print(c)
                param_vector = self.prior.transform(init_arr[idx_i])
                init_arr[idx_i] = param_vector
                # print (param_vector)
                if i == 0:
                    model_obj = self.sample_from_params(param_vector)
                else:
                    model_obj = self.sample_from_params(
                        param_vector, model_gal=model_obj
                    )
                # print (store_fn)
                output_arr[idx_i] = store_fn(model_obj)
                # print (model_obj.photometry)
                # print (shared_output[idx_i])
            # print (output_arr)
            # print (self.model_atlas[i])
        finally:
            self._release_pgb_pos

    def gen_samples(
        self,
        n_samples: int = 10,
        seed: int = 2744,
        parallel: int = 0,
    ) -> None:
        """
        Generate a model grid from a sampling of the joint prior volume.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples to generate. By default 10, although a
            useful number will typically be > 10^5.
        seed : int | None, optional
            The seed for the random sampling, by default 2744. If None,
            then a new seed will be generated each time this method is
            called.
        parallel : int, optional
            The number of processes to use when generating the model grid.
            By default this is 0, and the code will run on a single
            process. If set to an integer less than 0, this will run on
            the number of cores returned by `multiprocessing.cpu_count`.

        Notes
        -----
        If ``spec_wavs`` or ``index_list`` were passed as arguments during
        class initialisation, an error will be raised (until I get around
        to implementing them).
        """

        n_samples = int(n_samples)

        # Something to generate multiple samples
        # Check if running in MPI
        # Check what data needs to be extracted
        if self._model_kwargs["filt_list"] is not None:
            store_fn = self.store_photometry
            n_output = len(self._model_kwargs["filt_list"]) + 1
        elif self._model_kwargs["spec_wavs"] is not None:
            raise NotImplementedError(
                "This is more complicated than photometry (see `noise' and `calib' "
                "keys), and may be implemented at a later stage."
            )
        elif self._model_kwargs["index_list"] is not None:
            raise NotImplementedError(
                "This seems to be a rarely used part of bagpipes, and is unlikely "
                "to be implemented."
            )

        rng = np.random.default_rng(seed=seed)

        # if rank == 0:
        # print (n_samples, self.ndim)
        if parallel == 0:
            cubes = rng.random((n_samples, self.ndim))

            self.model_atlas = np.ndarray((n_samples, n_output), dtype=float)
            model_obj = None
            # print (self.model_atlas.shape)

            for i, c in tqdm(enumerate(cubes), total=n_samples):
                # print(c)
                param_vector = self.prior.transform(c)
                cubes[i] = param_vector
                # print (param_vector)
                if i == 0:
                    model_obj = self.sample_from_params(param_vector)
                else:
                    model_obj = self.sample_from_params(
                        param_vector, model_gal=model_obj
                    )
                # print (len(store_fn(model_obj)))
                self.model_atlas[i] = store_fn(model_obj)
                # print (self.model_atlas[i])
            self.param_vectors = cubes.copy()

        else:

            if parallel < 0:
                self.n_proc = cpu_count()
            else:
                self.n_proc = parallel

            print(
                f"Generating {n_samples} samples using {self.n_proc} process"
                f"{"es" if self.n_proc>1 else ""}, "
                f"with {self.ndim} free parameters."
            )

            # https://github.com/tqdm/tqdm/issues/1000
            manager = Manager()
            shared_list = manager.list([0] * self.n_proc)
            lock = manager.Lock()

            with SharedMemoryManager() as smm:
                cubes = rng.random((n_samples, self.ndim))
                # print (cubes.shape)
                shm_in = smm.SharedMemory(size=cubes.nbytes)
                shared_init_params = np.ndarray(
                    cubes.shape, dtype=cubes.dtype, buffer=shm_in.buf
                )
                shared_init_params[:] = cubes[:]
                # print (shared_init_params.shape)

                # if chunk_size is None:
                #     chunk_size = n_samples // parallel
                _indices = np.arange(n_samples)
                array_indices = np.array_split(_indices, self.n_proc)

                # cubes = rng.random((n_samples, self.ndim))
                shm_out = smm.SharedMemory(
                    size=(np.dtype(float).itemsize * n_samples * n_output)
                )
                shared_model_atlas = np.ndarray(
                    (n_samples, n_output), dtype=float, buffer=shm_out.buf
                )
                # shared_model_atlas[:] = cubes[:]

                with Pool(
                    processes=self.n_proc, initializer=tqdm.set_lock, initargs=(lock,)
                ) as pool:
                    multi_fn = partial(
                        self._shared_memory_worker,
                        shared_init=shm_in.name,
                        init_shape=shared_init_params.shape,
                        shared_output=shm_out.name,
                        output_shape=shared_model_atlas.shape,
                        store_fn=store_fn,
                        tqdm_list=shared_list,
                    )
                    pool.map(multi_fn, enumerate(array_indices))
                    self.model_atlas = shared_model_atlas.copy()
                    self.param_vectors = shared_init_params.copy()

    def write_samples(
        self,
        filepath: PathLike,
    ) -> None:
        """
        Write the generated grid to an HDF5 file.

        Parameters
        ----------
        filepath : PathLike
            The path to which the grid will be written.
        """

        # print(self.fit_instructions)
        # print(self.model_components)
        # print(self.params)

        if self.param_vectors is not None and self.model_atlas is not None:

            with h5py.File(filepath, "w") as file:

                # This is necessary for converting large arrays to strings
                np.set_printoptions(threshold=10**7)
                file.attrs["fit_instructions"] = str(self.fit_instructions)
                file.attrs["model_kwargs"] = str(self._model_kwargs)
                np.set_printoptions(threshold=10**4)

                for i, k in enumerate(self.params):
                    file.create_dataset(k, data=self.param_vectors[:, i])

                file.create_dataset("model_atlas", data=self.model_atlas)

            #     # self.results["fit_instructions"] = self.fit_instructions

            # file.close()


# def _test_shared_memory(indices, shared_init, shared_output):
#     # return
#     print(shared_init[indices].shape)


# @staticmethod
def fit_single(
    # self,
    galaxy: bagpipes.galaxy,
    z_range: ArrayLike | None = None,
    atlas_name: str | None = None,
    atlas_shape: tuple | None = None,
    param_samples_name: str | None = None,
    param_samples_shape: tuple | None = None,
    n_posterior: int = 500,
    rng: Generator | None = None,
    params: list | None = None,
) -> dict:
    """
    Fit a single `bagpipes.galaxy` object.

    Parameters
    ----------
    galaxy : `bagpipes.galaxy`
        A galaxy object containing the photometric data to be fitted.
        Note that fitting spectroscopic data is currently not
        supported.
    z_range : ArrayLike | None
        The redshift range of the galaxy to be fitted. By default
        ``None``, meaning the entire atlas will be used.
    atlas_name : str | None
        The name of the model atlas in shared memory.
    atlas_shape : tuple | None
        The shape of the model atlas in shared memory.
    param_samples_name : str | None
        The name of the parameter samples array in shared memory.
    param_samples_shape : tuple | None
        The shape of the parameter samples array in shared memory.
    n_posterior : int, optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete. Default is 500.
    rng : np.random.default_rng, optional
        The RNG generator.
    params : list, optional
        A list of the parameter names used in the fit.

    Returns
    -------
    dict
        A dictionary containing the results of the fit.
    """

    if galaxy.photometry_exists:
        log_error_factors = np.log(2 * np.pi * galaxy.photometry[:, 2] ** 2)
        K_phot = -0.5 * np.sum(log_error_factors)
        inv_sigma_sq_phot = 1.0 / galaxy.photometry[:, 2] ** 2

    if galaxy.index_list is not None:
        log_error_factors = np.log(2 * np.pi * galaxy.indices[:, 1] ** 2)
        K_ind = -0.5 * np.sum(log_error_factors)
        inv_sigma_sq_ind = 1.0 / galaxy.indices[:, 1] ** 2

    shm_model_atlas = shared_memory.SharedMemory(name=atlas_name)
    shared_model_atlas = np.ndarray(
        atlas_shape, dtype=float, buffer=shm_model_atlas.buf
    )

    shm_param_samples = shared_memory.SharedMemory(name=param_samples_name)
    shared_param_samples = np.ndarray(
        param_samples_shape, dtype=float, buffer=shm_param_samples.buf
    )

    # diff = (self.model_atlas - galaxy.photometry[:, 1]) ** 2

    # chisq_arr = np.nansum(
    #     (self.model_atlas - galaxy.photometry[:, 1]) ** 2 * inv_sigma_sq_phot,
    #     axis=-1,
    # )  # /(diff.shape[1]-len(self.params))
    if z_range is None:
        chisq_arr = calc_chisq(
            shared_model_atlas,
            np.ascontiguousarray(galaxy.photometry[:, 1]),
            inv_sigma_sq_phot,
        )
        param2d = shared_param_samples  # [self.param_samples[k] for k in self.params]
    else:
        # z_idxs = slice(np.argmin(),np.argmax(self.param_samples["redshift"]<=ID))
        # print (params)
        # print (np.asarray(params)=="redshift", shared_param_samples.shape)
        # print (shared_param_samples[np.asarray(params)=="redshift"])
        z_idxs = np.searchsorted(
            shared_param_samples[np.asarray(params) == "redshift"].ravel(), z_range
        )
        if z_idxs[0] == shared_param_samples.shape[1] or z_idxs[1] == 0:
            raise ValueError("Redshift range not covered by model atlas.")
        chisq_arr = calc_chisq(
            shared_model_atlas[z_idxs[0] : z_idxs[1]],
            np.ascontiguousarray(galaxy.photometry[:, 1]),
            inv_sigma_sq_phot,
        )
        # param2d = [
        #     self.param_samples[k][z_idxs[0] : z_idxs[-1]] for k in self.params
        # ]
        param2d = shared_param_samples[:, z_idxs[0] : z_idxs[-1]]
    map_idx = np.argmin(chisq_arr)
    lnlike_oned = K_phot - 0.5 * chisq_arr

    # param2d.append(lnlike_oned)
    # print ("1", param2d.shape)
    param2d = np.concatenate([param2d, np.atleast_2d(lnlike_oned)]).T
    # param2d = np.column_stack([param2d, lnlike_oned])
    # print ("2", param2d.shape)

    # chisq_arr /= self.ndim

    weights = np.exp(-0.5 * chisq_arr) / np.nansum(np.exp(-0.5 * chisq_arr))
    finite_weights = np.isfinite(weights)

    # import matplotlib.pyplot as plt
    # print (self.params)
    # # plt.scatter(self.param_samples["continuity:massformed"], chisq_arr/self.ndim)
    # # plt.ylim((0,1e3))
    # plt.scatter(self.param_samples["continuity:massformed"], weights)
    # plt.show()
    # exit()

    if np.nansum(finite_weights) == 0:
        samples2d = np.zeros((n_posterior, param2d.shape[1]))
    else:
        samples2d = rng.choice(
            param2d[finite_weights],
            n_posterior,
            p=weights[finite_weights],
        )

    results = {}

    results["samples2d"] = samples2d[:, :-1]
    # if self.add_errs is not None:
    #     results["samples2d"] += self.add_errs
    results["lnlike"] = samples2d[:, -1]
    results["lnz"] = lnlike_oned[map_idx]
    results["lnz_err"] = np.nan

    results["median"] = np.median(samples2d, axis=0)
    results["conf_int"] = np.percentile(results["samples2d"], (16, 84), axis=0)

    return results


class AtlasFitter:
    """
    A class to fit models to observational data.

    This class relies on a sampled grid produced by `AtlasGenerator`.
    In all other aspects, the functionality should be the same as using
    `bagpipes.fit_catalogue()`.

    Parameters
    ----------

    fit_instructions : dict
        A dictionary containing instructions on the kind of model which
        should be fitted to the data. This should match the previously
        generated model grid.
    atlas_path : os.PathLike
        The location of the previously generated model grid.
    out_path : os.PathLike
        The location to which all output files will be saved. This will be
        created if it does not already exist.
    seed : int | None, optional
        The seed for the random sampling, by default 2744. If ``None``,
        then a new seed will be generated each time this method is
        called.
    overwrite : bool, optional
        If ``True``, then any existing posterior distributions and output
        catalogues will be overwritten. By default ``False``.
    """

    def __init__(
        self,
        fit_instructions: dict,
        atlas_path: PathLike,
        out_path: PathLike,
        seed: int | None = 2744,
        overwrite: bool = False,
    ):

        self.run = "."
        self.fit_instructions = deepcopy(fit_instructions)
        self.n_posterior = 500

        self.rng = np.random.default_rng(seed=seed)
        self.out_path = Path(out_path)

        self._process_fit_instructions()

        self.overwrite = overwrite

        # # If a posterior file already exists load it.
        if Path(atlas_path).exists():
            self._load_atlas(atlas_path)

    def _process_fit_instructions(self):
        """
        Load the fit instructions and populate the class attributes.

        Originally from `bagpipes.fitting.fitted_model`.
        """
        all_keys = []  # All keys in fit_instructions and subs
        all_vals = []  # All vals in fit_instructions and subs

        self.params = []  # Parameters to be fitted
        self.limits = []  # Limits for fitted parameter values
        self.pdfs = []  # Probability densities within lims
        self.hyper_params = []  # Hyperparameters of prior distributions
        self.mirror_pars = {}  # Params which mirror a fitted param

        # Flatten the input fit_instructions dictionary.
        for key in list(self.fit_instructions):
            if not isinstance(self.fit_instructions[key], dict):
                all_keys.append(key)
                all_vals.append(self.fit_instructions[key])

            else:
                for sub_key in list(self.fit_instructions[key]):
                    all_keys.append(key + ":" + sub_key)
                    all_vals.append(self.fit_instructions[key][sub_key])

        # Sort the resulting lists alphabetically by parameter name.
        indices = np.argsort(all_keys)
        all_vals = [all_vals[i] for i in indices]
        all_keys.sort()

        # Find parameters to be fitted and extract their priors.
        for i in range(len(all_vals)):
            # R_curve cannot be fitted and is either unset or must be a 2D numpy array
            if not all_keys[i] == "R_curve":
                if isinstance(all_vals[i], tuple):
                    self.params.append(all_keys[i])
                    self.limits.append(all_vals[i])  # Limits on prior.

                    # Prior probability densities between these limits.
                    prior_key = all_keys[i] + "_prior"
                    if prior_key in list(all_keys):
                        self.pdfs.append(all_vals[all_keys.index(prior_key)])

                    else:
                        self.pdfs.append("uniform")

                    # Any hyper-parameters of these prior distributions.
                    self.hyper_params.append({})
                    for i in range(len(all_keys)):
                        if all_keys[i].startswith(prior_key + "_"):
                            hyp_key = all_keys[i][len(prior_key) + 1 :]
                            self.hyper_params[-1][hyp_key] = all_vals[i]

                # Find any parameters which mirror the value of a fit param.
                if all_vals[i] in all_keys:
                    self.mirror_pars[all_keys[i]] = all_vals[i]

                if all_vals[i] == "dirichlet":
                    n = all_vals[all_keys.index(all_keys[i][:-6])]
                    comp = all_keys[i].split(":")[0]
                    for j in range(1, n):
                        self.params.append(comp + ":dirichletr" + str(j))
                        self.pdfs.append("uniform")
                        self.limits.append((0.0, 1.0))
                        self.hyper_params.append({})

        # Find the dimensionality of the fit
        self.ndim = len(self.params)

    def _load_atlas(self, atlas_path: PathLike) -> None:
        """
        Load the generated grid from an HDF5 file.

        Parameters
        ----------
        atlas_path : PathLike
            The path to the model grid.
        """
        with h5py.File(atlas_path, "r") as file:
            fit_info_str = file.attrs["fit_instructions"]
            fit_info_str = fit_info_str.replace("numpy.", "")
            fit_info_str = fit_info_str.replace("np.", "")
            fit_info_str = fit_info_str.replace("array", "np.array")
            fit_info_str = fit_info_str.replace("float", "np.float")
            if eval(fit_info_str) != self.fit_instructions:
                err_text = ""
                for k, v in eval(fit_info_str).items():
                    if type(v) is dict:
                        for kk, vv in v.items():
                            if not vv == self.fit_instructions[k][kk]:
                                err_text += f"KEY: {kk}\t ATLAS: {vv}\t "
                                err_text += (
                                    f"FIT_INSTR: {self.fit_instructions[k][kk]}\n"
                                )

                    else:
                        if not v == self.fit_instructions[k]:
                            err_text += f"KEY: {k}\t ATLAS: {v}\t "
                            err_text += f"FIT_INSTR: {self.fit_instructions[k]}\n"

                raise ValueError(f"Fit instructions do not match:\n{err_text}")

            model_kwargs_str = file.attrs["model_kwargs"]
            model_kwargs_str = model_kwargs_str.replace("array", "np.array")
            model_kwargs_str = model_kwargs_str.replace("float", "np.float")

            self.model_kwargs = eval(model_kwargs_str)
            self.smm = SharedMemoryManager()
            self.smm.start()
            # cubes = rng.random((n_samples, self.ndim))
            # # print (cubes.shape)
            model_atlas = np.array(file["model_atlas"])
            physical_idx = (model_atlas[:, -1] != 1.0) & (
                np.isfinite(np.sum(model_atlas, axis=-1))
            )
            model_atlas = model_atlas[physical_idx][:, :-1]

            # shared_model_atlas = shared_model_atlas[physical_idx][:, :-1]
            param_samples = np.empty((len(self.params), model_atlas.shape[0]))
            for i, k in enumerate(self.params):
                param_samples[i] = np.array(file[k][physical_idx])

            if "redshift" in self.params:
                z_idx = np.argsort(np.array(file["redshift"][physical_idx]))
                model_atlas = model_atlas[z_idx]
                # for i, k in enumerate(self.params):
                #     # print (k, self.param_samples[k], self.param_samples[k].dtype, self.para)
                #     self.param_samples[i] = self.param_samples[i][z_idx]
                param_samples = param_samples[:, z_idx]

            shm_model_atlas = self.smm.SharedMemory(size=model_atlas.nbytes)
            self.atlas_shape = model_atlas.shape
            self.atlas_name = shm_model_atlas.name
            shared_model_atlas = np.ndarray(
                model_atlas.shape, dtype=model_atlas.dtype, buffer=shm_model_atlas.buf
            )
            shared_model_atlas[:] = model_atlas[:]

            shm_param_samples = self.smm.SharedMemory(size=param_samples.nbytes)
            self.param_samples_shape = param_samples.shape
            self.param_samples_name = shm_param_samples.name
            shared_param_samples = np.ndarray(
                param_samples.shape,
                dtype=param_samples.dtype,
                buffer=shm_param_samples.buf,
            )
            shared_param_samples[:] = param_samples[:]
            # model_atlas = np.array(file["model_atlas"]).nbytes

            self.n_samples = shared_model_atlas.shape[0]

    @staticmethod
    def _fit_object(
        # self,
        ID: str,
        filt_list: ArrayLike = None,
        z=None,
        load_data=None,
        spectrum_exists=None,
        photometry_exists=None,
        load_indices=None,
        index_list=None,
        posterior_parent_path=None,
        redshift_range=None,
        fit_instructions=None,
        out_path=None,
        run=None,
        n_posterior=None,
        full_catalogue=None,
        vars=None,
        atlas_name=None,
        atlas_shape=None,
        param_samples_name=None,
        param_samples_shape=None,
        rng=None,
        params=None,
        **kwargs,
    ) -> dict:
        """
        Fit a single object from the catalogue.

        The posterior output is saved to the corresponding directory, and
        the row of the results table is returned.

        Parameters
        ----------
        ID : str
            The unique identifier for the galaxy in the catalogue.

        Returns
        -------
        dict
            The row data for the results table.
        """

        # # Get the correct filt_list for this object
        # filt_list = self.cat_filt_list
        # if self.vary_filt_list:
        #     filt_list = self.cat_filt_list[np.argmax(self.IDs == ID)]

        galaxy = bagpipes.galaxy(
            ID,
            load_data,
            filt_list=filt_list,
            spectrum_exists=spectrum_exists,
            photometry_exists=photometry_exists,
            load_indices=load_indices,
            index_list=index_list,
        )

        # print("RNG", rng)
        # print(dir(rng))
        # posterior_path = self.out_path / "pipes" / "posterior" / self.run / f"{ID}.h5"
        posterior_path = posterior_parent_path / f"{ID}.h5"

        if not posterior_path.is_file():

            # if self.redshifts is not None:
            #     ind = np.argmax(self.IDs == ID)
            #     # print (f"{ind=}", f"{ID=}", self.IDs)
            #     z = self.redshifts[ind]
            if z is not None:
                z_range = [z - redshift_range / 2, z + redshift_range / 2]
            else:
                z_range = None

            results = fit_single(
                galaxy,
                z_range,
                atlas_name,
                atlas_shape,
                param_samples_name,
                param_samples_shape,
                n_posterior=n_posterior,
                rng=rng,
                params=params,
            )

            with h5py.File(posterior_path, "w") as file:

                # This is necessary for converting large arrays to strings
                np.set_printoptions(threshold=10**7)
                file.attrs["fit_instructions"] = str(fit_instructions)
                np.set_printoptions(threshold=10**4)

                for k in results.keys():
                    file.create_dataset(k, data=results[k])

                # results["fit_instructions"] = self.fit_instructions

        else:
            with h5py.File(posterior_path, "r") as file:
                results = {}
                # results["lnz"]
                results["lnz"] = file["lnz"][()]
                results["lnz_err"] = file["lnz_err"][()]
                # print (results)

        with temp_chdir(out_path):
            # Create a posterior object to hold the results of the fit.
            posterior_obj = bagpipes.fitting.posterior(
                galaxy, run=run, n_samples=n_posterior
            )

            samples = posterior_obj.samples

            row_data = {}
            row_data["#ID"] = ID

            for v in vars:
                if v == "UV_colour":
                    values = samples["uvj"][:, 0] - samples["uvj"][:, 1]

                elif v == "VJ_colour":
                    values = samples["uvj"][:, 1] - samples["uvj"][:, 2]

                else:
                    values = samples[v]

                row_data[f"{v}_16"] = np.percentile(values, 16)
                row_data[f"{v}_50"] = np.percentile(values, 50)
                row_data[f"{v}_84"] = np.percentile(values, 84)

            if z is not None:
                row_data["input_redshift"] = z
            else:
                row_data["input_redshift"] = np.nan

            row_data["log_evidence"] = results["lnz"]
            row_data["log_evidence_err"] = results["lnz_err"]

            if full_catalogue and photometry_exists:
                row_data["chisq_phot"] = np.min(samples["chisq_phot"])
                n_bands = np.sum(galaxy.photometry[:, 1] != 0.0)
                row_data["n_bands"] = n_bands

        return row_data

    def _setup_vars(self):
        """
        Set up a list of variables to go in the output catalogue.
        """

        self.vars = self.params.copy()
        self.vars += [
            "stellar_mass",
            "formed_mass",
            "sfr",
            "ssfr",
            "nsfr",
            "mass_weighted_age",
            "tform",
            "tquench",
        ]

        if self.full_catalogue:
            self.vars += ["UV_colour", "VJ_colour"]

    def _setup_colnames(self):
        """
        Set up the initial blank output catalogue.
        """

        cols = ["#ID"]
        for var in self.vars:
            cols += [var + "_16", var + "_50", var + "_84"]

        cols += ["input_redshift", "log_evidence", "log_evidence_err"]

        if self.full_catalogue and self.photometry_exists:
            cols += ["chisq_phot", "n_bands"]

        # self.cat = Table(np.zeros((self.IDs.shape[0], len(cols))),
        #                         names=cols)

        # self.cat["#ID"] = self.IDs
        # # self.cat.index = self.IDs

        # if self.redshifts is not None:
        #     self.cat["input_redshift"] = self.redshifts

        return cols

    def check_errs_dict(self, min_uncertainties: dict | None = None) -> None:
        """
        Initialise an optional additional uncertainty for the posterior.

        This aims to reduce discretisation effects where the n-dimensional
        parameter space is undersampled, and is equivalent to a
        convolution of the posterior distribution with a Gaussian
        distribution.

        Parameters
        ----------
        min_uncertainties : dict | None, optional
            A dictionary containing the (partial) names of the parameters
            as keys, and the standard deviation of the additional
            uncertainty. If ``None``, the posterior samples will not be
            modified.
        """

        if min_uncertainties is None:
            self.add_errs = None
        else:
            self.add_errs = np.zeros((self.n_posterior, len(self.params)))
            for k, v in min_uncertainties.items():
                for i in [j for j, n in enumerate(self.params) if k in n]:
                    self.add_errs[:, i] = self.rng.normal(
                        loc=0.0, scale=v, size=self.n_posterior
                    )

    def fit_catalogue(
        self,
        IDs: list[str],
        load_data: Callable,
        spectrum_exists: bool = False,
        photometry_exists: bool = True,
        make_plots: bool = False,
        cat_filt_list: ArrayLike | None = None,
        vary_filt_list: bool = False,
        redshifts: ArrayLike | None = None,
        redshift_range: float = 0.01,
        run: str = ".",
        analysis_function: Callable | None = None,
        n_posterior: int = 500,
        full_catalogue: bool = False,
        load_indices: Callable | str | None = None,
        index_list: list | None = None,
        parallel: int = 0,
        min_uncertainties: dict = default_min_errs,
    ):
        """
        Fit a catalogue of sources using the model grid.

        Parameters
        ----------
        IDs : list[str]
            A list of unique identifiers for the objects in the catalogue.
        load_data : function
            A function which takes ID as an argument and returns the model
            spectrum and photometry. Spectrum should come first and be an
            array with a column of wavelengths in Angstroms, a column of
            fluxes in erg/s/cm^2/A and a column of flux errors in the same
            units. Photometry should come second and be an array with a
            column of fluxes in microjanskys and a column of flux errors
            in the same units.
        spectrum_exists : bool, optional
            If the objects do not have spectroscopic data, set this to
            ``False``. In this case, ``load_data()`` should only return
            photometry. By default ``False``.
        photometry_exists : bool, optional
            If the objects do not have photometric data, set this to
            ``False``. In this case, ``load_data()`` should only return a
            spectrum. By default ``True``.
        make_plots : bool, optional
            Whether to make output plots for each object; by default
            ``False``.
        cat_filt_list : ArrayLike | None, optional
            The ``filt_list``, or a list of ``filt_list`` for the
            catalogue, by default ``None``.
        vary_filt_list : bool, optional
            If ``True``, each object has a different filter list, as
            described in ``cat_filt_list``, which must have
            ``len(cat_filt_list)==len(IDs)``. By default ``False``.
        redshifts : ArrayLike | None, optional
            A list of values of redshift for each object to be fixed to,
            by default ``None``.
        redshift_range : float, optional
            # If this is set, the redshift for each object will be assigned
            # a Gaussian prior centred on the value in ``redshifts``, with
            # this standard deviation. Hard limits will be placed at
            # :math:`{\\pm3\\sigma}`. By default, this is set to 0.0.
            TBD.
        run : str, optional
            The subfolder into which outputs will be saved, useful e.g.
            for fitting more than one model configuration to the same
            data. Note that the posterior outputs are already saved into a
            subfolder following the `bagpipes` convention of
            ``{out_path} / pipes``.
        analysis_function : function | None, optional
            A function to be run on each completed fit. This function must
            take the fit object as its only argument. By default None.
        n_posterior : int, optional
            How many equally weighted samples should be generated from the
            posterior once fitting is complete. Default is 500.
        full_catalogue : bool, optional
            This adds minimum :math:`{\\chi^2}` values and rest-frame UVJ
            magnitudes to the output catalogue. However, as this requires
            the model spectrum to be calculated, it can significantly
            increase the time taken. By default, ``False``.
        load_indices : function | str | None, optional
            Load spectral index information for the galaxy. This can
            either be a function which takes the galaxy ``ID`` and returns
            index values in the same order as they are defined in
            ``index_list``, or the str ``"from_spectrum"``, in which case
            the code will measure the indices from the observed spectrum
            for the galaxy. By default ``None``.
        index_list : list | None, optional
            A list of dicts containining definitions for spectral indices,
            by default `None``.
        parallel : int, optional
            The number of processes to use when generating the model grid.
            By default this is 0, and the code will run on a single
            process. If set to an integer less than 0, this will run on
            the number of cores returned by `multiprocessing.cpu_count`.
        min_uncertainties : dict | None, optional
            A dictionary containing the (partial) names of the parameters
            as keys, and the standard deviation of the additional
            uncertainty. If ``None``, the posterior samples will not be
            modified. By default, this is set by ``default_min_errs``.
        """

        self.IDs = np.array(IDs).astype(str)
        self.load_data = load_data
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        # self.make_plots = make_plots
        # self.cat_filt_list = cat_filt_list
        if vary_filt_list:
            assert len(cat_filt_list) == len(
                self.IDs
            ), "A variable filter list must have the same number of elements as the list of IDs."
        # self.vary_filt_list = vary_filt_list
        self.redshifts = redshifts
        self.redshift_range = redshift_range
        self.run = run
        self.analysis_function = analysis_function
        self.n_posterior = n_posterior
        self.full_catalogue = full_catalogue
        self.load_indices = load_indices
        self.index_list = index_list

        self.n_objects = len(self.IDs)
        self.done = np.zeros(self.IDs.shape[0]).astype(bool)
        self.cat = None
        self.vars = None

        (self.out_path / "pipes" / "posterior" / self.run).mkdir(
            exist_ok=True, parents=True
        )
        if self.overwrite:
            for file in (self.out_path / "pipes" / "posterior" / self.run).iterdir():
                file.unlink()

        self._setup_vars()
        col_names = self._setup_colnames()

        self.check_errs_dict(min_uncertainties)

        if parallel == 0:
            self.cat = Table(
                names=col_names, data=np.zeros((self.n_objects, len(self.params) + 1))
            )

            for i, id_i in enumerate(tqdm(self.IDs)):

                self.cat[i] = self._fit_object(id_i)

        else:

            if parallel < 0:
                n_proc = cpu_count()
            else:
                n_proc = parallel

            print(
                f"Fitting {self.n_objects} objects using {n_proc} process{"es" if n_proc>1 else ""}."
            )

            inputs = zip(
                self.IDs,
                cat_filt_list if vary_filt_list else repeat(cat_filt_list),
                (
                    self.redshifts
                    if hasattr(self.redshifts, "__iter__")
                    else repeat(self.redshifts if self.redshifts is not None else None)
                ),
            )
            mapped_fn = partial(
                self._fit_object,
                load_data=self.load_data,
                spectrum_exists=self.spectrum_exists,
                photometry_exists=self.photometry_exists,
                load_indices=self.load_indices,
                posterior_parent_path=self.out_path / "pipes" / "posterior" / self.run,
                redshift_range=self.redshift_range,
                fit_instructions=self.fit_instructions,
                out_path=self.out_path,
                run=self.run,
                n_posterior=self.n_posterior,
                vars=self.vars,
                full_catalogue=self.full_catalogue,
                atlas_name=self.atlas_name,
                atlas_shape=self.atlas_shape,
                param_samples_name=self.param_samples_name,
                param_samples_shape=self.param_samples_shape,
                rng=self.rng,
                params=self.params,
            )
            with Pool(processes=n_proc) as pool:
                fit_data = pool.starmap_async(
                    mapped_fn, tqdm(inputs, total=len(self.IDs))
                )
                fit_data.wait()
            self.cat = Table(
                names=col_names,
                data=fit_data.get(),
            )

        self.cat.write(self.out_path / f"{self.run}.fits", overwrite=self.overwrite)
