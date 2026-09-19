"""
A module for fitting objects using a grid search sampling of bagpipes.

Parts of the documentation here are copied from
`ACCarnall/bagpipes <https://github.com/ACCarnall/bagpipes>`__, for
clarity, and any use of this module should cite
`Carnall+18 <https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.4379C>`__.
"""

import contextlib
import multiprocessing
import os
from copy import deepcopy
from functools import partial
from itertools import repeat
from multiprocessing import Pool, cpu_count, shared_memory
from multiprocessing.managers import SharedMemoryManager
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Callable, Self

import bagpipes
import h5py
import numpy as np
from astropy.table import Table
from bagpipes.fitting.prior import dirichlet, prior
from numpy.typing import ArrayLike
from tqdm import tqdm

from bagpipes_extended.c_utils import calc_chisq, calc_scaling
from bagpipes_extended.sed.specgen import BagpipesSpecGenerator

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


def init_worker(
    shared_param_samples_name: str,
    shared_param_samples_shape: tuple[int],
    shared_model_atlas_name: str,
    shared_model_atlas_shape: tuple[int],
):
    """
    Initialize global objects within an isolated worker process.

    This function attaches the running process to existing system shared
    memory blocks and creates local instances of the
    `~bagpipes.fitting.prior` and
    `~bagpipes_extended.sed.specgen.BagpipesSpecGenerator` objects.

    Parameters
    ----------
    shared_param_samples_name : str
        The unique system name for the shared memory block of parameters.
    shared_param_samples_shape : tuple of int
        The shape of the parameters array.
    shared_model_atlas_name : str
        The unique system name for the shared memory block of the model
        atlas.
    shared_model_atlas_shape : tuple of int
        The shape of the model atlas.
    """

    global shm_param_samples, shared_param_samples
    shm_param_samples = shared_memory.SharedMemory(
        name=shared_param_samples_name, create=False
    )
    shared_param_samples = np.ndarray(
        shared_param_samples_shape, dtype=float, buffer=shm_param_samples.buf
    )

    global shm_model_atlas, shared_model_atlas
    shm_model_atlas = shared_memory.SharedMemory(
        name=shared_model_atlas_name, create=False
    )
    shared_model_atlas = np.ndarray(
        shared_model_atlas_shape, dtype=float, buffer=shm_model_atlas.buf
    )


def fit_single(
    galaxy: bagpipes.galaxy,
    z_range: ArrayLike | None = None,
    n_posterior: int = 500,
    rng_seed: int | None = None,
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
    n_posterior : int, optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete. Default is 500.
    rng_seed : int, optional
        The seed for the RNG generator.
    params : list, optional
        A list of the parameter names used in the fit.

    Returns
    -------
    dict
        A dictionary containing the results of the fit.
    """
    rng = np.random.default_rng(seed=rng_seed)

    if galaxy.photometry_exists:
        log_error_factors = np.log(2 * np.pi * galaxy.photometry[:, 2] ** 2)
        K_phot = -0.5 * np.sum(log_error_factors)
        inv_sigma_sq_phot = 1.0 / galaxy.photometry[:, 2] ** 2

    if galaxy.index_list is not None:
        log_error_factors = np.log(2 * np.pi * galaxy.indices[:, 1] ** 2)
        K_ind = -0.5 * np.sum(log_error_factors)
        inv_sigma_sq_ind = 1.0 / galaxy.indices[:, 1] ** 2

    if z_range is None:
        scaling = calc_scaling(
            shared_model_atlas,
            np.ascontiguousarray(galaxy.photometry[:, 1]),
            inv_sigma_sq_phot,
        )
        chisq_arr = calc_chisq(
            shared_model_atlas,
            np.ascontiguousarray(galaxy.photometry[:, 1]),
            inv_sigma_sq_phot,
            scaling,
        )
        param2d = shared_param_samples
    else:
        z_idxs = np.searchsorted(
            shared_param_samples[np.asarray(params) == "redshift"].ravel(), z_range
        )
        if z_idxs[0] == shared_param_samples.shape[1] or z_idxs[1] == 0:
            raise ValueError("Redshift range not covered by model atlas.")

        scaling = calc_scaling(
            shared_model_atlas[z_idxs[0] : z_idxs[1]],
            np.ascontiguousarray(galaxy.photometry[:, 1]),
            inv_sigma_sq_phot,
        )
        chisq_arr = calc_chisq(
            shared_model_atlas[z_idxs[0] : z_idxs[1]],
            np.ascontiguousarray(galaxy.photometry[:, 1]),
            inv_sigma_sq_phot,
            scaling,
        )

        param2d = shared_param_samples[:, z_idxs[0] : z_idxs[1]]
    map_idx = np.argmin(chisq_arr)
    lnlike_oned = K_phot - 0.5 * chisq_arr

    param2d = np.concatenate([param2d, np.atleast_2d(lnlike_oned)]).T

    for i, p in enumerate(params):
        if "massformed" in p:
            # Log of the scaling factor since massformed is already log10
            param2d[:, i] += np.log10(scaling)

    weights = np.exp(-0.5 * chisq_arr) / np.nansum(np.exp(-0.5 * chisq_arr))
    finite_weights = np.isfinite(weights)

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
    results["lnlike"] = samples2d[:, -1]
    results["lnz"] = lnlike_oned[map_idx]
    results["lnz_err"] = np.nan

    results["median"] = np.median(samples2d, axis=0)
    results["conf_int"] = np.percentile(results["samples2d"], (16, 84), axis=0)

    return results


def fit_object(
    ID: str,
    filt_list: ArrayLike | None = None,
    z: float | None = None,
    load_data: Callable | None = None,
    posterior_parent_path: Path | None = None,
    out_path: Path | None = None,
    spectrum_exists: bool = False,
    photometry_exists: bool = True,
    load_indices: Callable | str | None = None,
    index_list: list | None = None,
    redshift_range: float = 0.01,
    fit_instructions: dict | None = None,
    run: str = ".",
    n_posterior: int = 500,
    full_catalogue: bool = False,
    vars: list[str] | None = None,
    rng_seed: int | None = None,
    params: list[str] | None = None,
) -> dict:
    """
    Fit a single object from the catalogue.

    The posterior output is saved to the corresponding directory, and
    the row of the results table is returned.

    Parameters
    ----------
    ID : str
        The unique identifier for the galaxy in the catalogue.
    filt_list : list, optional
        A list of paths to filter curve files, which should contain a
        column of wavelengths in angstroms followed by a column of
        transmitted fraction values. Only needed for photometric data.
    z : float, optional
        The central redshift of the object. If ``None`` (default), the
        entire model atlas will be used for the fit.
    load_data : function
        A function which takes ID as an argument and returns the model
        spectrum and photometry. Spectrum should come first and be an
        array with a column of wavelengths in Angstroms, a column of
        fluxes in erg/s/cm^2/A and a column of flux errors in the same
        units. Photometry should come second and be an array with a
        column of fluxes in microjanskys and a column of flux errors
        in the same units.
    posterior_parent_path : Path
        The output directory for the posterior ``*.h5`` files.
    out_path : os.PathLike
        The location to which all output files will be saved. This will be
        created if it does not already exist.
    spectrum_exists : bool, optional
        If the objects do not have spectroscopic data, set this to
        ``False``. In this case, ``load_data()`` should only return
        photometry. By default ``False``.
    photometry_exists : bool, optional
        If the objects do not have photometric data, set this to
        ``False``. In this case, ``load_data()`` should only return a
        spectrum. By default ``True``.
    load_indices : function | str | None, optional
        Load spectral index information for the galaxy. This can
        either be a function which takes the galaxy ``ID`` and returns
        index values in the same order as they are defined in
        ``index_list``, or the str ``"from_spectrum"``, in which case
        the code will measure the indices from the observed spectrum
        for the galaxy. By default ``None``.
    index_list : list | None, optional
        A dict containining definitions for spectral indices,
        by default `None``.
    redshift_range : float, optional
        If this is set, the redshift for each object will be assigned
        a uniform prior centred on the value in ``redshifts``, within
        the range
        :math:`{\\pm 0.5\\times z_{\\rm{range}}}`. By default, this is
        set to 0.01.
    fit_instructions : dict
        A dictionary containing the details of the model, as well as any
        constraints and priors on the parameters.
    run : str, optional
        The subfolder into which outputs will be saved, useful e.g.
        for fitting more than one model configuration to the same
        data. Note that the posterior outputs are already saved into a
        subfolder following the `bagpipes` convention of
        ``{out_path} / pipes``.
    n_posterior : int, optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete. Default is 500.
    full_catalogue : bool, optional
        This adds minimum :math:`{\\chi^2}` values and rest-frame UVJ
        magnitudes to the output catalogue. However, as this requires
        the model spectrum to be calculated, it can significantly
        increase the time taken. By default, ``False``.
    vars : list of strings, optional
        A list of variables to be included in the output catalogue.
    rng_seed : int, optional
        The seed for the RNG generator.
    params : list, optional
        A list of the parameter names used in the fit.

    Returns
    -------
    dict
        The row data for the results table.
    """

    galaxy = bagpipes.galaxy(
        ID,
        load_data,
        filt_list=filt_list,
        spectrum_exists=spectrum_exists,
        photometry_exists=photometry_exists,
        load_indices=load_indices,
        index_list=index_list,
    )

    posterior_path = posterior_parent_path / f"{ID}.h5"

    if not posterior_path.is_file():

        if z is not None:
            z_range = [z - redshift_range / 2, z + redshift_range / 2]
        else:
            z_range = None

        results = fit_single(
            galaxy,
            z_range,
            n_posterior=n_posterior,
            params=params,
            rng_seed=rng_seed + int(ID),
        )

        with h5py.File(posterior_path, "w") as file:

            # This is necessary for converting large arrays to strings
            np.set_printoptions(threshold=10**7)
            file.attrs["fit_instructions"] = str(fit_instructions)
            np.set_printoptions(threshold=10**4)

            for k in results.keys():
                file.create_dataset(k, data=results[k])

    else:
        with h5py.File(posterior_path, "r") as file:
            results = {}
            results["lnz"] = file["lnz"][()]
            results["lnz_err"] = file["lnz_err"][()]

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

        self.smm = SharedMemoryManager()
        self.smm.start()

        self.run = "."
        self.fit_instructions = deepcopy(fit_instructions)
        self.n_posterior = 500

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.out_path = Path(out_path)

        self._process_fit_instructions()

        self.overwrite = overwrite
        self.full_catalogue = False
        self.photometry_exists = False

        # # If a posterior file already exists load it.
        if Path(atlas_path).exists():
            self._load_atlas(atlas_path)

    def __enter__(self) -> Self:
        """
        Enter the runtime context manager framework.

        Returns
        -------
        Self
            The current class instance.
        """
        return self

    def __del__(self) -> None:
        """
        Ensure that the Pool is terminated correctly.
        """

        if hasattr(self, "_process_pool"):
            try:
                self._process_pool.close()
                self._process_pool.join()
            except Exception:
                pass
            finally:
                del self._process_pool

        if hasattr(self, "smm"):
            try:
                self.smm.shutdown()
            except Exception:
                pass
            finally:
                del self.smm

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Exit the runtime context manager framework.

        Cleans up and teardowns background resources safely.

        Parameters
        ----------
        exc_type : type of BaseException or None
            The raised exception class type details if an error occurs.
        exc_value : BaseException or None
            The distinct instance contents for exceptions raised.
        traceback : TracebackType or None
            The low level execution call-stack tracking information maps.
        """
        self.__del__()

    def close(self) -> None:
        """
        A method to explicitly destroy the object.
        """
        self.__del__()

    def initialise_process_pool(self, cpu_count: int) -> None:
        """
        Initialise a pool of processes.

        This is stored as a class attribute, to reduce the overhead of
        creating this each time it is needed.

        Parameters
        ----------
        cpu_count : int
            The number of processes to create.
        """

        self._process_pool = multiprocessing.Pool(
            processes=cpu_count,
            initializer=init_worker,
            initargs=(
                self.shm_param_samples.name,
                self.shared_param_samples.shape,
                self.shm_model_atlas.name,
                self.shared_model_atlas.shape,
            ),
        )

    @property
    def process_pool(self) -> multiprocessing.pool.Pool:
        """The pool of workers for all multiprocessing (`~multiprocessing.Pool`, read-only)."""
        return self._process_pool

    @process_pool.setter
    def process_pool(self, value: None = None):  # numpydoc ignore=GL08
        raise AttributeError(
            "`self.process_pool` cannot be set directly. Initialise this "
            "attribute using `self.initialise_process_pool(cpu_count)` instead."
        )

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
                            if not vv == self.fit_instructions.get(k, {}).get(kk):
                                err_text += f"KEY: {kk}\t ATLAS: {vv}\t "
                                err_text += f"FIT_INSTR: {self.fit_instructions.get(k, {}).get(kk)}\n"

                    else:
                        if not v == self.fit_instructions.get(k):
                            err_text += f"KEY: {k}\t ATLAS: {v}\t "
                            err_text += f"FIT_INSTR: {self.fit_instructions.get(k)}\n"

                raise ValueError(f"Fit instructions do not match:\n{err_text}")

            model_kwargs_str = file.attrs["model_kwargs"]
            model_kwargs_str = model_kwargs_str.replace("array", "np.array")
            model_kwargs_str = model_kwargs_str.replace("float", "np.float")

            self.model_kwargs = eval(model_kwargs_str)

            model_atlas = np.array(file["model_atlas"])
            physical_idx = (model_atlas[:, -1] != 1.0) & (
                np.isfinite(np.sum(model_atlas, axis=-1))
            )
            model_atlas = model_atlas[physical_idx][:, :-1]

            param_samples = np.empty((len(self.params), model_atlas.shape[0]))
            for i, k in enumerate(self.params):
                param_samples[i] = np.array(file[k][physical_idx])

            if "redshift" in self.params:
                z_idx = np.argsort(np.array(file["redshift"][physical_idx]))
                model_atlas = model_atlas[z_idx]
                param_samples = param_samples[:, z_idx]

            self.shm_model_atlas = self.smm.SharedMemory(size=model_atlas.nbytes)
            self.shared_model_atlas = np.ndarray(
                model_atlas.shape,
                dtype=model_atlas.dtype,
                buffer=self.shm_model_atlas.buf,
            )
            self.shared_model_atlas[:] = model_atlas[:]

            self.shm_param_samples = self.smm.SharedMemory(size=param_samples.nbytes)
            self.shared_param_samples = np.ndarray(
                param_samples.shape,
                dtype=param_samples.dtype,
                buffer=self.shm_param_samples.buf,
            )
            self.shared_param_samples[:] = param_samples[:]

            self.n_samples = self.shared_model_atlas.shape[0]

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
        min_uncertainties: dict | None = None,
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
            If this is set, the redshift for each object will be assigned
            a uniform prior centred on the value in ``redshifts``, within
            the range
            :math:`{\\pm 0.5\\times z_{\\rm{range}}}`. By default, this is
            set to 0.01.
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
        if vary_filt_list:
            assert len(cat_filt_list) == len(
                self.IDs
            ), "A variable filter list must have the same number of elements as the list of IDs."
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

        if parallel < 0:
            n_proc = cpu_count()
        elif parallel == 0:
            n_proc = 1
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
            fit_object,
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
            rng_seed=self.seed,
            params=self.params,
        )

        self.initialise_process_pool(n_proc)

        fit_data = self.process_pool.starmap(
            mapped_fn, tqdm(inputs, total=len(self.IDs))
        )
        self.cat = Table(
            names=col_names,
            data=fit_data,
        )

        self.cat.write(self.out_path / f"{self.run}.fits", overwrite=self.overwrite)
