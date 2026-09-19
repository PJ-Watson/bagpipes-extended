"""
A module for fitting objects using a grid search sampling of bagpipes.

Parts of the documentation here are copied from
`ACCarnall/bagpipes <https://github.com/ACCarnall/bagpipes>`__, for
clarity, and any use of this module should cite
`Carnall+18 <https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.4379C>`__.
"""

import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count, shared_memory
from multiprocessing.managers import SharedMemoryManager
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Callable, Self

import bagpipes
import h5py
import numpy as np
import psutil
from bagpipes.fitting.prior import dirichlet, prior
from numpy.random import Generator
from numpy.typing import ArrayLike
from tqdm import tqdm

from bagpipes_extended.sed.specgen import BagpipesSpecGenerator

__all__ = ["AtlasGenerator"]


def init_worker(
    shared_init_params_name: str,
    shared_init_params_shape: tuple[int],
    shared_model_atlas_name: str,
    shared_model_atlas_shape: tuple[int],
    fit_instructions: dict,
    model_kwargs: dict,
    prior_args: tuple,
):
    """
    Initialize global objects within an isolated worker process.

    This function attaches the running process to existing system shared
    memory blocks and creates local instances of the
    `~bagpipes.fitting.prior` and
    `~bagpipes_extended.sed.specgen.BagpipesSpecGenerator` objects.

    Parameters
    ----------
    shared_init_params_name : str
        The unique system name for the shared memory block of parameters.
    shared_init_params_shape : tuple of int
        The shape of the parameters array.
    shared_model_atlas_name : str
        The unique system name for the shared memory block of the model
        atlas.
    shared_model_atlas_shape : tuple of int
        The shape of the model atlas.
    fit_instructions : dict
        A dictionary containing the details of the model, as well as any
        constraints and priors on the parameters.
    model_kwargs : dict
        Any additional keyword arguments to pass to
        `~bagpipes_extended.sed.specgen.ExtendedModelGalaxy`.
    prior_args : tuple
        Arguments required to initialise `~bagpipes.fitting.prior`.
    """

    global shm_init_params, shared_init_params
    shm_init_params = shared_memory.SharedMemory(
        name=shared_init_params_name, create=False
    )
    shared_init_params = np.ndarray(
        shared_init_params_shape, dtype=float, buffer=shm_init_params.buf
    )

    global shm_model_atlas, shared_model_atlas
    shm_model_atlas = shared_memory.SharedMemory(
        name=shared_model_atlas_name, create=False
    )
    shared_model_atlas = np.ndarray(
        shared_model_atlas_shape, dtype=float, buffer=shm_model_atlas.buf
    )

    global spec_generator
    spec_generator = BagpipesSpecGenerator(
        fit_instructions=fit_instructions, **model_kwargs
    )

    global prior_inst
    prior_inst = prior(*prior_args)


def worker_process_components(
    arr_idx: int,
) -> None:
    """
    Process a single model galaxy and fill in a row of the shared arrays.

    Transforms the unit cube-sampled parameters to physical parameters,
    and updates the process instance of
    `~bagpipes_extended.sed.specgen.BagpipesSpecGenerator`. The output
    values are placed directly into the shared model atlas.

    Parameters
    ----------
    arr_idx : int
        The row index corresponding to the sample being processed.
    """

    # Transform the unit cube to physical parameters
    param_vector = prior_inst.transform(shared_init_params[arr_idx])

    # Update the shared memory array of parameters
    shared_init_params[arr_idx] = param_vector

    shared_model_atlas[arr_idx] = spec_generator.sample_phot(param_vector)


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

        self.smm = SharedMemoryManager()
        self.smm.start()

        self.fit_instructions = deepcopy(fit_instructions)
        self.model_components = deepcopy(fit_instructions)

        # self._set_constants()
        self._process_fit_instructions()

        self.prior = prior(self.limits, self.pdfs, self.hyper_params)

        self._model_kwargs = {
            "filt_list": filt_list,
            "spec_wavs": spec_wavs,
            "index_list": index_list,
            "spec_units": spec_units,
            "phot_units": phot_units,
        }

        self.param_vectors = None
        self.model_atlas = None

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
                self.shm_init_params.name,
                self.param_vectors.shape,
                self.shm_model_atlas.name,
                self.model_atlas.shape,
                self.fit_instructions,
                self._model_kwargs,
                (self.limits, self.pdfs, self.hyper_params),
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

    def initialise_shared_memory(self, n_samples: int, n_output: int):
        """
        Intitialise the shared memory arrays for multiprocessing.

        Currently constructs one array for the sampled parameter vectors,
        and one to hold the model photometry.

        Parameters
        ----------
        n_samples : int
            The number of models to be generated.
        n_output : int
            The number of output variables, e.g. the number of photometric
            bands and a flag to indicate an unphysical SFH.

        Raises
        ------
        MemoryError
            If the amount of memory requested exceeds the current amount available.
        """

        # Check that there is enough free memory before trying to allocate it

        total_req = (n_samples * (self.ndim + n_output)) * np.dtype(float).itemsize
        total_avail = psutil.virtual_memory().available

        if total_req > total_avail:
            raise MemoryError(
                "The required memory would exceed the amount available. "
                "Try reducing the number of samples."
            )

        cubes = self.rng.random((n_samples, self.ndim), dtype=float)
        self.shm_init_params = self.smm.SharedMemory(size=cubes.nbytes)
        self.param_vectors = np.ndarray(
            cubes.shape, dtype=float, buffer=self.shm_init_params.buf
        )
        self.param_vectors[:] = cubes[:]

        self.shm_model_atlas = self.smm.SharedMemory(
            size=(np.dtype(float).itemsize * n_samples * n_output)
        )
        self.model_atlas = np.ndarray(
            (n_samples, n_output), dtype=float, buffer=self.shm_model_atlas.buf
        )

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
            # store_fn = self.store_photometry
            n_output = len(self._model_kwargs["filt_list"]) + 1
        elif self._model_kwargs["spec_wavs"] is not None:
            raise NotImplementedError(
                "Spectroscopic sampling is currently unsupported."
                "This is more complicated than photometry (see `noise' and `calib' "
                "keys), and may be implemented at a later stage."
            )
        elif self._model_kwargs["index_list"] is not None:
            raise NotImplementedError(
                "Line index sampling is currently unsupported."
                "This seems to be a rarely used part of bagpipes, and is unlikely "
                "to be implemented."
            )

        self.rng = np.random.default_rng(seed=seed)

        if parallel < 0:
            self.n_proc = cpu_count()
        elif parallel == 0:
            self.n_proc = 1
        else:
            self.n_proc = parallel

        print(
            f"Generating {n_samples} samples using {self.n_proc} process"
            f"{"es" if self.n_proc>1 else ""}, "
            f"with {self.ndim} free parameters."
        )

        self.initialise_shared_memory(n_samples, n_output)

        self.initialise_process_pool(self.n_proc)

        # self.process_pool.map(worker_process_components, tqdm(np.arange(n_samples)))
        for _ in tqdm(
            self.process_pool.imap_unordered(
                worker_process_components, np.arange(n_samples)
            ),
            total=n_samples,
            mininterval=0.1,
        ):
            pass

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
