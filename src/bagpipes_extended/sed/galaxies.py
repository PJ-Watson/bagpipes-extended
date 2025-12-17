"""Modified functions for observed and model galaxies."""

import os
from collections.abc import Callable
from copy import deepcopy

import h5py
import numpy as np
from bagpipes import filters, utils
from bagpipes.fitting import fitted_model as bagpipes_fitted_model
from bagpipes.fitting.fit import fit as bagpipes_fit_obj
from bagpipes.fitting.posterior import posterior
from bagpipes.input.galaxy import galaxy as bagpipes_galaxy
from bagpipes.input.spectral_indices import measure_index
from bagpipes.models import model_galaxy as bagpipes_model_galaxy
from numpy.typing import ArrayLike

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    rank = 0
    size = 1


def mod_get_advanced_quantities(self):
    """Calculates advanced derived posterior quantities, these are
    slower because they require the full model spectra. """

    if "spectrum_full" in list(self.samples):
        return

    self.fitted_model._update_model_components(self.samples2d[0, :])
    self.model_galaxy = bagpipes_model_galaxy(self.fitted_model.model_components,
                                        filt_list=self.galaxy.filt_list,
                                        spec_wavs=self.galaxy.spec_wavs,
                                        index_list=self.galaxy.index_list,
                                        spec_units=self.galaxy.out_units,
                                        phot_units=self.galaxy.out_units)

    all_names = ["photometry", "spectrum", "spectrum_full", "uvj",
                    "indices"]

    all_model_keys = dir(self.model_galaxy)
    quantity_names = [q for q in all_names if q in all_model_keys]

    for q in quantity_names:
        size = getattr(self.model_galaxy, q).shape[0]
        self.samples[q] = np.zeros((self.n_samples, size))

    if self.galaxy.photometry_exists:
        self.samples["chisq_phot"] = np.zeros(self.n_samples)

    if self.galaxy.line_labels is not None:
        self.samples["chisq_lines"] = np.zeros(self.n_samples)

    if "dla" in list(self.fitted_model.model_components):
        size = self.model_galaxy.spectrum_full.shape[0]
        self.samples["dla_transmission"] = np.zeros((self.n_samples, size))

    if "dust" in list(self.fitted_model.model_components):
        size = self.model_galaxy.spectrum_full.shape[0]
        self.samples["dust_curve"] = np.zeros((self.n_samples, size))

    if "calib" in list(self.fitted_model.model_components):
        size = self.model_galaxy.spectrum.shape[0]
        self.samples["calib"] = np.zeros((self.n_samples, size))

    if "noise" in list(self.fitted_model.model_components):
        type = self.fitted_model.model_components["noise"]["type"]
        if type.startswith("GP"):
            size = self.model_galaxy.spectrum.shape[0]
            self.samples["noise"] = np.zeros((self.n_samples, size))

    for i in range(self.n_samples):
        param = self.samples2d[self.indices[i], :]
        self.fitted_model._update_model_components(param)
        self.fitted_model.lnlike(param)

        if self.galaxy.photometry_exists:
            self.samples["chisq_phot"][i] = self.fitted_model.chisq_phot

        if self.galaxy.line_labels is not None:
            self.samples["chisq_lines"][i] = self.fitted_model.chisq_lines

        if "dla" in list(self.fitted_model.model_components):
            self.samples["dla_transmission"][i] = self.fitted_model.model_galaxy.dla_trans

        if "dust" in list(self.fitted_model.model_components):
            dust_curve = self.fitted_model.model_galaxy.dust_atten.A_cont
            self.samples["dust_curve"][i] = dust_curve

        if "calib" in list(self.fitted_model.model_components):
            self.samples["calib"][i] = self.fitted_model.calib.model

        if "noise" in list(self.fitted_model.model_components):
            type = self.fitted_model.model_components["noise"]["type"]
            if type.startswith("GP"):
                self.samples["noise"][i] = self.fitted_model.noise.mean()

        for q in quantity_names:
            if q == "spectrum":
                spectrum = getattr(self.fitted_model.model_galaxy, q)[:, 1]
                self.samples[q][i] = spectrum
                continue

            self.samples[q][i] = getattr(self.fitted_model.model_galaxy, q)


posterior.get_advanced_quantities = mod_get_advanced_quantities

class ObsGalaxy(bagpipes_galaxy):
    """
    A container for observational data loaded into Bagpipes.

    Parameters
    ----------
    ID : str
        A string denoting the ID of the object to be loaded. This will be
        passed to load_data.
    load_data : function
        User-defined function which should take ID as an argument and
        return spectroscopic and/or photometric data. Spectroscopy
        should come first and be an array containing a column of
        wavelengths in Angstroms, then a column of fluxes and finally a
        column of flux errors. Photometry should come second and be an
        array containing a column of fluxes and a column of flux errors.
    spec_units : str, optional
        Units of the input spectrum, defaults to ergs s^-1 cm^-2 A^-1,
        "ergscma". Other units (microjanskys; mujy) will be converted to
        ergscma by default within the class (see `out_units`).
    phot_units : str, optional
        Units of the input photometry, defaults to microjanskys, "mujy"
        The photometry will be converted to ergscma by default within the
        class (see `out_units`).
    spectrum_exists : bool, optional
        If you do not have a spectrum for this object, set this to
        False. In this case, load_data should only return photometry.
    photometry_exists : bool, optional
        If you do not have photometry for this object, set this to
        False. In this case, load_data should only return a spectrum.
        the class (see out_units).
    filt_list : list, optional
        A list of paths to filter curve files, which should contain a
        column of wavelengths in angstroms followed by a column of
        transmitted fraction values. Only needed for photometric data.
    out_units : str, optional
        Units to convert the inputs to within the class. Defaults to
        ergs s^-1 cm^-2 A^-1, “ergscma”.
    load_indices : function or str, optional
        Load spectral index information for the galaxy. This can either
        be a function which takes the galaxy ID and returns index values
        in the same order as they are defined in index_list, or the str
        "from_spectrum", in which case the code will measure the indices
        from the observed spectrum for the galaxy.
    index_list : list, optional
        A list of dicts containining definitions for spectral indices.
    index_redshift : float, optional
        Observed redshift for this galaxy. This is only ever used if the
        user requests the code to calculate spectral indices from the
        observed spectrum.
    input_spec_cov_matrix : bool, optional
        If `True`, the input spectroscopy is expected to contain the
        covariance matrix.
    lines_list : list, optional
        A list of emission line names, matching those in Cloudy.This can
        only be used if `spectrum_exists=True` or
        `photometry_exists=True`. If `True`, the last component of
        `load_data` should be a set of line fluxes. By default `False`.
    lines_units : str, optional
        By default, this is `CGS`, i.e. ergs s^-1 cm^-2. If provided as
        SI units (`"SI"`, W m^-2), line fluxes will be converted
        internally to CGS units.
    """

    def __init__(
        self,
        ID: str,
        load_data: Callable[[str], ArrayLike] | str | None = None,
        spec_units: str = "ergscma",
        phot_units: str = "mujy",
        spectrum_exists: bool = True,
        photometry_exists: bool = True,
        filt_list: ArrayLike | None = None,
        out_units: str = "ergscma",
        load_line_fluxes: Callable[[str], ArrayLike] | str | None = None,
        load_indices: Callable[[str], ArrayLike] | str | None = None,
        index_list: list | None = None,
        index_redshift: float | None = None,
        input_spec_cov_matrix: bool = False,
        # lines_list: list | None = None,
        # lines_units: str = "CGS",
    ):
        self.ID = str(ID)
        self.phot_units = phot_units
        self.spec_units = spec_units
        self.out_units = out_units
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.filt_list = filt_list
        self.spec_wavs = None
        self.line_labels = None
        self.index_list = index_list
        self.index_redshift = index_redshift

        # # Attempt to load the data from the load_data function.
        # try:
        #     if not spectrum_exists and not photometry_exists:
        #         raise ValueError("Bagpipes: Object must have some data.")

        #     elif (
        #         (lines_list is not None)
        #         and (not photometry_exists)
        #         and (not spectrum_exists)
        #     ):
        #         raise ValueError("Line fluxes cannot be loaded without other data.")

        #     elif spectrum_exists:
        #         if not photometry_exists and (lines_list is None):
        #             self.spectrum = load_data(self.ID)
        #         elif photometry_exists and (lines_list is None):
        #             self.spectrum, phot_nowavs = load_data(self.ID)
        #         elif not photometry_exists and (lines_list is not None):
        #             self.spectrum, line_fluxes = load_data(self.ID)
        #         else:
        #             self.spectrum, phot_nowavs, line_fluxes = load_data(self.ID)
        #     else:
        #         if photometry_exists and (lines_list is None):
        #             phot_nowavs = load_data(self.ID)
        #         else:
        #             phot_nowavs, line_fluxes = load_data(self.ID)

        # except ValueError:
        #     print(
        #         "load_data did not return expected outputs, did you "
        #         "remember to set photometry_exists/spectrum_exists to "
        #         "false?"
        #     )
        #     raise

        # Attempt to load the data from the load_data function.
        if spectrum_exists or photometry_exists:
            try:
                if not photometry_exists:
                    self.spectrum = load_data(self.ID)

                elif not spectrum_exists:
                    phot_nowavs = load_data(self.ID)

                else:
                    self.spectrum, phot_nowavs = load_data(self.ID)

            except TypeError:
                    print("load_data did not return expected outputs, did you "
                          "forget to set one or both of photometry_exists and "
                          "spectrum_exists to False?")
                    raise

        # If photometry is provided, add filter effective wavelengths to array
        if self.photometry_exists:
            self.filter_set = filters.filter_set(filt_list)
            self.photometry = np.c_[self.filter_set.eff_wavs, phot_nowavs]

        # # If line fluxes provided, associate these with the Cloudy line names
        # if lines_list is not None:
        #     self.line_fluxes = np.array(line_fluxes)
        #     if lines_units == "SI":
        #         line_fluxes *= 1e3
        #     self.line_names = []
        #     for l in lines_list:
        #         if isinstance(l, str):
        #             self.line_names.append([l])
        #         else:
        #             self.line_names.append(l)
        #     assert self.line_fluxes.shape[0] == len(
        #         self.line_names
        #     ), "Number of emission line names does not match the number of line fluxes."
        # else:
        #     self.line_names = None

        # Perform setup in the case of separate covariance matrix for spectrum
        if input_spec_cov_matrix:
            self.spec_cov = self.spectrum[1]
            self.spectrum = np.c_[self.spectrum[0], np.sqrt(np.diagonal(self.spec_cov))]

            self.spec_cov_inv = np.linalg.inv(self.spec_cov)
            # self.spec_cov_det = np.linalg.det(self.spec_cov)

        else:
            self.spec_cov = None

        # Perform any unit conversions.
        self._convert_units()

        # Deal with loading any emission line fluxes
        if load_line_fluxes is not None:
            self.line_labels, self.line_fluxes = load_line_fluxes(self.ID)

        # Mask the regions of the spectrum specified in masks/[ID].mask
        if self.spectrum_exists:
            self.spectrum = self._mask(self.spectrum)
            self.spec_wavs = self.spectrum[:, 0]

            # Remove points at the edges of the spectrum with zero flux.
            startn = 0
            while self.spectrum[startn, 1] == 0.0:
                startn += 1

            endn = 0
            while self.spectrum[-endn - 1, 1] == 0.0:
                endn += 1

            if endn == 0:
                self.spectrum = self.spectrum[startn:, :]

            else:
                self.spectrum = self.spectrum[startn:-endn, :]

            self.spec_wavs = self.spectrum[:, 0]

        # Deal with any spectral index calculations.
        if load_indices is not None:
            self.index_names = [ind["name"] for ind in self.index_list]

            if callable(load_indices):
                self.indices = load_indices(self.ID)

            elif load_indices == "from_spectrum":
                self.indices = np.zeros((len(self.index_list), 2))
                for i in range(self.indices.shape[0]):
                    self.indices[i] = measure_index(
                        self.index_list[i], self.spectrum, self.index_redshift
                    )


# class ModelGalaxy(bagpipes_model_galaxy):
#     """
#     Builds model galaxy spectra.

#     Can calculate predictions for spectroscopic and photometric observables.

#     Parameters
#     ----------
#     model_components : dict
#         A dictionary containing information about the model you wish to
#         generate.
#     filt_list : list, optional
#         A list of paths to filter curve files, which should contain a
#         column of wavelengths in angstroms followed by a column of
#         transmitted fraction values. Only required if photometric output
#         is desired.
#     spec_wavs : array, optional
#         An array of wavelengths at which spectral fluxes should be
#         returned. Only required if spectroscopic output is desired.
#     spec_units : str, optional
#         The units the output spectrum will be returned in. Default is
#         "ergscma" for ergs per second per centimetre squared per
#         angstrom, can also be set to "mujy" for microjanskys.
#     phot_units : str, optional
#         The units the output spectrum will be returned in. Default is
#         "ergscma" for ergs per second per centimetre squared per
#         angstrom, can also be set to "mujy" for microjanskys.
#     index_list : list, optional
#         A list of dicts containining definitions for spectral indices.
#     lines_list : list, optional
#         A list of emission line names, matching those in Cloudy.
#     """

#     def __init__(
#         self,
#         model_components : dict,
#         filt_list : list | None = None,
#         spec_wavs : ArrayLike | None =None,
#         spec_units:str="ergscma",
#         phot_units: str="ergscma",
#         index_list: list|None=None,
#         lines_list: list | None = None,
#     ):

#         if (spec_wavs is not None) and (index_list is not None):
#             raise ValueError("Cannot specify both spec_wavs and index_list.")

#         if model_components["redshift"] > config.max_redshift:
#             raise ValueError(
#                 "Bagpipes attempted to create a model with too "
#                 "high redshift. Please increase max_redshift in "
#                 "bagpipes/config.py before making this model."
#             )

#         self.spec_wavs = spec_wavs
#         self.filt_list = filt_list
#         self.spec_units = spec_units
#         self.phot_units = phot_units
#         self.index_list = index_list
#         self.lines_list = lines_list

#         if self.index_list is not None:
#             self.spec_wavs = self._get_index_spec_wavs(model_components)

#         # Create a filter_set object to manage the filter curves.
#         if filt_list is not None:
#             self.filter_set = filters.filter_set(filt_list)

#         # Calculate the optimal wavelength sampling for the model.
#         self.wavelengths = self._get_wavelength_sampling()

#         # Resample the filter curves onto wavelengths.
#         if filt_list is not None:
#             self.filter_set.resample_filter_curves(self.wavelengths)

#         # Set up a filter_set for calculating rest-frame UVJ magnitudes.
#         uvj_filt_list = np.loadtxt(
#             utils.install_dir + "/filters/UVJ.filt_list", dtype="str"
#         )

#         self.uvj_filter_set = filters.filter_set(uvj_filt_list)
#         self.uvj_filter_set.resample_filter_curves(self.wavelengths)

#         # Create relevant physical models.
#         self.sfh = star_formation_history(model_components)
#         self.stellar = stellar(self.wavelengths)
#         self.igm = igm(self.wavelengths)
#         self.nebular = False
#         self.dust_atten = False
#         self.dust_emission = False
#         self.agn = False

#         if "nebular" in list(model_components):
#             if "velshift" not in model_components["nebular"]:
#                 model_components["nebular"]["velshift"] = 0.0

#             self.nebular = nebular(
#                 self.wavelengths, model_components["nebular"]["velshift"]
#             )

#             if "metallicity" in list(model_components["nebular"]):
#                 self.neb_sfh = star_formation_history(model_components)

#         if "dust" in list(model_components):
#             self.dust_emission = dust_emission(self.wavelengths)
#             self.dust_atten = dust_attenuation(
#                 self.wavelengths, model_components["dust"]
#             )

#         if "agn" in list(model_components):
#             self.agn = agn(self.wavelengths)

#         self.update(model_components)


class FittedGalaxy(bagpipes_fitted_model):
    """A modified version of `bagpipes.fit.fitted_model`."""

    def _set_constants(self):
        """Calculate constant factors used in the lnlike function."""

        if self.galaxy.photometry_exists:
            log_error_factors = np.log(2 * np.pi * self.galaxy.photometry[:, 2] ** 2)
            self.K_phot = -0.5 * np.sum(log_error_factors)
            self.inv_sigma_sq_phot = 1.0 / self.galaxy.photometry[:, 2] ** 2

        if self.galaxy.index_list is not None:
            log_error_factors = np.log(2 * np.pi * self.galaxy.indices[:, 1] ** 2)
            self.K_ind = -0.5 * np.sum(log_error_factors)
            self.inv_sigma_sq_ind = 1.0 / self.galaxy.indices[:, 1] ** 2

        if self.galaxy.line_labels is not None:
            log_error_factors = np.log(2 * np.pi * self.galaxy.line_fluxes[:, 1] ** 2)
            self.K_lines = -0.5 * np.sum(log_error_factors)
            self.inv_sigma_sq_lines = 1.0 / self.galaxy.line_fluxes[:, 1] ** 2

    def lnlike(self, x: ArrayLike, ndim: int = 0, nparam: int = 0):
        """
        Return the log-likelihood for a given parameter vector.

        Parameters
        ----------
        x : ArrayLike
            The parameter vector used to update the model components.
        ndim : int, optional
            Unused, by default 0.
        nparam : int, optional
            Unused, by default 0.
        """

        if self.time_calls:
            time0 = time.time()

            if self.n_calls == 0:
                self.wall_time0 = time.time()

        # Update the model_galaxy with the parameters from the sampler.
        self._update_model_components(x)

        if self.model_galaxy is None:
            self.model_galaxy = bagpipes_model_galaxy(
                self.model_components,
                filt_list=self.galaxy.filt_list,
                spec_wavs=self.galaxy.spec_wavs,
                index_list=self.galaxy.index_list,
                spec_units=self.galaxy.out_units,
                phot_units=self.galaxy.out_units,
            )

        self.model_galaxy.update(self.model_components)

        # Return zero likelihood if SFH is older than the universe.
        if self.model_galaxy.sfh.unphysical:
            self.chisq_phot = np.nan
            return -9.99 * 10**99

        lnlike = 0.0

        if self.galaxy.spectrum_exists and self.galaxy.index_list is None:
            lnlike += self._lnlike_spec()

        if self.galaxy.photometry_exists:
            lnlike += self._lnlike_phot()

        if self.galaxy.index_list is not None:
            lnlike += self._lnlike_indices()

        if self.galaxy.line_labels is not None:
            lnlike += self._lnlike_line_fluxes()

        # Return zero likelihood if lnlike = nan (something went wrong).
        if np.isnan(lnlike):
            print("Bagpipes: lnlike was nan, replaced with zero probability.")
            return -9.99 * 10**99

        if not np.isfinite(lnlike):
            print("Bagpipes: lnlike was infinite, replaced with zero probability.")
            return -9.99*10**99

        # Functionality for timing likelihood calls.
        if self.time_calls:
            self.times[self.n_calls] = time.time() - time0
            self.n_calls += 1

            if self.n_calls == 1000:
                self.n_calls = 0
                print("Mean likelihood call time:", np.round(np.mean(self.times), 4))
                print(
                    "Wall time per lnlike call:",
                    np.round((time.time() - self.wall_time0) / 1000.0, 4),
                )

        return lnlike
    
    def _lnlike_line_fluxes(self):
        """ Calculates the log-likelihood for spectral indices. """

        labels = self.galaxy.line_labels
        model_line_fluxes = np.zeros_like(self.inv_sigma_sq_lines)
        for i, line_set in enumerate(labels):
            for l in np.atleast_1d(line_set):
                model_line_fluxes[i] += self.model_galaxy.line_fluxes[l]
        model_line_fluxes = np.array(model_line_fluxes)

        diff = (self.galaxy.line_fluxes[:, 0] - model_line_fluxes)**2
        self.chisq_lines = np.sum(diff*self.inv_sigma_sq_lines)

        return self.K_lines - 0.5*self.chisq_lines

def _lnlike_line_fluxes(self):
    """ Calculates the log-likelihood for spectral indices. """

    labels = self.galaxy.line_labels
    model_line_fluxes = np.zeros_like(self.inv_sigma_sq_lines)
    for i, line_set in enumerate(labels):
        for l in np.atleast_1d(line_set):
            model_line_fluxes[i] += self.model_galaxy.line_fluxes[l]
    model_line_fluxes = np.array(model_line_fluxes)

    diff = (self.galaxy.line_fluxes[:, 0] - model_line_fluxes)**2
    self.chisq_lines = np.sum(diff*self.inv_sigma_sq_lines)

    return self.K_lines - 0.5*self.chisq_lines

bagpipes_fitted_model._lnlike_line_fluxes = _lnlike_line_fluxes


class FitObj(bagpipes_fit_obj):
    """
    Top-level class for fitting models to observational data.

    Interfaces with MultiNest or nautilus to sample from the posterior
    distribution of a fitted_model object. Performs loading and saving of
    results.

    Parameters
    ----------
    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    fit_instructions : dict
        A dictionary containing instructions on the kind of model which
        should be fitted to the data.

    run : str, optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    time_calls : bool, optional
        Whether to print information on the average time taken for
        likelihood calls.

    n_posterior : int, optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete. Default is 500.
    """

    def __init__(
        self, galaxy, fit_instructions, run=".", time_calls=False, n_posterior=500
    ):

        self.run = run
        self.galaxy = galaxy
        self.fit_instructions = deepcopy(fit_instructions)
        self.n_posterior = n_posterior

        # Set up the directory structure for saving outputs.
        if rank == 0:
            utils.make_dirs(run=run)

        # The base name for output files.
        self.fname = "pipes/posterior/" + run + "/" + self.galaxy.ID + "_"

        # A dictionary containing properties of the model to be saved.
        self.results = {}

        # If a posterior file already exists load it.
        if os.path.exists(self.fname[:-1] + ".h5"):
            file = h5py.File(self.fname[:-1] + ".h5", "r")

            self.posterior = posterior(self.galaxy, run=run, n_samples=n_posterior)

            fit_info_str = file.attrs["fit_instructions"]
            fit_info_str = fit_info_str.replace("array", "np.array")
            fit_info_str = fit_info_str.replace("float", "np.float")
            fit_info_str = fit_info_str.replace("np.np.", "np.")
            self.fit_instructions = eval(fit_info_str)

            for k in file.keys():
                self.results[k] = np.array(file[k])
                if np.sum(self.results[k].shape) == 1:
                    self.results[k] = self.results[k][0]

            if rank == 0:
                print("\nResults loaded from " + self.fname[:-1] + ".h5\n")

        # Set up the model which is to be fitted to the data.
        self.fitted_model = FittedGalaxy(
            galaxy, self.fit_instructions, time_calls=time_calls
        )
