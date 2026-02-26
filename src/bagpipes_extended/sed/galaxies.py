"""Modified functions for observed and model galaxies."""

import os
from collections.abc import Callable
from copy import deepcopy

import h5py
import numpy as np
from bagpipes import config, filters, utils
from bagpipes.fitting import fitted_model as bagpipes_fitted_model
from bagpipes.fitting.fit import fit as bagpipes_fit_obj
from bagpipes.fitting.posterior import posterior
from bagpipes.input.galaxy import galaxy as bagpipes_galaxy
from bagpipes.input.spectral_indices import measure_index
from bagpipes.models import model_galaxy as bagpipes_model_galaxy
from bagpipes.models import star_formation_history as bagpipes_star_formation_history
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


def mod_calculate_derived_quantities(self):
    """
    Calculate derived quantities from the star fformation history.

    This extends the bagpipes functionality to include additional
    nth percentile formation times.
    """

    self.stellar_mass = np.log10(np.sum(self.live_frac_grid * self.ceh.grid))
    self.formed_mass = np.log10(np.sum(self.ceh.grid))

    age_mask = self.ages < config.sfr_timescale
    self.sfr = np.sum(self.sfh[age_mask] * self.age_widths[age_mask])
    self.sfr /= self.age_widths[age_mask].sum()

    # ssfr and nsfr: if sfr=0, set as nan to avoid divide by 0 warning
    if self.sfr == 0:
        self.ssfr = np.nan
        self.nsfr = np.nan
    else:
        self.ssfr = np.log10(self.sfr) - self.stellar_mass
        self.nsfr = np.log10(self.sfr * self.age_of_universe) - self.stellar_mass

    self.mass_weighted_age = np.sum(self.sfh * self.age_widths * self.ages)
    self.mass_weighted_age /= np.sum(self.sfh * self.age_widths)

    # Calculate nth percentile formation time
    perc = 90
    cum_sfh = np.cumsum(self.sfh * self.age_widths) / np.sum(self.sfh * self.age_widths)
    self.tform10 = (
        self.age_of_universe
        - self.ages[np.argmin(np.abs(cum_sfh - (100 - 10) / 100.0))]
    ) * 10**-9
    self.tform50 = (
        self.age_of_universe
        - self.ages[np.argmin(np.abs(cum_sfh - (100 - 50) / 100.0))]
    ) * 10**-9
    self.tform90 = (
        self.age_of_universe
        - self.ages[np.argmin(np.abs(cum_sfh - (100 - 90) / 100.0))]
    ) * 10**-9
    # self.tform_percentile = self.ages[np.argmin(np.abs(cum_sfh - (100 - perc)/100.))]  # In years

    self.mass_weighted_zmet = np.sum(self.live_frac_grid * self.ceh.grid, axis=1)
    self.mass_weighted_zmet /= np.sum(self.live_frac_grid * self.ceh.grid)
    self.mass_weighted_zmet *= config.metallicities
    self.mass_weighted_zmet = np.sum(self.mass_weighted_zmet)

    self.tform = self.age_of_universe - self.mass_weighted_age

    self.tform *= 10**-9
    self.mass_weighted_age *= 10**-9

    mass_assembly = np.cumsum(self.sfh[::-1] * self.age_widths[::-1])[::-1]
    tunivs = self.age_of_universe - self.ages
    mean_sfrs = mass_assembly / tunivs
    normed_sfrs = np.zeros_like(self.sfh)
    sf_mask = self.sfh > 0.0
    normed_sfrs[sf_mask] = self.sfh[sf_mask] / mean_sfrs[sf_mask]

    if self.sfr > 0.1 * mean_sfrs[0]:
        self.tquench = 99.0

    else:
        quench_ind = np.argmax(normed_sfrs > 0.1)
        self.tquench = tunivs[quench_ind] * 10**-9


bagpipes_star_formation_history._calculate_derived_quantities = (
    mod_calculate_derived_quantities
)


def mod_get_advanced_quantities(self):
    """
    Calculate advanced derived posterior quantities.

    These are slower because they require the full model spectra.
    """

    if "spectrum_full" in list(self.samples):
        return

    self.fitted_model._update_model_components(self.samples2d[0, :])
    self.model_galaxy = bagpipes_model_galaxy(
        self.fitted_model.model_components,
        filt_list=self.galaxy.filt_list,
        spec_wavs=self.galaxy.spec_wavs,
        index_list=self.galaxy.index_list,
        spec_units=self.galaxy.out_units,
        phot_units=self.galaxy.out_units,
    )

    all_names = ["photometry", "spectrum", "spectrum_full", "uvj", "indices"]

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
            self.samples["dla_transmission"][
                i
            ] = self.fitted_model.model_galaxy.dla_trans

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


def mod_get_basic_quantities(self):
    """
    Calculate basic posterior quantities.

    These are fast as they are derived only from the SFH model, not the
    spectral model.
    """

    if "stellar_mass" in list(self.samples):
        return

    self.fitted_model._update_model_components(self.samples2d[0, :])
    self.sfh = bagpipes_star_formation_history(self.fitted_model.model_components)

    quantity_names = [
        "stellar_mass",
        "formed_mass",
        "sfr",
        "ssfr",
        "nsfr",
        "mass_weighted_age",
        "tform",
        "tquench",
        "tform10",
        "tform50",
        "tform90",
        "mass_weighted_zmet",
    ]

    for q in quantity_names:
        self.samples[q] = np.zeros(self.n_samples)

    self.samples["sfh"] = np.zeros((self.n_samples, self.sfh.ages.shape[0]))

    quantity_names += ["sfh"]

    for i in range(self.n_samples):
        param = self.samples2d[self.indices[i], :]
        self.fitted_model._update_model_components(param)
        self.sfh.update(self.fitted_model.model_components)

        for q in quantity_names:
            self.samples[q][i] = getattr(self.sfh, q)


posterior.get_basic_quantities = mod_get_basic_quantities


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
    load_line_fluxes : function or str, optional
        Load observed line fluxes for a galaxy. The function should
        return a list of line labels in Cloudy format, as well as an
        array with a column of flux values in erg/s/cm^2/AA and a column
        of corresponding uncertainties in the same units. It is not
        recommended to use this functionality at the same time as loading
        and fitting observed spectroscopic data with the code.
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
                print(
                    "load_data did not return expected outputs, did you "
                    "forget to set one or both of photometry_exists and "
                    "spectrum_exists to False?"
                )
                raise

        # If photometry is provided, add filter effective wavelengths to array
        if self.photometry_exists:
            self.filter_set = filters.filter_set(filt_list)
            self.photometry = np.c_[self.filter_set.eff_wavs, phot_nowavs]

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


def _lnlike_line_fluxes(self):
    """Calculates the log-likelihood for spectral indices."""

    labels = self.galaxy.line_labels
    model_line_fluxes = np.zeros_like(self.inv_sigma_sq_lines)
    for i, line_set in enumerate(labels):
        for l in np.atleast_1d(line_set):
            model_line_fluxes[i] += self.model_galaxy.line_fluxes[l]
    model_line_fluxes = np.array(model_line_fluxes)

    diff = (self.galaxy.line_fluxes[:, 0] - model_line_fluxes) ** 2
    self.chisq_lines = np.sum(diff * self.inv_sigma_sq_lines)

    return self.K_lines - 0.5 * self.chisq_lines


bagpipes_fitted_model._lnlike_line_fluxes = _lnlike_line_fluxes


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
            return -9.99 * 10**99

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
