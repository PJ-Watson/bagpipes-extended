"""
Functions and classes for fast generation of bagpipes model spectra.
"""

import warnings
from copy import deepcopy
from functools import partial

import bagpipes
import numpy as np
from bagpipes import config, filters, utils
from bagpipes.input.spectral_indices import measure_index
from bagpipes.models import chemical_enrichment_history
from bagpipes.models import model_galaxy as BagpipesModelGalaxy
from bagpipes.models import star_formation_history as BagpipesSFH
from bagpipes.models.agn_model import agn
from bagpipes.models.dust_attenuation_model import dust_attenuation
from bagpipes.models.dust_emission_model import dust_emission
from bagpipes.models.igm_model import igm
from bagpipes.models.nebular_model import nebular
from bagpipes.models.stellar_model import stellar

# Grizli is much faster, but shouldn't be a required dependency
try:
    from grizli.utils_numba.interp import interp_conserve_c as interp_conserve_flux
except:
    from spectres import spectres

    interp_conserve_flux = partial(spectres, fill=0.0)

from numpy.typing import ArrayLike
from tqdm import tqdm


class ExtendedSFH(BagpipesSFH):
    """
    An extended version of `bagpipes.star_formation_history`.
    """

    def update(self, model_components: dict, fast_spec_only: bool = False):
        """
        Update the SFH.

        Parameters
        ----------
        model_components : dict
            The new components of the model galaxy.
        fast_spec_only : bool, optional
            Only perform the operations necessary to generate a model
            spectrum (i.e. do not calculate derived quantities). By
            default False.
        """

        self.model_components = model_components
        self.redshift = self.model_components["redshift"]

        self.sfh = np.zeros_like(self.ages)  # Star-formation history

        self.unphysical = False
        self.age_of_universe = 10**9 * np.interp(
            self.redshift, utils.z_array, utils.age_at_z
        )

        # Calculate the star-formation history for each of the components.
        for i in range(len(self.components)):

            name = self.components[i]
            func = self.components[i]

            if name not in dir(self):
                func = name[:-1]

            self.component_sfrs[name] = np.zeros_like(self.ages)
            self.component_weights[name] = np.zeros_like(config.age_sampling)

            getattr(self, func)(self.component_sfrs[name], self.model_components[name])

            # Normalise to the correct mass.
            mass_norm = np.sum(self.component_sfrs[name] * self.age_widths)
            desired_mass = 10 ** self.model_components[name]["massformed"]

            self.component_sfrs[name] *= desired_mass / mass_norm
            self.sfh += self.component_sfrs[name]

            # Sum up contributions to each age bin to create SSP weights
            weights = self.component_sfrs[name] * self.age_widths
            self.component_weights[name] = np.histogram(
                self.ages, bins=config.age_bins, weights=weights
            )[0]
        # Check no stars formed before the Big Bang.
        if self.sfh[self.ages > self.age_of_universe].max() > 0.0:
            self.unphysical = True

        # ceh: Chemical enrichment history object
        self.ceh = chemical_enrichment_history(
            self.model_components, self.component_weights
        )

        if not fast_spec_only:
            self._calculate_derived_quantities()


class ExtendedModelGalaxy(BagpipesModelGalaxy):
    """
    An extension of `bagpipes.models.model_galaxy`.

    This class allows a model spectrum to be generated, with one or more
    nebular emission lines excluded.

    Parameters
    ----------
    model_components : dict
        A dictionary containing information about the model you wish to
        generate.
    filt_list : list, optional
        A list of paths to filter curve files, which should contain a
        column of wavelengths in angstroms followed by a column of
        transmitted fraction values. Only required if photometric output
        is desired.
    spec_wavs : array,optional
        An array of wavelengths at which spectral fluxes should be
        returned. Only required of spectroscopic output is desired.
    spec_units : str, optional
        The units the output spectrum will be returned in. Default is
        "ergscma" for ergs per second per centimetre squared per
        angstrom, can also be set to "mujy" for microjanskys.
    phot_units : str, optional
        The units the output photometry will be returned in. Default is
        "ergscma" for ergs per second per centimetre squared per
        angstrom, can also be set to "mujy" for microjanskys.
    index_list : list, optional
        A list of dicts containining definitions for spectral indices.
    """

    def __init__(
        self,
        model_components,
        filt_list=None,
        spec_wavs=None,
        spec_units="ergscma",
        phot_units="ergscma",
        index_list=None,
    ):

        if (spec_wavs is not None) and (index_list is not None):
            raise ValueError("Cannot specify both spec_wavs and index_list.")

        if model_components["redshift"] > config.max_redshift:
            raise ValueError(
                "Bagpipes attempted to create a model with too "
                "high redshift. Please increase max_redshift in "
                "bagpipes/config.py before making this model."
            )

        self.spec_wavs = spec_wavs
        self.filt_list = filt_list
        self.spec_units = spec_units
        self.phot_units = phot_units
        self.index_list = index_list

        if self.index_list is not None:
            self.spec_wavs = self._get_index_spec_wavs(model_components)

        # Create a filter_set object to manage the filter curves.
        if filt_list is not None:
            self.filter_set = filters.filter_set(filt_list)

        # Calculate the optimal wavelength sampling for the model.
        self.wavelengths = self._get_wavelength_sampling()

        # Resample the filter curves onto wavelengths.
        if filt_list is not None:
            self.filter_set.resample_filter_curves(self.wavelengths)

        # Set up a filter_set for calculating rest-frame UVJ magnitudes.
        uvj_filt_list = np.loadtxt(
            utils.install_dir + "/filters/UVJ.filt_list", dtype="str"
        )

        self.uvj_filter_set = filters.filter_set(uvj_filt_list)
        self.uvj_filter_set.resample_filter_curves(self.wavelengths)

        # Create relevant physical models.
        self.sfh = ExtendedSFH(model_components)
        self.stellar = stellar(self.wavelengths)
        self.igm = igm(self.wavelengths)
        self.nebular = False
        self.dust_atten = False
        self.agn_dust_atten = False
        self.dust_emission = False
        self.agn = False

        if "nebular" in list(model_components):
            if "velshift" not in model_components["nebular"]:
                model_components["nebular"]["velshift"] = 0.0

            self.nebular = nebular(
                self.wavelengths, model_components["nebular"]["velshift"]
            )

            if "metallicity" in list(model_components["nebular"]):
                self.neb_sfh = ExtendedSFH(model_components)

        if "dust" in list(model_components):
            self.dust_emission = dust_emission(self.wavelengths)
            self.dust_atten = dust_attenuation(
                self.wavelengths, model_components["dust"]
            )

        if "agn_dust" in list(model_components):
            self.agn_dust_atten = dust_attenuation(
                self.wavelengths, model_components["agn_dust"]
            )

        if "agn" in list(model_components):
            self.agn = agn(self.wavelengths)

        self.update(model_components)

    def update(
        self,
        model_components: dict,
        cont_only: bool = False,
        rm_line: list[str] | None = None,
        fast_spec_only: bool = True,
    ):
        """
        Update the model outputs based on ``model_components``.

        Update the model outputs to reflect new parameter values in
        the model_components dictionary. Note that only the changing of
        numerical values is supported.

        Parameters
        ----------
        model_components : dict
            A dictionary containing information about the model to be
            generated.
        cont_only : bool, optional
            Generate only the continuum spectra, with no nebular emission
            at all. By default ``False``.
        rm_line : list[str] | None, optional
            The names of one or more lines to exclude from the model
            spectrum, based on the `Cloudy <https://www.nublado.org/>`__
            naming convention (see `here
            <https://bagpipes.readthedocs.io/en/latest/model_galaxies.html#getting-observables-line-fluxes>`__
            for more details). By default ``None``.
        fast_spec_only : bool, optional
            Only perform the operations necessary to generate a model
            spectrum (i.e. do not calculate derived quantities). By
            default False.
        """

        self.model_comp = model_components
        self.sfh.update(model_components, fast_spec_only)
        if self.dust_atten:
            self.dust_atten.update(model_components["dust"])
        if self.agn_dust_atten:
            self.agn_dust_atten.update(model_components["agn_dust"])

        # If the SFH is unphysical do not caclulate the full spectrum
        if self.sfh.unphysical:
            warnings.warn(
                "The requested model includes stars which formed "
                "before the Big Bang, no spectrum generated.",
                RuntimeWarning,
            )

            self.spectrum_full = np.zeros_like(self.wavelengths)
            self.uvj = np.zeros(3)

        else:
            self._calculate_full_spectrum(
                model_components,
                cont_only=cont_only,
                rm_line=rm_line,
                fast_spec_only=fast_spec_only,
            )

        if self.spec_wavs is not None:
            self._calculate_spectrum(model_components)

        # Add any AGN component:
        if self.agn:
            self.agn.update(self.model_comp["agn"])
            agn_spec = self.agn.spectrum
            agn_spec *= self.igm.trans(self.model_comp["redshift"])

            if self.agn_dust_atten:
                agn_trans = 10 ** (
                    -self.model_comp["agn_dust"]["Av"]
                    * self.agn_dust_atten.A_cont
                    / 2.5
                )
                agn_spec *= agn_trans

            self.spectrum_full += agn_spec / (1.0 + self.model_comp["redshift"])

            if self.spec_wavs is not None:
                zplus1 = self.model_comp["redshift"] + 1.0
                agn_interp = np.interp(
                    self.spec_wavs,
                    self.wavelengths * zplus1,
                    agn_spec / zplus1,
                    left=0,
                    right=0,
                )

                self.spectrum[:, 1] += agn_interp

        if not fast_spec_only:

            if self.filt_list is not None:
                self._calculate_photometry(model_components["redshift"])

            if not self.sfh.unphysical:
                self._calculate_uvj_mags()

            # Deal with any spectral index calculations.
            if self.index_list is not None:
                self.index_names = [ind["name"] for ind in self.index_list]

                self.indices = np.zeros(len(self.index_list))
                for i in range(self.indices.shape[0]):
                    self.indices[i] = measure_index(
                        self.index_list[i], self.spectrum, model_components["redshift"]
                    )

    def _calculate_full_spectrum(
        self,
        model_comp: dict,
        cont_only: bool = True,
        rm_line: list[str] | str | None = None,
        fast_spec_only: bool = True,
    ):
        """
        Calculate a full model spectrum given a set of model components.

        This method combines the models for the various emission
        and absorption processes to generate the internal full galaxy
        spectrum held within the class. The `_calculate_photometry` and
        `_calculate_spectrum` methods generate observables using this
        internal full spectrum.

        Parameters
        ----------
        model_comp : dict
            A dictionary containing information about the model to be
            generated.
        cont_only : bool, optional
            Generate only the continuum spectra, with no nebular emission
            at all. By default ``False``.
        rm_line : list[str] | None, optional
            The names of one or more lines to exclude from the model
            spectrum, based on the `Cloudy <https://www.nublado.org/>`__
            naming convention (see `here
            <https://bagpipes.readthedocs.io/en/latest/model_galaxies.html#getting-observables-line-fluxes>`__
            for more details). By default ``None``.
        fast_spec_only : bool, optional
            Only perform the operations necessary to generate a model
            spectrum (i.e. do not calculate derived quantities). By
            default False.
        """

        t_bc = 0.01
        fesc = 0.0

        if "t_bc" in list(model_comp):
            t_bc = model_comp["t_bc"]

        spectrum_bc, spectrum = self.stellar.spectrum(self.sfh.ceh.grid, t_bc)
        em_lines = np.zeros(config.line_wavs.shape)

        # keep a copy of original BC spectrum for later (i.e., fesc=100%)
        spectrum_bc_f100 = np.copy(spectrum_bc)

        if self.nebular and not cont_only:
            grid = np.copy(self.sfh.ceh.grid)

            # adding picket fence fesc
            if "fesc" in list(model_comp["nebular"]):
                fesc = model_comp["nebular"]["fesc"]

                if not (0.0 <= fesc <= 1.0):
                    raise ValueError("fesc must be between 0 and 1.")

            if "metallicity" in list(model_comp["nebular"]):
                nebular_metallicity = model_comp["nebular"]["metallicity"]
                neb_comp = deepcopy(model_comp)
                for comp in list(neb_comp):
                    if isinstance(neb_comp[comp], dict):
                        neb_comp[comp]["metallicity"] = nebular_metallicity

                self.neb_sfh.update(neb_comp, fast_spec_only)
                grid = self.neb_sfh.ceh.grid

            em_lines += self.nebular.line_fluxes(
                grid, t_bc, model_comp["nebular"]["logU"]
            )

            spectrum_bc[self.wavelengths < 912.0] *= fesc
            spectrum_bc += self.nebular.spectrum(
                grid, t_bc, model_comp["nebular"]["logU"]
            )

        # Add attenuation due to stellar birth clouds.
        if self.dust_atten:
            dust_flux = 0.0  # Total attenuated flux for energy balance.

            # Add extra attenuation to birth clouds.
            eta = 1.0
            if "eta" in list(model_comp["dust"]):
                eta = model_comp["dust"]["eta"]
                bc_Av_reduced = (eta - 1.0) * model_comp["dust"]["Av"]
                if self.dust_atten.type == "VW07":
                    bc_trans_red = 10 ** (
                        -bc_Av_reduced * self.dust_atten.A_cont_bc / 2.5
                    )
                else:
                    bc_trans_red = 10 ** (-bc_Av_reduced * self.dust_atten.A_cont / 2.5)

                spectrum_bc_dust = spectrum_bc * bc_trans_red
                dust_flux += np.trapezoid(
                    spectrum_bc - spectrum_bc_dust, x=self.wavelengths
                ) * (1.0 - fesc)

                spectrum_bc = spectrum_bc_dust

            # Attenuate emission line fluxes.
            if self.dust_atten.type == "VW07":
                Av = model_comp["dust"]["Av"]
                # Apply birth cloud attenuation first
                em_lines *= 10 ** (-bc_Av_reduced * self.dust_atten.A_line_bc / 2.5)
                # Then apply general ISM attenuation
                em_lines *= 10 ** (-Av * self.dust_atten.A_line_ism / 2.5)

            else:
                bc_Av = eta * model_comp["dust"]["Av"]
                em_lines *= 10 ** (-bc_Av * self.dust_atten.A_line / 2.5)

        # Track fesc on line fluxes before (potentially) adding diff dust
        em_lines = em_lines * (1.0 - fesc)

        # Add attenuation due to the diffuse ISM.
        if self.dust_atten:
            trans = 10 ** (-model_comp["dust"]["Av"] * self.dust_atten.A_cont / 2.5)
            dust_spectrum = spectrum * trans
            dust_spectrum_bc = spectrum_bc * trans

            dust_flux += np.trapezoid(spectrum - dust_spectrum, x=self.wavelengths)
            dust_flux += np.trapezoid(
                spectrum_bc - dust_spectrum_bc, x=self.wavelengths
            ) * (1.0 - fesc)

            spectrum = (
                dust_spectrum
                + dust_spectrum_bc * (1.0 - fesc)
                + spectrum_bc_f100 * fesc
            )

            self.spectrum_bc = (spectrum_bc * trans) * (
                1.0 - fesc
            ) + spectrum_bc_f100 * fesc

            # Add dust emission.
            qpah, umin, gamma = 2.0, 1.0, 0.01
            if "qpah" in list(model_comp["dust"]):
                qpah = model_comp["dust"]["qpah"]

            if "umin" in list(model_comp["dust"]):
                umin = model_comp["dust"]["umin"]

            if "gamma" in list(model_comp["dust"]):
                gamma = model_comp["dust"]["gamma"]

            spectrum += dust_flux * self.dust_emission.spectrum(qpah, umin, gamma)

        # if no diffuse dust is added, just add birth cloud spectrum to spectrum.
        else:
            spectrum += spectrum_bc * (1.0 - fesc) + spectrum_bc_f100 * fesc

        spectrum *= self.igm.trans(model_comp["redshift"])

        if self.dust_atten:
            self.spectrum_bc *= self.igm.trans(model_comp["redshift"])

        if "dla" in list(model_comp):
            if "redshift" in list(model_comp["dla"]):
                wavelengths_DLA_rest = (
                    self.wavelengths
                    * (1.0 + model_comp["redshift"])
                    / (1.0 + model_comp["dla"]["redshift"])
                )
            else:
                wavelengths_DLA_rest = self.wavelengths
            self.dla_trans = dla_trans(
                wavelengths_DLA_rest,
                N_HI=10 ** model_comp["dla"]["logN_HI"],
                T=model_comp["dla"]["T"],
                b_turb=(
                    model_comp["dla"]["b_turb"]
                    if "b_turb" in list(model_comp["dla"])
                    else 0.0
                ),
            )
            spectrum *= self.dla_trans
            if self.dust_atten:
                self.spectrum_bc *= self.dla_trans

        # Convert from luminosity to observed flux at redshift z.
        self.lum_flux = 1.0
        if model_comp["redshift"] > 0.0:
            ldist_cm = (
                3.086
                * 10**24
                * np.interp(
                    model_comp["redshift"],
                    utils.z_array,
                    utils.ldist_at_z,
                    left=0,
                    right=0,
                )
            )

            self.lum_flux = 4 * np.pi * ldist_cm**2

        spectrum /= self.lum_flux * (1.0 + model_comp["redshift"])

        if self.dust_atten:
            self.spectrum_bc /= self.lum_flux * (1.0 + model_comp["redshift"])

        em_lines /= self.lum_flux

        # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
        spectrum *= 3.826 * 10**33

        if self.dust_atten:
            self.spectrum_bc *= 3.826 * 10**33

        em_lines *= 3.826 * 10**33

        self.line_fluxes = dict(zip(config.line_names, em_lines))

        self.spectrum_full = spectrum

    def _calculate_spectrum(self, model_comp):
        """
        This method generates predictions for observed spectroscopy.

        It optionally applies a Gaussian velocity dispersion then
        resamples onto the specified set of observed wavelengths.
        It has been modified to use vacuum wavelengths before the final
        resampling stage.

        """

        zplusone = model_comp["redshift"] + 1.0

        if "veldisp" in list(model_comp):
            vres = 3 * 10**5 / config.R_spec / 2.0
            sigma_pix = model_comp["veldisp"] / vres
            k_size = 4 * int(sigma_pix + 1)
            x_kernel_pix = np.arange(-k_size, k_size + 1)

            kernel = np.exp(-(x_kernel_pix**2) / (2 * sigma_pix**2))
            kernel /= np.trapezoid(kernel)  # Explicitly normalise kernel

            spectrum = np.convolve(self.spectrum_full, kernel, mode="valid")
            redshifted_wavs = zplusone * self.wavelengths[k_size:-k_size]

        else:
            spectrum = self.spectrum_full
            redshifted_wavs = zplusone * self.wavelengths

        if "R_curve" in list(model_comp):
            oversample = 4  # Number of samples per FWHM at resolution R
            new_wavs = self._get_R_curve_wav_sampling(oversample=oversample)

            spectrum = interp_conserve_flux(new_wavs, redshifted_wavs, spectrum)
            redshifted_wavs = new_wavs

            sigma_pix = oversample / 2.35  # sigma width of kernel in pixels
            k_size = 4 * int(sigma_pix + 1)
            x_kernel_pix = np.arange(-k_size, k_size + 1)

            kernel = np.exp(-(x_kernel_pix**2) / (2 * sigma_pix**2))
            kernel /= np.trapezoid(kernel)  # Explicitly normalise kernel

            # Disperse non-uniformly sampled spectrum
            spectrum = np.convolve(spectrum, kernel, mode="valid")
            redshifted_wavs = redshifted_wavs[k_size:-k_size]

        # Added by PJW
        vac_redshifted_wavs = air_to_vac(redshifted_wavs)

        fluxes = interp_conserve_flux(self.spec_wavs, vac_redshifted_wavs, spectrum)

        if self.spec_units == "mujy":
            fluxes /= 10**-29 * 2.9979 * 10**18 / self.spec_wavs**2

        self.spectrum = np.c_[self.spec_wavs, fluxes]


def air_to_vac(wavelength: ArrayLike) -> ArrayLike:
    """
    Convert air to vacuum wavelengths.

    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006.

    TODO: check against most recent specutils conversions.

    Parameters
    ----------
    wavelength : ArrayLike
        The wavelengths in Angstroms.
    """

    sigma2 = (1 / (wavelength / 1e4)) ** 2

    refr = 1 + 1e-6 * (287.6155 + 1.62887 * sigma2 + 0.01360 * sigma2**2)

    # Only convert above 2000A
    wavelength[wavelength > 2e3] *= refr[wavelength > 2e3]

    return wavelength


class BagpipesSpecGenerator(object):
    """
    Class for generating spectra using bagpipes model galaxies.

    Parameters
    ----------
    fit_instructions : dict
        A dictionary containing information about the model to be
        generated.
    veldisp : float, optional
        The velocity dispersion of the model galaxy in km/s. By default
        ``veldisp=500``.
    spec_wavs : np.ndarray[float] | None, optional
        The wavelengths onto which the spectrum will be resampled (in
        Angstroms), by default ``np.arange(1e4, 2.3e4, 22.5)``.
    **model_kwargs : dict, optional
        Any additional keyword arguments to pass to
        `~niriss_tools.grism.specgen.ExtendedModelGalaxy`.
    """

    def __init__(
        self,
        fit_instructions: dict,
        veldisp: float | None = None,
        spec_wavs: np.ndarray[float] | None = None,
        **model_kwargs,
    ):

        self.fit_instructions = deepcopy(fit_instructions)
        self.model_components = deepcopy(fit_instructions)
        if veldisp is not None:
            self.model_components["veldisp"] = veldisp

        self._process_fit_instructions()

        # comps = self.update_model_components(np.ones(len(self.params), dtype=float))
        # print (f"{comps=}", flush=True)
        self.spec_wavs = spec_wavs
        self.model_kwargs = model_kwargs

        self.model_gal = None
        # self.model_gal = ExtendedModelGalaxy(
        #     comps, spec_wavs=self.spec_wavs, **model_kwargs
        # )

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

    def update_model_components(
        self,
        param: ArrayLike,
    ) -> dict:
        """
        Generate a model object with the current parameters.

        Originally from `bagpipes.fitting.fitted_model`, modified to allow
        for running in parallel (no attributes overwritten).

        Parameters
        ----------
        param : ArrayLike
            The array of parameters to be updated.

        Returns
        -------
        dict
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

    def sample_spec(
        self,
        param_vector: ArrayLike,
        cont_only: bool = False,
        rm_line: list[str] | str | None = None,
        return_line_fluxes: bool = False,
        fast_spec_only: bool = True,
    ) -> ArrayLike:
        """
        Regenerate a model spectrum from a sampled parameter vector.

        Parameters
        ----------
        param_vector : ArrayLike
            The array of parameters to be updated.
        cont_only : bool, optional
            If ``True``, the model spectrum will be generated without any
            nebular emission. By default ``False``.
        rm_line : list[str] | None, optional
            The names of one or more lines to exclude from the model
            spectrum, based on the `Cloudy <https://www.nublado.org/>`__
            naming convention (see `here
            <https://bagpipes.readthedocs.io/en/latest/model_galaxies.html\
#getting-observables-line-fluxes>`__
            for more details). By default ``None``.
        return_line_fluxes : bool, optional
            If ``True``, return the total line flux for all lines named in
            `~bagpipes.config.line_names`. By default ``False``.
        fast_spec_only : bool, optional
            Only perform the operations necessary to generate a model
            spectrum (i.e. do not calculate derived quantities). By
            default ``True``.

        Returns
        -------
        ArrayLike
            A 2D array, containing the wavelengths and corresponding
            fluxes of the modelled galaxy spectrum.
        """

        new_comps = self.update_model_components(param_vector)

        if self.model_gal is None:
            self.model_gal = ExtendedModelGalaxy(
                new_comps, spec_wavs=self.spec_wavs, **self.model_kwargs
            )

        self.model_gal.update(
            new_comps,
            cont_only=cont_only,
            rm_line=rm_line,
            fast_spec_only=fast_spec_only,
        )

        if return_line_fluxes:
            return self.model_gal.spectrum[:, 1], self.model_gal.line_fluxes
        else:
            return self.model_gal.spectrum.T

    def sample_phot(
        self,
        param_vector: ArrayLike,
        return_sfh_unphysical: bool = True,
        fast_spec_only: bool = True,
    ) -> ArrayLike:
        """
        Regenerate a model spectrum from a sampled parameter vector.

        Parameters
        ----------
        param_vector : ArrayLike
            The array of parameters to be updated.
        return_sfh_unphysical : bool, optional
            Return whether the model SFH is unphysical, by default
            ``True``.
        fast_spec_only : bool, optional
            Only perform the operations necessary to generate a model
            spectrum (i.e. do not calculate derived quantities). By
            default ``True``.

        Returns
        -------
        ArrayLike
            A 1D array, containing the photometric fluxes of the modelled
            galaxy spectrum, and (optionally) a flag for whether the SFH
            is unphysical.
        """

        new_comps = self.update_model_components(param_vector)

        if self.model_gal is None:
            self.model_gal = ExtendedModelGalaxy(
                new_comps, spec_wavs=self.spec_wavs, **self.model_kwargs
            )

        self.model_gal.update(
            new_comps,
            fast_spec_only=fast_spec_only,
        )
        if fast_spec_only:
            self.model_gal._calculate_photometry(new_comps["redshift"])

        if return_sfh_unphysical:
            return np.asarray(
                [*self.model_gal.photometry, float(self.model_gal.sfh.unphysical)]
            )
        else:
            return np.asarray([*self.model_gal.photometry])
