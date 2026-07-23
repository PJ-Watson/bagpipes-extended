"""An extension to `bagpipes.fit_catalogue`."""

from __future__ import absolute_import, division, print_function

import copy
import os
from collections.abc import Callable
from glob import glob

import numpy as np
import pandas as pd
from astropy.table import Table
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

import bagpipes
from bagpipes import utils
from bagpipes.catalogue import fit_catalogue as bagpipes_fit_catalogue
from bagpipes.input.galaxy import galaxy

from bagpipes_extended.sed.continuity_varied_z import contvz
from bagpipes_extended.sed.galaxies import FitObj, ObsGalaxy
from bagpipes_extended.sed.plotting import mujy_summary_plot

bagpipes.models.star_formation_history.contvz = contvz


class fit_catalogue(bagpipes_fit_catalogue):
    """
    Fit a model to a catalogue of galaxies. Modified slightly.

    Parameters
    ----------

    IDs : list
        A list of ID numbers for galaxies in the catalogue.

    fit_instructions : dict
        A dictionary containing the details of the model to be fitted to
        the data.

    load_data : function
        Function which takes ID as an argument and returns the model
        spectrum and photometry. Spectrum should come first and be an
        array with a column of wavelengths in Angstroms, a column of
        fluxes in erg/s/cm^2/A and a column of flux errors in the same
        units. Photometry should come second and be an array with a
        column of fluxes in microjanskys and a column of flux errors
        in the same units.

    spectrum_exists : bool, optional
        If the objects do not have spectroscopic data set this to False.
        In this case, load_data should only return photometry.

    photometry_exists : bool, optional
        If the objects do not have photometric data set this to False.
        In this case, load_data should only return a spectrum.

    make_plots : bool, optional
        Whether to make output plots for each object.

    cat_filt_list : list, optional
        The filt_list, or list of filt_lists for the catalogue.

    vary_filt_list : bool, optional
        If True, changes the filter list for each object. When True,
        each entry in cat_filt_list is expected to be a different
        filt_list corresponding to each object in the catalogue.

    redshifts : list, optional
        List of values for the redshift for each object to be fixed to.

    redshift_sigma : float, optional
        If this is set, the redshift for each object will be assigned a
        Gaussian prior centred on the value in redshifts with this
        standard deviation. Hard limits will be placed at 3 sigma.

    run : str, optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    analysis_function : function, optional
        Specify some function to be run on each completed fit, must
        take the fit object as its only argument.

    time_calls : bool, optional
        Whether to print information on the average time taken for
        likelihood calls.

    n_posterior : int, optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete for each object. Default 500.

    full_catalogue : bool, optional
        Adds minimum chi-squared values and rest-frame UVJ mags to the
        output catalogue, takes extra time, default False.

    load_indices : function | str, optional
        Load spectral index information for the galaxy. This can either
        be a function which takes the galaxy ID and returns index values
        in the same order as they are defined in index_list, or the str
        `“from_spectrum”`, in which case the code will measure the indices
        from the observed spectrum for the galaxy.

    index_list : list, optional
        A list of dicts containining the definitions for spectral indices.

    track_backlog : bool, optional
        When using `mpi_serial`, report the number of objects waiting to
        be added to the catalogue by the “zero” core that compiles results
        from all the others. High numbers mean cores are waiting around
        doing nothing.

    spec_units : str, optional
        Units of the input spectrum, defaults to ergs s^-1 cm^-2 A^-1
        (`“ergscma”`). Other units (microjanskys; `"mujy"`) will be
        converted to ergscma by default within the class.

    phot_units : str, optional
        Units of the input photometry, defaults to microjanskys, `“mujy”`.
        The photometry will be converted to ergscma by default within the
        class.

    load_line_fluxes : function | str, optional
        Load observed line fluxes for a galaxy. The function should
        return a list of line labels in Cloudy format, as well as an
        array with a column of flux values in erg/s/cm^2/AA and a column
        of corresponding uncertainties in the same units. It is not
        recommended to use this functionality at the same time as loading
        and fitting observed spectroscopic data with the code.

    mujy_plot : bool, optional
        Make an additional plot with the spectrum and photometry in
        `"mujy"`, alongside the SFH and stellar mass histogram.
    """

    def __init__(
        self,
        IDs: list,
        fit_instructions: dict,
        load_data: Callable[[str], ArrayLike],
        spectrum_exists: bool = True,
        photometry_exists: bool = True,
        make_plots: bool = False,
        cat_filt_list: ArrayLike | None = None,
        vary_filt_list: bool = False,
        redshifts: ArrayLike | None = None,
        redshift_sigma: ArrayLike | float | None = None,
        run: str = ".",
        analysis_function: Callable[[FitObj], None] | None = None,
        time_calls: bool = False,
        n_posterior: int = 500,
        full_catalogue: bool = False,
        load_indices: Callable[[str], ArrayLike] | str | None = None,
        index_list: list | None = None,
        track_backlog: bool = False,
        spec_units: str = "ergscma",
        phot_units: str = "mujy",
        load_line_fluxes: Callable[[str], ArrayLike] | None = None,
        mujy_plot: bool = False,
    ):

        self.IDs = np.array(IDs).astype(str)
        self.fit_instructions = fit_instructions
        self.load_data = load_data
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.make_plots = make_plots
        self.cat_filt_list = cat_filt_list
        self.vary_filt_list = vary_filt_list
        self.redshifts = redshifts
        self.redshift_sigma = redshift_sigma
        self.run = run
        self.analysis_function = analysis_function
        self.time_calls = time_calls
        self.n_posterior = n_posterior
        self.full_catalogue = full_catalogue
        self.load_indices = load_indices
        self.index_list = index_list
        self.spec_units = spec_units
        self.phot_units = phot_units
        self.load_line_fluxes = load_line_fluxes
        self.mujy_plot = mujy_plot

        self.n_objects = len(self.IDs)
        self.done = np.zeros(self.IDs.shape[0]).astype(bool)
        self.cat = None
        self.vars = None

        if rank == 0:
            utils.make_dirs(run=run)

    def _fit_object(
        self, ID, verbose=False, n_live=400, use_MPI=True, sampler="multinest", pool=1
    ):
        """Fit the specified object and update the catalogue."""

        # Set the correct redshift for this object
        self._set_redshift(ID)

        # Get the correct filt_list for this object
        filt_list = self.cat_filt_list
        if self.vary_filt_list:
            filt_list = self.cat_filt_list[np.argmax(self.IDs == ID)]

        # Load up the observational data for this object
        self.galaxy = ObsGalaxy(
            ID,
            self.load_data,
            filt_list=filt_list,
            spectrum_exists=self.spectrum_exists,
            photometry_exists=self.photometry_exists,
            load_indices=self.load_indices,
            index_list=self.index_list,
            spec_units=self.spec_units,
            phot_units=self.phot_units,
            load_line_fluxes=self.load_line_fluxes,
        )

        # Fit the object
        self.obj_fit = FitObj(
            self.galaxy,
            self.fit_instructions,
            run=self.run,
            time_calls=self.time_calls,
            n_posterior=self.n_posterior,
        )

        self.obj_fit.fit(
            verbose=verbose, n_live=n_live, use_MPI=use_MPI, sampler=sampler, pool=pool
        )

        if rank == 0:
            # print(self.fit_instructions)
            if self.vars is None:
                self._setup_vars()

            if self.cat is None:
                self._setup_catalogue()

            if self.analysis_function is not None:
                self.analysis_function(self.obj_fit)

            # Make plots if necessary
            if self.make_plots:
                self.obj_fit.plot_spectrum_posterior()
                self.obj_fit.plot_corner()
                self.obj_fit.plot_1d_posterior()
                self.obj_fit.plot_sfh_posterior()

                if self.mujy_plot:
                    mujy_summary_plot(self.obj_fit)

                if "calib" in list(self.obj_fit.fitted_model.fit_instructions):
                    self.obj_fit.plot_calibration()

            # Add fitting results to output catalogue
            # Avoid calculating advanced quantities if already done for plots
            if self.full_catalogue and not (
                "spectrum_full" in list(self.obj_fit.posterior.samples)
            ):
                self.obj_fit.posterior.get_advanced_quantities()

            samples = self.obj_fit.posterior.samples

            for v in self.vars:

                if v == "UV_colour":
                    values = samples["uvj"][:, 0] - samples["uvj"][:, 1]

                elif v == "VJ_colour":
                    values = samples["uvj"][:, 1] - samples["uvj"][:, 2]

                else:
                    values = samples[v]

                self.cat.loc[ID, v + "_16"] = np.percentile(values, 16)
                self.cat.loc[ID, v + "_50"] = np.percentile(values, 50)
                self.cat.loc[ID, v + "_84"] = np.percentile(values, 84)

            results = self.obj_fit.results
            self.cat.loc[ID, "log_evidence"] = results["lnz"]
            self.cat.loc[ID, "log_evidence_err"] = results["lnz_err"]

            if self.full_catalogue and self.photometry_exists:

                if self.load_line_fluxes is not None:

                    samples["chisq_tot"] = (
                        samples["chisq_phot"] + samples["chisq_lines"]
                    )

                    min_idx = np.argmin(samples["chisq_tot"])
                    self.cat.loc[ID, "chisq_tot"] = np.min(samples["chisq_tot"])

                    self.cat.loc[ID, "chisq_lines"] = samples["chisq_lines"][min_idx]
                    n_lines = np.sum(self.galaxy.line_fluxes[:, 0] != 0.0)

                    self.cat.loc[ID, "n_lines"] = n_lines

                    self.cat.loc[ID, "chisq_phot"] = samples["chisq_phot"][min_idx]

                else:
                    self.cat.loc[ID, "chisq_phot"] = np.min(samples["chisq_phot"])

                n_bands = np.sum(self.galaxy.photometry[:, 1] != 0.0)

                self.cat.loc[ID, "n_bands"] = n_bands

                if "continuity" in self.fit_instructions:
                    for i, bin_edge_i in enumerate(
                        self.fit_instructions["continuity"].get("bin_edges", [])
                    ):
                        self.cat.loc[ID, f"bin_edge_{i}"] = bin_edge_i
                elif "contvz" in self.fit_instructions:
                    try:
                        med_z = np.percentile(samples["redshift"], 50)
                    except:
                        med_z = self.cat.loc[ID, "input_redshift"]

                    age_univ = np.floor(
                        10**9 * np.interp(med_z, utils.z_array, utils.age_at_z)
                    )
                    bin_edges_low = np.atleast_1d(
                        np.array(
                            self.fit_instructions["contvz"].get("bin_edges_low", [0])
                        )
                    )

                    if self.fit_instructions["contvz"].get("bin_frac_high") is not None:
                        bin_edges_high = (
                            np.atleast_1d(
                                np.array(
                                    [
                                        self.fit_instructions["contvz"].get(
                                            "bin_frac_high"
                                        ),
                                        1.0,
                                    ]
                                )
                            )
                            * age_univ
                            * 10 ** (-6)
                        )
                    else:
                        bin_edges_high = np.atleast_1d(
                            np.array(
                                self.fit_instructions["contvz"].get(
                                    "bin_edges_high", [0]
                                )
                            )
                        ) + age_univ * 10 ** (-6)
                    n_bins = self.fit_instructions["contvz"].get("n_bins", 7)

                    bin_edges = np.concatenate(
                        [
                            bin_edges_low,
                            np.geomspace(
                                bin_edges_low[-1],
                                bin_edges_high[0],
                                num=n_bins
                                - len(bin_edges_low)
                                - len(bin_edges_high)
                                + 3,
                            )[1:-1],
                            bin_edges_high,
                        ]
                    ).astype(int)
                    for i, bin_edge_i in enumerate(bin_edges):
                        self.cat.loc[ID, f"bin_edge_{i}"] = bin_edge_i

    def _setup_vars(self):
        """Set up list of variables to go in the output catalogue."""

        self.vars = copy.copy(self.obj_fit.fitted_model.params)
        self.vars += [
            "stellar_mass",
            "formed_mass",
            "sfr",
            "ssfr",
            "nsfr",
            "mass_weighted_age",
            "mass_weighted_zmet",
            "tform",
            "tquench",
        ]

        if self.full_catalogue:
            self.vars += [
                "UV_colour",
                "VJ_colour",
                "tform10",
                "tform50",
                "tform90",
                "sfr10",
                "sfr100",
            ]

    def _setup_catalogue(self):
        """Set up the initial blank output catalogue."""

        cols = []
        for var in self.vars:
            cols += [var + "_16", var + "_50", var + "_84"]

        cols += ["input_redshift", "log_evidence", "log_evidence_err"]

        if self.full_catalogue and self.photometry_exists:
            if self.load_line_fluxes is not None:
                cols += ["chisq_tot", "chisq_lines", "n_lines"]

            cols += ["chisq_phot", "n_bands"]

            if "continuity" in self.fit_instructions:
                cols += [
                    f"bin_edge_{i}"
                    for i in np.arange(
                        len(self.fit_instructions["continuity"].get("bin_edges", []))
                    )
                ]
            elif "contvz" in self.fit_instructions:
                cols += [
                    f"bin_edge_{i}"
                    for i in np.arange(
                        self.fit_instructions["contvz"].get("n_bins", -1) + 1
                    )
                ]

        self.cat = pd.DataFrame(np.zeros((self.IDs.shape[0], len(cols))), columns=cols)

        self.cat.loc[:, "#ID"] = self.IDs
        self.cat = self.cat[["#ID"] + cols]
        self.cat.index = self.IDs

        if self.redshifts is not None:
            self.cat.loc[:, "input_redshift"] = self.redshifts
