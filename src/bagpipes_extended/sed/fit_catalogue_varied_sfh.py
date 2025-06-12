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

from bagpipes import utils
from bagpipes.catalogue import fit_catalogue as bagpipes_fit_catalogue
from bagpipes.input.galaxy import galaxy

from bagpipes_extended.sed.galaxies import FitObj, ObsGalaxy


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
        IDs: list,
        fit_instructions: dict,
        load_data: Callable[[str], ArrayLike],
        spectrum_exists: bool = True,
        photometry_exists: bool = True,
        make_plots: bool = False,
        cat_filt_list: ArrayLike | None = None,
        vary_filt_list: bool = False,
        redshifts: ArrayLike | None = None,
        redshift_sigma: float = 0.0,
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
        lines_list: list | None = None,
        lines_units: str = "CGS",
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
        self.lines_list = lines_list
        self.lines_units = lines_units

        self.n_objects = len(self.IDs)
        self.done = np.zeros(self.IDs.shape[0]).astype(bool)
        self.cat = None
        self.vars = None

        if rank == 0:
            utils.make_dirs(run=run)

    # def fit(
    #     self,
    #     verbose=False,
    #     n_live=400,
    #     mpi_serial=False,
    #     track_backlog=False,
    #     sampler="multinest",
    #     pool=1,
    # ):
    #     """Run through the catalogue fitting each object.

    #     Parameters
    #     ----------

    #     verbose : bool - optional
    #         Set to True to get progress updates from the sampler.

    #     n_live : int - optional
    #         Number of live points: reducing speeds up the code but may
    #         lead to unreliable results.

    #     mpi_serial : bool - optional
    #         When running through mpirun/mpiexec, the default behaviour
    #         is to fit one object at a time, using all available cores.
    #         When mpi_serial=True, each core will fit different objects.

    #     track_backlog : bool - optional
    #         When using mpi_serial, report the number of objects waiting
    #         to be added to the catalogue by the "zero" core that
    #         compiles results from all the others. High numbers mean
    #         cores are waiting around doing nothing.
    #     """

    #     if rank == 0:
    #         cat_file = "pipes/cats/" + self.run + ".fits"
    #         if os.path.exists(cat_file):
    #             self.cat = Table.read(cat_file).to_pandas()
    #             self.cat.index = self.IDs
    #             self.done = (self.cat.loc[:, "log_evidence"] != 0.0).values

    #     if size > 1 and mpi_serial:
    #         self._fit_mpi_serial(n_live=n_live, track_backlog=track_backlog)
    #         return

    #     for i in range(self.n_objects):

    #         # Check to see if the object has been fitted already
    #         if rank == 0:
    #             obj_done = self.done[i]

    #             for j in range(1, size):
    #                 comm.send(obj_done, dest=j)

    #         else:
    #             obj_done = comm.recv(source=0)

    #         if obj_done:
    #             continue

    #         # If not fit the object and update the output catalogue
    #         self._fit_object(
    #             self.IDs[i], verbose=verbose, n_live=n_live, sampler=sampler, pool=pool
    #         )

    #         self.done[i] = True

    #         # Save the updated output catalogue.
    #         if rank == 0:
    #             save_cat = Table.from_pandas(self.cat)
    #             save_cat.write(
    #                 "pipes/cats/" + self.run + ".fits", format="fits", overwrite=True
    #             )

    #             print(
    #                 "Bagpipes:",
    #                 np.sum(self.done),
    #                 "out of",
    #                 self.done.shape[0],
    #                 "objects completed.",
    #             )

    # def _fit_mpi_serial(
    #     self, verbose=False, n_live=400, track_backlog=False, sampler="multinest"
    # ):
    #     """Run through the catalogue fitting multiple objects at once
    #     on different cores."""

    #     self.done = self.done.astype(int)
    #     self.done[self.done == 1] += 1

    #     if rank == 0:  # The 0 process manages others, does no fitting
    #         for i in range(1, size):
    #             if not np.min(self.done):  # give out first IDs to fit
    #                 newID = self.IDs[np.argmin(self.done)]
    #                 comm.send(newID, dest=i)
    #                 self.done[np.argmin(self.done)] += 1

    #             else:  # Alternatively tell process all objects are done
    #                 comm.send(None, dest=i)

    #         if np.min(self.done) == 2:  # If all objects are done end
    #             return

    #         while True:  # Add results to catalogue + distribute new IDs
    #             # Wait for an object to be finished by any process
    #             oldID, done_rank = comm.recv(source=MPI.ANY_SOURCE)
    #             self.done[self.IDs == oldID] += 1  # mark as done

    #             if not np.min(self.done):  # Send new ID to process
    #                 newID = self.IDs[np.argmin(self.done != 0)]
    #                 self.done[self.IDs == newID] += 1  # mark in prep
    #                 comm.send(newID, dest=done_rank)  # send new ID

    #             else:  # Alternatively tell process all objects are done
    #                 comm.send(None, dest=done_rank)

    #             # Load posterior for finished object to update catalogue
    #             self._fit_object(
    #                 oldID, use_MPI=False, verbose=False, n_live=n_live, sampler=sampler
    #             )

    #             save_cat = Table.from_pandas(self.cat)
    #             save_cat.write(
    #                 "pipes/cats/" + self.run + ".fits", format="fits", overwrite=True
    #             )

    #             if track_backlog:
    #                 n_done = len(glob("pipes/posterior/" + self.run + "/*.h5"))
    #                 n_cat = np.sum(self.cat["stellar_mass_50"] > 0.0)
    #                 backlog = n_done - n_cat

    #                 print(
    #                     "Bagpipes:",
    #                     np.sum(self.done == 2),
    #                     "out of",
    #                     self.done.shape[0],
    #                     "objects completed.",
    #                     "Backlog:",
    #                     backlog,
    #                     "/",
    #                     size - 1,
    #                     "cores",
    #                 )
    #             else:
    #                 print(
    #                     "Bagpipes:",
    #                     np.sum(self.done == 2),
    #                     "out of",
    #                     self.done.shape[0],
    #                     "objects completed.",
    #                 )

    #             if np.min(self.done) == 2:  # if all objects done end
    #                 return

    #     else:  # All ranks other than 0 fit objects as directed by 0
    #         while True:
    #             ID = comm.recv(source=0)  # receive new ID to fit

    #             if ID is None:  # If no new ID is given then end
    #                 return

    #             self.n_posterior = 5  # hacky, these don't get used
    #             self._fit_object(
    #                 ID, use_MPI=False, verbose=False, n_live=n_live, sampler=sampler
    #             )

    #             comm.send([ID, rank], dest=0)  # Tell 0 object is done

    # def _set_redshift(self, ID):
    #     """Sets the corrrect redshift (range) in self.fit_instructions
    #     for the object being fitted."""

    #     if self.redshifts is not None:
    #         ind = np.argmax(self.IDs == ID)

    #         if self.redshift_sigma > 0.:
    #             z = self.redshifts[ind]
    #             sig = self.redshift_sigma
    #             self.fit_instructions["redshift_prior"] = "Gaussian"
    #             self.fit_instructions["redshift_prior_mu"] = z
    #             self.fit_instructions["redshift_prior_sigma"] = sig
    #             self.fit_instructions["redshift"] = (z - 3*sig, z + 3*sig)

    #         else:
    #             self.fit_instructions["redshift"] = self.redshifts[ind]

    #         # self.fit_instructions = generate_fit_params(
    #         #     self.redshifts[ind],
    #         #     z_range=0,
    #         # )

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
            lines_list=self.lines_list,
            lines_units=self.lines_units,
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
            print(self.fit_instructions)
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

                if "calib" in list(self.obj_fit.fitted_model.fit_instructions):
                    self.obj_fit.plot_calibration()

            # Add fitting results to output catalogue
            if self.full_catalogue:
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
                self.cat.loc[ID, "chisq_phot"] = np.min(samples["chisq_phot"])
                n_bands = np.sum(self.galaxy.photometry[:, 1] != 0.0)
                self.cat.loc[ID, "n_bands"] = n_bands
                if "continuity" in self.fit_instructions:
                    for i, bin_edge_i in enumerate(
                        self.fit_instructions["continuity"].get("bin_edges", [])
                    ):
                        self.cat.loc[ID, f"bin_edge_{i}"] = bin_edge_i
                elif "continuity_varied_z" in self.fit_instructions:
                    try:
                        med_z = np.percentile(samples["redshift"], 50)
                    except:
                        med_z = self.cat.loc[ID, "input_redshift"]

                    age_univ = 10**9 * np.interp(med_z, utils.z_array, utils.age_at_z)
                    bin_edges_low = np.array(
                        self.fit_instructions["continuity_varied_z"].get(
                            "bin_edges_low", [0]
                        )
                    )
                    bin_edges_high = np.array(
                        self.fit_instructions["continuity_varied_z"].get(
                            "bin_edges_high", [age_univ * 10 ** (-6)]
                        )
                    )
                    n_bins = self.fit_instructions["continuity_varied_z"].get(
                        "n_bins", 7
                    )

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
            self.vars += ["UV_colour", "VJ_colour"]

    def _setup_catalogue(self):
        """Set up the initial blank output catalogue."""

        cols = ["#ID"]
        for var in self.vars:
            cols += [var + "_16", var + "_50", var + "_84"]

        cols += ["input_redshift", "log_evidence", "log_evidence_err"]

        if self.full_catalogue and self.photometry_exists:
            cols += ["chisq_phot", "n_bands"]

            if "continuity" in self.fit_instructions:
                cols += [
                    f"bin_edge_{i}"
                    for i in np.arange(
                        len(self.fit_instructions["continuity"].get("bin_edges", []))
                    )
                ]
            elif "continuity_varied_z" in self.fit_instructions:
                cols += [
                    f"bin_edge_{i}"
                    for i in np.arange(
                        self.fit_instructions["continuity_varied_z"].get("n_bins", -1)
                        + 1
                    )
                ]

        self.cat = pd.DataFrame(np.zeros((self.IDs.shape[0], len(cols))), columns=cols)

        self.cat.loc[:, "#ID"] = self.IDs
        self.cat.index = self.IDs

        if self.redshifts is not None:
            self.cat.loc[:, "input_redshift"] = self.redshifts
