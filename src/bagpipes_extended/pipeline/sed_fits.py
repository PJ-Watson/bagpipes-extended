"""
Functions and classes for fitting SEDs.
"""

from os import PathLike

import numpy as np
from astropy.table import Table
from numpy.typing import ArrayLike

__all__ = ["load_photom_bagpipes", "generate_fit_params"]


def load_photom_bagpipes(
    str_id: str,
    phot_cat: Table | PathLike,
    id_colname: str = "bin_id",
    zeropoint: float = 28.9,
    cat_hdu_index: int | str = 0,
    extra_frac_err: float = 0.1,
    line_mapping: dict = None,
    line_cat: Table | PathLike = None,
    sci_suffix: str = "_sci",
    var_suffix: str = "_var",
    err_suffix: str = "_err",
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """
    Load photometry from a catalogue to bagpipes-formatted data.

    The output fluxes and uncertainties are scaled to microJanskys.

    Parameters
    ----------
    str_id : str
        The ID of the object in the photometric catalogue to fit.
    phot_cat : Table | os.PathLike
        The location of the photometric catalogue.
    id_colname : str, optional
        The name of the column containing ``str_id``, by default
        ``"bin_id"``.
    zeropoint : float, optional
        The AB magnitude zeropoint, by default ``28.9``.
    cat_hdu_index : int | str, optional
        The index or name of the HDU containing the photometric catalogue,
        by default ``0``.
    extra_frac_err : float, optional
        An additional fractional error to be added to the photometric
        uncertainties. By default ``extra_frac_err=0.1``, i.e. 10% of the
        measured flux will be added in quadrature to the estimated
        uncertainty.
    line_mapping : dict, optional
        A dictionary mapping the column names of emission line fluxes to
        one or more emission line names as used in CLOUDY, e.g.:
        ``{"o2_3727_3730": ["Blnd  3726.00A", "Blnd  3729.00A"],}``.
    line_cat : Table | os.PathLike, optional
        The name of the table containing the emission line fluxes, if
        different to ``phot_cat``.
    sci_suffix : str, optional
        The suffix for column names containing the flux in filters of
        interest. By default ``"_sci"``.
    var_suffix : str, optional
        The suffix for column names containing the variance of the flux in
        filters of interest. By default ``"_var"``.
    err_suffix : str, optional
        The suffix for column names containing the uncertainty of the flux
        in filters of interest. By default ``"_err"``. If present, columns
        matching ``var_suffix`` will be used instead.

    Returns
    -------
    ArrayLike
        An Nx2 array containing the fluxes and their associated
        uncertainties in all photometric bands.
    """

    if not isinstance(phot_cat, Table):
        phot_cat = Table.read(phot_cat, hdu=cat_hdu_index)

    row_idx = (phot_cat[id_colname].astype(int) == int(str_id)).nonzero()[0][0]

    fluxes = []
    errs = []
    for c in phot_cat.colnames:
        if c.lower().endswith(sci_suffix):
            fluxes.append(phot_cat[c][row_idx])
        elif c.lower().endswith(var_suffix):
            errs.append(np.sqrt(phot_cat[c][row_idx]))
        elif c.lower().endswith(err_suffix):
            errs.append(phot_cat[c][row_idx])

    if zeropoint == 28.9:
        flux_scale = 1e-2
    else:
        flux_scale = 10 ** ((8.9 - zeropoint) / 2.5 + 6)

    flux = np.asarray(fluxes) * flux_scale
    flux_err = np.asarray(errs) * flux_scale
    flux_err = np.sqrt(flux_err**2 + (extra_frac_err * flux) ** 2)
    flux = flux.copy()
    flux_err = flux_err.copy()
    bad_values = (
        ~np.isfinite(flux) | (flux <= 0) | ~np.isfinite(flux_err) | (flux_err <= 0)
    )
    flux[bad_values] = 0.0
    flux_err[bad_values] = 1e30

    if line_mapping is None:
        return np.c_[flux, flux_err]
    else:
        if line_cat is None:
            line_cat = phot_cat
        if not isinstance(line_cat, Table):
            line_cat = Table.read(line_cat, hdu=cat_hdu_index)

        row_idx = (line_cat[id_colname] == int(str_id)).nonzero()[0][0]
        em_lines = []
        em_lines_errs = []
        for l, n in line_mapping.items():
            em_lines.append(line_cat[f"{l}_flux"][row_idx])
            em_lines_errs.append(line_cat[f"{l}_error"][row_idx])
        em_lines = np.asarray(em_lines)
        em_lines_errs = np.asarray(em_lines_errs)
        em_lines_errs = np.sqrt(em_lines_errs**2 + (extra_frac_err * em_lines) ** 2)
        bad_values = (
            ~np.isfinite(em_lines)
            | (em_lines <= 0)
            | ~np.isfinite(em_lines_errs)
            | (em_lines_errs <= 0)
        )
        em_lines[bad_values] = 0.0
        em_lines_errs[bad_values] = 1e30

        return np.c_[flux, flux_err], np.c_[em_lines, em_lines_errs]


def generate_fit_params(
    obj_z: float | ArrayLike,
    z_range: float = 0.01,
    num_age_bins: int = 5,
    min_age_bin: float = 30,
    sfh_type: str = "continuity",
) -> dict:
    """
    Generate a dictionary of fit parameters for Bagpipes.

    This uses the Leja+19 continuity SFH.

    Parameters
    ----------
    obj_z : float | ArrayLike
        The redshift of the object to fit. If a scalar value is passed,
        and ``z_range==0.0``, the object will be fit to a single redshift
        value. If ``z_range!=0.0``, this will be the centre of the
        redshift window. If an array is passed, this explicity sets the
        redshift range to use for fitting.
    z_range : float, optional
        The maximum redshift range to search over, by default 0.01. To fit
        to a single redshift, pass a single value for ``obj_z``, and set
        ``z_range=0.0``. If ``obj_z`` is ``ArrayLike``, this parameter is
        ignored.
    num_age_bins : int, optional
        The number of age bins to fit, each of which will have a constant
        star formation rate following Leja+19. By default ``5`` bins are
        generated.
    min_age_bin : float, optional
        The minimum age to use for the continuity SFH in Myr, i.e. the
        first bin will range from ``(0,min_age_bin)``. By default 30.
    sfh_type : str, optional
        The type of SFH prior to generate. Currently supports `continuity`
        (Leja+19, fixed age bins), `continuity_varied_z` (Leja+19, only
        the youngest age bin is fixed), and `dblplaw`.

    Returns
    -------
    dict
        A dictionary containing the necessary fit parameters for
        ``bagpipes``.
    """

    fit_params = {}
    if z_range == 0.0:
        fit_params["redshift"] = obj_z
    elif hasattr(obj_z, "__len__"):
        fit_params["redshift"] = tuple(obj_z[:2])
    else:
        fit_params["redshift"] = (
            round(obj_z - z_range / 2, 3),
            round(obj_z + z_range / 2, 3),
        )

    from astropy.cosmology import FlatLambdaCDM

    # Set up necessary variables for cosmological calculations.
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    age_at_z = cosmo.age(np.nanmax(fit_params["redshift"])).value

    match sfh_type:

        case "continuity":

            age_bins = np.geomspace(min_age_bin, age_at_z * 1e3, num=num_age_bins)

            age_bins = np.insert(age_bins, 0, 0.0)

            continuity = {
                "massformed": (5.0, 11.0),
                "metallicity": (0.0, 3.0),
                "metallicity_prior_mu": 1.0,
                "metallicity_prior_sigma": 0.5,
                "bin_edges": np.floor(age_bins).tolist(),
            }

            for i in range(1, len(continuity["bin_edges"]) - 1):
                continuity["dsfr" + str(i)] = (-10.0, 10.0)
                continuity["dsfr" + str(i) + "_prior"] = "student_t"
                continuity["dsfr" + str(i) + "_prior_scale"] = (
                    0.5  # Defaults to 0.3 (Leja19), we aim for a broader sample
                )
                continuity["dsfr" + str(i) + "_prior_df"] = (
                    2  # Defaults to this value as in Leja19, but can be set
                )

            fit_params["continuity"] = continuity
        case "continuity_varied_z":

            continuity = {
                "massformed": (5.0, 11.0),
                "metallicity": (0.0, 3.0),
                "metallicity_prior_mu": 1.0,
                "metallicity_prior_sigma": 0.5,
                "bin_edges_low": [0, min_age_bin],
                "n_bins": num_age_bins,
            }

            for i in range(1, num_age_bins):
                continuity["dsfr" + str(i)] = (-10.0, 10.0)
                continuity["dsfr" + str(i) + "_prior"] = "student_t"
                continuity["dsfr" + str(i) + "_prior_scale"] = (
                    0.5  # Defaults to 0.3 (Leja19), we aim for a broader sample
                )
                continuity["dsfr" + str(i) + "_prior_df"] = (
                    2  # Defaults to this value as in Leja19, but can be set
                )

            fit_params["continuity_varied_z"] = continuity

        case "dblplaw":
            fit_params["dblplaw"] = {
                "massformed": (5.0, 11.0),
                "metallicity": (0.0, 3.0),
                "metallicity_prior_mu": 1.0,
                "metallicity_prior_sigma": 0.5,
                "alpha": (0.1, 1000),
                "alpha_prior": "log_10",
                "beta": (0.1, 1000),
                "beta_prior": "log_10",
                "tau": (0.1, np.floor(age_at_z * 1e3) / 1e3),
            }

    fit_params["dust"] = {
        "type": "Cardelli",
        "Av": (0.0, 2.0),
        "eta": 2.0,
    }
    fit_params["nebular"] = {"logU": (-3.5, -2.0)}
    fit_params["t_bc"] = 0.02

    return fit_params
