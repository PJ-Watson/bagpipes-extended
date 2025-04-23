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
) -> ArrayLike:
    """
    Load photometry from a catalogue to bagpipes-formatted data.

    The output fluxes and uncertainties are scaled to microJanskys.

    Parameters
    ----------
    str_id : str
        The ID of the object in the photometric catalogue to fit.
    phot_cat : os.PathLike
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

    Returns
    -------
    ArrayLike
        An Nx2 array containing the fluxes and their associated
        uncertainties in all photometric bands.
    """

    if not isinstance(phot_cat, Table):
        phot_cat = Table.read(phot_cat, hdu=cat_hdu_index)

    row_idx = (phot_cat[id_colname] == str_id).nonzero()[0][0]

    fluxes = []
    errs = []
    for c in phot_cat.colnames:
        if "_sci" in c.lower():
            fluxes.append(phot_cat[c][row_idx])
        elif "_var" in c.lower():
            errs.append(np.sqrt(phot_cat[c][row_idx]))
        elif "_err" in c.lower():
            errs.append(phot_cat[c][row_idx])
    if zeropoint == 28.9:
        flux_scale = 1e-2
    else:
        flux_scale = 10 ** ((8.9 - zeropoint) / 2.5 + 6)

    flux = np.asarray(fluxes) * flux_scale
    flux_err = np.asarray(errs) * flux_scale
    flux_err = np.sqrt(flux_err**2 + (0.1 * flux) ** 2)
    flux = flux.copy()
    flux_err = flux_err.copy()
    bad_values = (
        ~np.isfinite(flux) | (flux <= 0) | ~np.isfinite(flux_err) | (flux_err <= 0)
    )
    flux[bad_values] = 0.0
    flux_err[bad_values] = 1e30
    return np.c_[flux, flux_err]


def generate_fit_params(
    obj_z: float | ArrayLike,
    z_range: float = 0.01,
    num_age_bins: int = 5,
    min_age_bin: float = 30,
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

    Returns
    -------
    dict
        A dictionary containing the necessary fit parameters for
        ``bagpipes``.
    """

    fit_params = {}
    if (z_range == 0.0) or (type(obj_z) is ArrayLike):
        fit_params["redshift"] = obj_z
    else:
        fit_params["redshift"] = (obj_z - z_range / 2, obj_z + z_range / 2)

    from astropy.cosmology import FlatLambdaCDM

    # Set up necessary variables for cosmological calculations.
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    age_at_z = cosmo.age(np.nanmax(fit_params["redshift"])).value

    age_bins = np.geomspace(min_age_bin, age_at_z * 1e3, num=num_age_bins)

    age_bins = np.insert(age_bins, 0, 0.0)

    continuity = {
        "massformed": (3.0, 11.0),
        "metallicity": (0.0, 3.0),
        "metallicity_prior_mu": 1.0,
        "metallicity_prior_sigma": 0.5,
        "bin_edges": age_bins.tolist(),
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

    fit_params["dust"] = {
        "type": "Cardelli",
        "Av": (0.0, 2.0),
        "eta": 2.0,
    }
    fit_params["nebular"] = {"logU": (-3.5, -2.0)}
    fit_params["t_bc"] = 0.02

    return fit_params
