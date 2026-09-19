"""
Array optimisations to reduce intermediate memory usage.
"""


import numpy as np

cimport cython
cimport numpy as np

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from libc.math cimport pow


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_chisq(
    DTYPE_t[:,::1] models,
    DTYPE_t[::1] obs,
    DTYPE_t[::1] inv_sig_sq,
    DTYPE_t[::1] scaling,
):
    """
    Calculate the chi-squared values for an array of models.

    Parameters
    ----------
    models : 2D `~np.ndarray`
        An array of shape (N, M), for N models, and M observations.
    obs : 1D `~np.ndarray`
        A 1D array containing M observations.
    inv_sig_sq : 1D `~np.ndarray`
        A 1D array containing the inverse squared uncertainties for each
        of the M observations.
    scaling : 1D `~np.ndarray`
        A 1D array containing the scaling factors to be applied to each
        of the N models.

    Returns
    -------
    chisq : 1D `~np.ndarray`
        The calculated chi-squared values, of shape (N,).
    """

    cdef size_t i, j, I, J
    I = models.shape[0]
    J = models.shape[1]

    cdef DTYPE_t[::1] chisq = np.zeros(I, dtype=DTYPE)

    for i in range(I):
        for j in range(J):
            chisq[i] += pow(models[i,j]*scaling[i]-obs[j], 2)*inv_sig_sq[j]

    return np.asarray(chisq)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calc_scaling(
    DTYPE_t[:,::1] models,
    DTYPE_t[::1] obs,
    DTYPE_t[::1] inv_sig_sq,
):
    """
    Calculate the scaling factor for an array of models.

    Analogous to the default scaling in CIGALE, as bagpipes has no concept
    of upper limits.

    Parameters
    ----------
    models : 2D `~np.ndarray`
        An array of shape (N, M), for N models, and M observations.
    obs : 1D `~np.ndarray`
        A 1D array containing M observations.
    inv_sig_sq : 1D `~np.ndarray`
        A 1D array containing the inverse squared uncertainties for each
        of the M observations.

    Returns
    -------
        scaling : 1D `~np.ndarray`
        The scaling factor to match the models to observations, of shape
        (N,).
    """

    I = models.shape[0]
    J = models.shape[1]

    cdef DTYPE_t current_num, current_denom, m_val, obs_weighted

    cdef np.ndarray[DTYPE_t, ndim=1] scaling = np.empty(I, dtype=DTYPE)
    cdef DTYPE_t[:] scaling_view = scaling

    for i in range(I):
        current_num = 0.0
        current_denom = 0.0

        for j in range(J):

            m_val = models[i, j]
            obs_weighted = obs[j] * inv_sig_sq[j]

            current_num += m_val * obs_weighted
            current_denom += (m_val * m_val) * inv_sig_sq[j]

        scaling_view[i] = current_num / current_denom

    return scaling
