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
            chisq[i] += pow(models[i,j]-obs[j], 2)*inv_sig_sq[j]

    return np.asarray(chisq)
