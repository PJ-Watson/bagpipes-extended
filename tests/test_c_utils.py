"""
Tests and benchmarks for c_utils.
"""

import time

import numpy as np
import pytest

# Import the compiled Cython functions
from bagpipes_extended.c_utils import array_ops


def numpy_calc_chisq(
    models: np.ndarray, obs: np.ndarray, inv_sig_sq: np.ndarray, scaling: np.ndarray
) -> np.ndarray:
    """
    Pure-NumPy baseline implementation using broadcasting.

    Parameters
    ----------
    models : np.ndarray
        A 2D float64 array of shape (N, M) representing model predictions,
        where N is the number of models and M is the number of observations.
    obs : np.ndarray
        A 1D float64 array of shape (M,) containing the observed values.
    inv_sig_sq : np.ndarray
        A 1D float64 array of shape (M,) containing the inverse variance
        weights (1 / sigma^2) for the observations.
    scaling : np.ndarray
        A 1D float64 array of shape (N,) containing the specific scaling
        factors applied to each model.

    Returns
    -------
    np.ndarray
        A 1D float64 array of shape (N,) containing the calculated
        chi-squared values for each model row.
    """
    # models shape: (N, M), scaling shape: (N,) -> scaled shape: (N, M)
    # obs shape: (M,) -> broadcasting subtracts obs from every row
    scaled_diff = models * scaling[:, np.newaxis] - obs

    # Square the differences and weight by inverse uncertainties
    weighted_sq = (scaled_diff**2) * inv_sig_sq

    # Sum over observations (axis 1) to get shape (N,)
    return np.sum(weighted_sq, axis=1)


def numpy_calc_scaling(
    models: np.ndarray, obs: np.ndarray, inv_sig_sq: np.ndarray
) -> np.ndarray:
    """
    Pure-NumPy baseline implementation using broadcasting.

    Parameters
    ----------
    models : np.ndarray
        A 2D float64 array of shape (N, M) representing model predictions,
        where N is the number of models and M is the number of observations.
    obs : np.ndarray
        A 1D float64 array of shape (M,) containing the observed values.
    inv_sig_sq : np.ndarray
        A 1D float64 array of shape (M,) containing the inverse variance
        weights (1 / sigma^2) for the observations.

    Returns
    -------
    np.ndarray
        A 1D float64 array of shape (N,) containing the analytical
        optimal scaling factors for each model row.
    """
    # obs * inv_sig_sq shape: (M,)
    obs_weighted = obs * inv_sig_sq

    # Matrix-vector multiplication to compute numerator for all N models
    # (N, M) dot (M,) -> (N,)
    num = np.dot(models, obs_weighted)

    # Compute denominator: models^2 * inv_sig_sq summed over axis 1
    denom = np.sum((models**2) * inv_sig_sq, axis=1)

    return num / denom


def numpy_calc_scaling_and_chisq(
    models: np.ndarray, obs: np.ndarray, inv_sig_sq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure-NumPy baseline implementation for combined scaling and chi-squared.

    Parameters
    ----------
    models : np.ndarray
        A 2D float64 array of shape (N, M) representing model predictions,
        where N is the number of models and M is the number of observations.
    obs : np.ndarray
        A 1D float64 array of shape (M,) containing the observed values.
    inv_sig_sq : np.ndarray
        A 1D float64 array of shape (M,) containing the inverse variance
        weights (1 / sigma^2) for the observations.

    Returns
    -------
    scaling : np.ndarray
        A 1D float64 array of shape (N,) containing the analytical
        optimal scaling factors for each model row.
    chisq : np.ndarray
        A 1D float64 array of shape (N,) containing the minimum
        chi-squared values corresponding to the optimal scaling factors.
    """
    scaling = numpy_calc_scaling(models, obs, inv_sig_sq)
    chisq = numpy_calc_chisq(models, obs, inv_sig_sq, scaling)
    return scaling, chisq


# Helper to generate reproducible mock data
def make_mock_data(
    num_models: int, num_obs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper to generate reproducible mock data.

    Parameters
    ----------
    num_models : int
        The number of discrete grid models (rows) to generate.
    num_obs : int
        The number of simulated observational data points (columns) to generate.

    Returns
    -------
    models : np.ndarray
        2D float64 array of shape (num_models, num_obs).
    obs : np.ndarray
        1D float64 array of shape (num_obs,).
    inv_sig_sq : np.ndarray
        1D float64 array of shape (num_obs,).
    scaling : np.ndarray
        1D float64 array of shape (num_models,).
    """
    rng = np.random.default_rng(seed=42)

    models = rng.uniform(0.1, 10.0, size=(num_models, num_obs)).astype(np.float64)
    obs = rng.uniform(0.1, 10.0, size=num_obs).astype(np.float64)
    inv_sig_sq = rng.uniform(0.5, 2.0, size=num_obs).astype(np.float64)
    scaling = rng.uniform(0.8, 1.2, size=num_models).astype(np.float64)

    return models, obs, inv_sig_sq, scaling


class TestEdgeCasesAndValidation:
    """
    Validation and robust boundary test suite for `c_utils`.

    This suite ensures analytical and floating-point parity between the
    memory-optimised Cython implementation
    (`bagpipes_extended.c_utils.array_ops`) and the vectorized pure-NumPy
    reference baselines.
    """

    @pytest.fixture
    def standard_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide a repeatable mock data array for standard test validation.

        Returns
        -------
        tuple of np.ndarray
            A 4-element tuple containing:
            - models (np.ndarray): 2D array of shape (100, 500)
            representing model data.
            - obs (np.ndarray): 1D array of shape (500,) representing
            observations.
            - inv_sig_sq (np.ndarray): 1D array of shape (500,)
            representing inverse variance weights.
            - scaling (np.ndarray): 1D array of shape (100,) representing
            model scale factors.
        """
        return make_mock_data(100, 500)

    def test_validation_calc_chisq(
        self, standard_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """
        Verify Cython outputs match NumPy within floating-point tolerance.

        Parameters
        ----------
        standard_data : tuple of np.ndarray
            The injected fixture containing the evaluation matrices:
            `(models, obs, inv_sig_sq, scaling)`.

        Raises
        ------
        AssertionError
            If the output arrays diverge from the baseline beyond the
            absolute or relative floor limit of 1e-12.
        """
        models, obs, inv_sig_sq, scaling = standard_data
        expected = numpy_calc_chisq(models, obs, inv_sig_sq, scaling)
        actual = array_ops.calc_chisq(models, obs, inv_sig_sq, scaling)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_validation_calc_scaling(
        self, standard_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """
        Verify Cython outputs match NumPy within floating-point tolerance.

        Parameters
        ----------
        standard_data : tuple of np.ndarray
            The injected fixture containing the evaluation matrices:
            `(models, obs, inv_sig_sq, scaling)`.

        Raises
        ------
        AssertionError
            If the output arrays diverge from the baseline beyond the
            absolute or relative floor limit of 1e-12.
        """
        models, obs, inv_sig_sq, _ = standard_data
        expected = numpy_calc_scaling(models, obs, inv_sig_sq)
        actual = array_ops.calc_scaling(models, obs, inv_sig_sq)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_validation_calc_scaling_and_chisq(
        self, standard_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """
        Verify fused Cython outputs match NumPy within floating-point tolerance.

        Parameters
        ----------
        standard_data : tuple of np.ndarray
            The injected fixture containing the evaluation matrices:
            `(models, obs, inv_sig_sq, scaling)`.

        Raises
        ------
        AssertionError
            If either the scaling or chi-squared arrays diverge from the
            baseline beyond the absolute or relative floor limit of 1e-12.
        """
        models, obs, inv_sig_sq, _ = standard_data
        expected_scaling, expected_chisq = numpy_calc_scaling_and_chisq(
            models, obs, inv_sig_sq
        )
        actual_scaling, actual_chisq = array_ops.calc_scaling_and_chisq(
            models, obs, inv_sig_sq
        )

    def test_equivalence_with_consecutive_cython_calls(
        self, standard_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """
        Verify fused Cython function matches consecutive Cython calls.

        Ensures that `calc_scaling_and_chisq` produces identical scaling
        factors and chi-squared values to calling `calc_scaling` followed
        by `calc_chisq`.

        Parameters
        ----------
        standard_data : tuple of np.ndarray
            The injected fixture containing the evaluation matrices:
            `(models, obs, inv_sig_sq, scaling)`.

        Raises
        ------
        AssertionError
            If the fused outputs diverge from the consecutive Cython outputs
            beyond the absolute or relative floor limit of 1e-12.
        """
        models, obs, inv_sig_sq, _ = standard_data

        expected_scaling = array_ops.calc_scaling(models, obs, inv_sig_sq)
        expected_chisq = array_ops.calc_chisq(models, obs, inv_sig_sq, expected_scaling)

        actual_scaling, actual_chisq = array_ops.calc_scaling_and_chisq(
            models, obs, inv_sig_sq
        )

        np.testing.assert_allclose(
            actual_scaling, expected_scaling, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(actual_chisq, expected_chisq, rtol=1e-12, atol=1e-12)
