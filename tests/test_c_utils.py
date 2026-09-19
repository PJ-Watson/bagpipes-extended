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

    def test_edge_case_zeros_and_infs(self) -> None:
        """
        Validate calculations behave uniformly with zeros, NaNs, and inf.

        This test introduces null values across subsets of weights,
        observations, models, and scaling arrays to check if the functions
        return the same output without unexpected behavior.

        Raises
        ------
        AssertionError
            If math operations do not resolve identically across both
            execution paths.
        """
        models, obs, inv_sig_sq, scaling = make_mock_data(10, 50)

        # Inject structural edge cases
        inv_sig_sq[0:5] = 0.0  # Zero-weight observations
        obs[5:10] = 0.0  # Perfectly zero observations
        models[0, :] = 0.0  # A completely zeroed-out model
        scaling[1] = 0.0  # A model with zero scaling factor

        # Test chi-squared equivalence under zero constraints
        expected_chisq = numpy_calc_chisq(models, obs, inv_sig_sq, scaling)
        actual_chisq = array_ops.calc_chisq(models, obs, inv_sig_sq, scaling)
        np.testing.assert_allclose(actual_chisq, expected_chisq, rtol=1e-12, atol=1e-12)

        # Test scaling factor equivalence (Note: avoiding global divide-by-zero bounds)
        expected_scale = numpy_calc_scaling(models, obs, inv_sig_sq)
        actual_scale = array_ops.calc_scaling(models, obs, inv_sig_sq)
        np.testing.assert_allclose(actual_scale, expected_scale, rtol=1e-12, atol=1e-12)

    def test_edge_case_empty_arrays(self) -> None:
        """
        Check that zero-size arrays doesn't segfault.

        Verifies that passing empty or completely unallocated memory views
        (0 models or 0 observations) exits gracefully and returns valid
        empty arrays rather than causing illegal memory accesses.

        Raises
        ------
        AssertionError
            If returned arrays are not empty or do not preserve the
            expected shape signatures.
        """
        models = np.empty((0, 0), dtype=np.float64)
        obs = np.empty((0,), dtype=np.float64)
        inv_sig_sq = np.empty((0,), dtype=np.float64)
        scaling = np.empty((0,), dtype=np.float64)

        # Cython slices must return correctly dimensioned empty arrays safely
        actual_chisq = array_ops.calc_chisq(models, obs, inv_sig_sq, scaling)
        actual_scale = array_ops.calc_scaling(models, obs, inv_sig_sq)

        assert actual_chisq.shape == (0,)
        assert actual_scale.shape == (0,)


@pytest.mark.parametrize(
    "num_models, num_obs",
    [
        (10, 50),  # Small scale
        (100, 500),  # Medium scale
        (1000, 2000),  # Large scale
        (10000, 10),  # Realistic
    ],
    ids=["small", "medium", "large", "realistic"],
)
class TestPerformanceBenchmarks:
    """
    Performance profiling and automated regression verification suite.

    AI:
    This suite uses `pytest-benchmark` to capture detailed statistics on
    execution times and leverages high-precision monotonic timers to
    enforce explicit performance thresholds against the baseline on large
    array profiles.

    PJW:
    Frankly, I put this in purely because I was curious as to how much of
    a speedup there was in using compiled functions for the low-level
    array operations. For realistic data, it seems to be several times
    faster than the numpy implementation above.
    """

    # --- Chi-Squared Benchmarks ---

    def test_bench_numpy_calc_chisq(
        self, benchmark, num_models: int, num_obs: int
    ) -> None:
        """
        Benchmark the pure-NumPy chi-squared baseline.

        Parameters
        ----------
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            The injected testing framework benchmarking object.
        num_models : int
            The number of distinct model variants generated for the test.
        num_obs : int
            The total number of simulated observations per object.
        """
        models, obs, inv_sig_sq, scaling = make_mock_data(num_models, num_obs)
        benchmark(numpy_calc_chisq, models, obs, inv_sig_sq, scaling)

    def test_bench_cython_calc_chisq(
        self, benchmark, num_models: int, num_obs: int
    ) -> None:
        """
        Benchmark the memory-optimized Cython chi-squared variant.

        Parameters
        ----------
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            The injected testing framework benchmarking object.
        num_models : int
            The number of distinct model variants generated for the test.
        num_obs : int
            The total number of simulated observations per object.
        """
        models, obs, inv_sig_sq, scaling = make_mock_data(num_models, num_obs)
        benchmark(array_ops.calc_chisq, models, obs, inv_sig_sq, scaling)

    # --- Scaling Factor Benchmarks ---

    def test_bench_numpy_calc_scaling(
        self, benchmark, num_models: int, num_obs: int
    ) -> None:
        """
        Benchmark the pure-NumPy scaling factor baseline.

        Parameters
        ----------
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            The injected testing framework benchmarking object.
        num_models : int
            The number of distinct model variants generated for the test.
        num_obs : int
            The total number of simulated observations per object.
        """
        models, obs, inv_sig_sq, _ = make_mock_data(num_models, num_obs)
        benchmark(numpy_calc_scaling, models, obs, inv_sig_sq)

    def test_bench_cython_calc_scaling(
        self, benchmark, num_models: int, num_obs: int
    ) -> None:
        """
        Benchmark the memory-optimized Cython scaling variant.

        Parameters
        ----------
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            The injected testing framework benchmarking object.
        num_models : int
            The number of distinct model variants generated for the test.
        num_obs : int
            The total number of simulated observations per object.
        """
        models, obs, inv_sig_sq, _ = make_mock_data(num_models, num_obs)
        benchmark(array_ops.calc_scaling, models, obs, inv_sig_sq)

    # --- Automated Performance Threshold Validation ---

    def test_assert_cython_speed_advantage(self, num_models: int, num_obs: int) -> None:
        """
        Enforce that Cython must outperform NumPy on large arrays.

        For now, we require a 2x improvement in speed for `num_models > 1000`.

        This assertion isolates micro-benchmark overhead using raw
        `time.perf_counter` loops to evaluate scaling efficiency. It skips
        evaluation automatically if the parameters fall below the defined
        minimum array scale thresholds.

        Parameters
        ----------
        num_models : int
            The number of distinct model variants generated for the test.
        num_obs : int
            The total number of simulated observations per object.

        Raises
        ------
        AssertionError
            If the performance ratio of the Cython implementation is equal
            to or greater than 50% of the NumPy reference execution
            duration on the same target dimensions.
        """
        # Only enforce the threshold constraint on large array datasets
        if num_models < 1000:
            pytest.skip(
                "Speed threshold assertion targeted for large-scale arrays only."
            )

        models, obs, inv_sig_sq, scaling = make_mock_data(num_models, num_obs)
        iterations = 10  # Average out fluctuations across multiple iterations

        # Time the NumPy baseline
        start_np = time.perf_counter()
        for _ in range(iterations):
            _ = numpy_calc_chisq(models, obs, inv_sig_sq, scaling)
            _ = numpy_calc_scaling(models, obs, inv_sig_sq)
        numpy_duration = (time.perf_counter() - start_np) / iterations

        # Time the Cython implementation
        start_cy = time.perf_counter()
        for _ in range(iterations):
            _ = array_ops.calc_chisq(models, obs, inv_sig_sq, scaling)
            _ = array_ops.calc_scaling(models, obs, inv_sig_sq)
        cython_duration = (time.perf_counter() - start_cy) / iterations

        # Evaluate performance scaling ratio
        speed_ratio = cython_duration / numpy_duration
        assert speed_ratio < 0.50, (
            f"Performance regression detected! Cython was expected to be >2x faster, "
            f"but ran at {speed_ratio:.2%} of NumPy's runtime "
            f"(Cython: {cython_duration:.5f}s, NumPy: {numpy_duration:.5f}s)"
        )
