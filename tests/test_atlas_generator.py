"""
Unit tests for the AtlasGenerator module using pytest.
"""

import multiprocessing as mp
import os
import tempfile
import time
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

mp.set_start_method("fork")

from bagpipes_extended.sed.atlas_generator import (
    AtlasGenerator,
    init_worker,
    worker_process_components,
)


@pytest.fixture
def base_fit_instructions() -> dict[str, Any]:
    """
    Provide a basic, valid fit instructions dictionary.

    Includes `age` as a free parameter.

    Returns
    -------
    dict
        A dictionary containing redshift, and a parameterisation of the
        star formation history and dust attenuation.
    """
    return {
        "redshift": 0.2,
        "exponential": {
            "age": (0.1, 15.0),
            "age_prior": "uniform",
            "tau": 1.2,
            "massformed": 11.0,
        },
        "dust": {
            "type": "Calzetti",
            "Av": 0.4,
        },
    }


@pytest.fixture
def mock_dependencies() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """
    Mock external packages to ensure isolated execution.

    Yields
    ------
    mock_spec_gen : unittest.mock.MagicMock
        The mocked BagpipesSpecGenerator class.
    mock_prior_instance : unittest.mock.MagicMock
        The mocked prior instance handling parameter transformations.
    """
    with (
        patch(
            "bagpipes_extended.sed.atlas_generator.BagpipesSpecGenerator"
        ) as mock_spec_gen,
        patch("bagpipes_extended.sed.atlas_generator.prior") as mock_prior,
    ):

        # Configure standard mock instance methods behaviors
        mock_instance = MagicMock()
        mock_instance.sample_phot.return_value = np.array([1.0, 2.0, 0.0])
        mock_spec_gen.return_value = mock_instance

        mock_prior_instance = MagicMock()
        mock_prior_instance.transform.return_value = np.array([5.0])
        mock_prior.return_value = mock_prior_instance

        yield mock_spec_gen, mock_prior_instance


@pytest.mark.usefixtures("mock_dependencies")
class TestAtlasGenerator:
    """
    Tests for `bagpipes_extended.sed.atlas_generator.Atlas_generator`.

    Checks instantiation, limits, memory allocation, and serialization.
    """

    @pytest.fixture(autouse=True)
    def setup_generator(
        self, base_fit_instructions: dict[str, Any]
    ) -> Generator[None, None, None]:
        """
        Automatically instantiates and cleans up the generator context.

        Parameters
        ----------
        base_fit_instructions : dict[str, Any]
            Fixture providing baseline configuration settings.

        Yields
        ------
        None
            Yields control back to the consuming test function context.
        """
        # Note: Certain tests modify initial keywords; we configure a standard setup here.
        # Tests requiring custom initialization keywords can override or initialize manually.
        self.fit_instructions = base_fit_instructions
        self.filt_list = ["f1", "f2"]

        with AtlasGenerator(
            fit_instructions=self.fit_instructions, filt_list=self.filt_list
        ) as generator:
            self.gen = generator
            yield

    def test_initialization_and_parsing(self) -> None:
        """
        Test that parsing maps parameters, limits, and sizes correctly.
        """
        assert self.gen.ndim == 1
        assert "exponential:age" in self.gen.params
        assert self.gen.limits[0] == (0.1, 15.0)
        assert self.gen.pdfs[0] == "uniform"
        assert self.gen.param_vectors is None
        assert self.gen.model_atlas is None

    def test_unsupported_data_types_raise_error(self) -> None:
        """
        Test that unsupported configs immediately raise a clean exception.

        Raises
        ------
        NotImplementedError
            If spectroscopic sampling or line index sampling is requested.
        """
        # Overriding default instance for specific failure state checking
        with AtlasGenerator(
            fit_instructions=self.fit_instructions, spec_wavs=[5000.0]
        ) as generator:
            with pytest.raises(
                NotImplementedError,
                match="Spectroscopic sampling is currently unsupported.*",
            ):
                generator.gen_samples(n_samples=5)

        with AtlasGenerator(
            fit_instructions=self.fit_instructions, index_list=["index1"]
        ) as generator:
            with pytest.raises(
                NotImplementedError,
                match="Line index sampling is currently unsupported.*",
            ):
                generator.gen_samples(n_samples=5)

    def test_memory_limit_exhaustion_raises_error(self) -> None:
        """
        Test that excessive sample sizes raise errors correctly.

        Raises
        ------
        MemoryError
            If the memory required for the requested sample space exceeds
            the amount available.
        """
        self.gen.rng = np.random.default_rng(seed=42)
        with pytest.raises(
            MemoryError,
            match="The required memory would exceed the amount available",
        ):
            self.gen.initialise_shared_memory(n_samples=10**15, n_output=2)

    def test_shared_memory_allocation(self) -> None:
        """
        Test that shared memory arrays allocate exact shapes and bounds.
        """
        self.gen.rng = np.random.default_rng(seed=123)
        n_samples = 4
        n_output = 3  # (2 filters + 1 unphysical flag column)

        self.gen.initialise_shared_memory(n_samples=n_samples, n_output=n_output)

        assert self.gen.param_vectors.shape == (4, 1)
        assert self.gen.model_atlas.shape == (4, 3)
        assert np.all((self.gen.param_vectors >= 0.0) & (self.gen.param_vectors <= 1.0))

    def test_process_pool_read_only_restriction(self) -> None:
        """
        Test if the read-only property prevents manual overwriting.

        Raises
        ------
        AttributeError
            If an attempt is made to directly overwrite the process pool
            property.
        """
        with pytest.raises(AttributeError, match="cannot be set directly"):
            self.gen.process_pool = None

    def test_end_to_end_generation_and_hdf5_serialization(self) -> None:
        """
        Validate the entire pipeline lifecycle and check data writing.
        """
        self.gen.gen_samples(n_samples=3, seed=99, parallel=0)

        assert self.gen.param_vectors is not None
        assert self.gen.model_atlas is not None

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.gen.write_samples(tmp_path)

            with h5py.File(tmp_path, "r") as h5_file:
                assert "exponential:age" in h5_file
                assert "model_atlas" in h5_file
                assert "fit_instructions" in h5_file.attrs
                assert "model_kwargs" in h5_file.attrs

                saved_atlas = h5_file["model_atlas"][:]
                assert saved_atlas.shape == (3, 3)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @pytest.mark.parametrize("cores", [0, 2, -1])
    def test_generation_performance_benchmark(
        self, cores: int, mock_dependencies: tuple[MagicMock, MagicMock], benchmark: Any
    ) -> None:
        """
        Benchmark the sample generation with varying numbers of cores.

        Measures generation runtime scaling characteristics to optimize
        multiprocessing execution setups.

        Parameters
        ----------
        cores : int
            The number of processes to use.
        mock_dependencies : tuple[unittest.mock.MagicMock, unittest.mock.MagicMock]
            Fixture isolating the execution environment from external packages.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            The pytest performance tracking profiling hook.
        """
        n_samples = 1000
        mock_spec_gen, _ = mock_dependencies

        def _simulate_processing_load(param_vector):
            time.sleep(0.001)
            return np.array([1.0, 2.0, 0.0])

        mock_spec_gen.return_value.sample_phot.side_effect = _simulate_processing_load

        def _run_grid():
            self.gen.gen_samples(n_samples=n_samples, seed=42, parallel=cores)

        benchmark.pedantic(_run_grid, rounds=3, iterations=1)

        total_tracked_time = benchmark.stats.stats.mean
        throughput = n_samples / total_tracked_time

        print(f"\n📊 PERFORMANCE SUMMARY FOR CORE SETTING [{cores}]:")
        print(f" -> Mean execution time: {total_tracked_time:.4f} seconds")
        print(f" -> Grid scaling throughput: {throughput:.2f} samples/sec")

        assert self.gen.model_atlas.shape == (n_samples, 3)


def test_worker_process_components_execution(
    base_fit_instructions: dict[str, Any],
) -> None:
    """
    Directly test the global-scoped method mimicking a single process.

    Parameters
    ----------
    base_fit_instructions : dict[str, Any]
        Fixture providing baseline configuration settings.
    """
    mock_spec_gen = MagicMock()
    mock_spec_gen.sample_phot.return_value = np.array([10.0, 20.0])

    mock_prior = MagicMock()
    mock_prior.transform.return_value = np.array([4.5])

    import bagpipes_extended.sed.atlas_generator

    bagpipes_extended.sed.atlas_generator.shared_init_params = np.array([[0.5]])
    bagpipes_extended.sed.atlas_generator.shared_model_atlas = np.array([[0.0, 0.0]])
    bagpipes_extended.sed.atlas_generator.spec_generator = mock_spec_gen
    bagpipes_extended.sed.atlas_generator.prior_inst = mock_prior

    worker_process_components(arr_idx=0)

    assert bagpipes_extended.sed.atlas_generator.shared_init_params[0, 0] == 4.5
    assert np.array_equal(
        bagpipes_extended.sed.atlas_generator.shared_model_atlas[0], [10.0, 20.0]
    )
    mock_prior.transform.assert_called_once()
    mock_spec_gen.sample_phot.assert_called_once_with(np.array([4.5]))
