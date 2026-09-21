"""
Unit tests for the AtlasGenerator module using pytest.
"""

import os
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from astropy.table import Table

# Assuming the module is named atlas_fitter_module.py
from bagpipes_extended.sed.atlas_fitter import (
    AtlasFitter,
    fit_object,
    fit_single,
    init_worker,
    temp_chdir,
)


class TestAtlasFitterSuite:
    """
    Tests for `bagpipes_extended.sed.atlas_fitter.Atlas_fitter`.

    Check instatiation, configurations, and core logic of AtlasFitter.
    """

    @pytest.fixture
    def sample_fit_instructions(self) -> dict[str, Any]:
        """
        Provide a standard set of fit instructions.

        Returns
        -------
        dict
            A dictionary containing nested priors and parameter
            bounds for physical modeling.
        """
        return {
            "redshift": (0.1, 2.0),
            "redshift_prior": "uniform",
            "continuity": {
                "massformed": (8.0, 12.0),
                "massformed_prior": "gaussian",
                "massformed_prior_mu": 10.0,
                "massformed_prior_sigma": 0.5,
            },
            "dust": {"type": "Calzetti", "Av": (0.0, 4.0)},
        }

    @pytest.fixture
    def mock_fitter_dependencies(self) -> Generator[None, None, None]:
        """
        Mock the shared memory manager and process initialization systems.

        Yields
        ------
        None
            Yields control back to the test block while mocking backend
            multiprocessing managers and process pools.
        """
        with (
            patch("multiprocessing.managers.SharedMemoryManager.start"),
            patch("multiprocessing.managers.SharedMemoryManager.shutdown", create=True),
            patch("multiprocessing.Pool"),
        ):
            yield

    @pytest.fixture
    def base_fitter(
        self,
        sample_fit_instructions: dict,
        mock_fitter_dependencies: None,
        tmp_path: Path,
    ) -> AtlasFitter:
        """
        Construct an AtlasFitter instance with mocked paths and memory.

        Parameters
        ----------
        sample_fit_instructions : dict
            The base prior limits and distribution configurations fixture.
        mock_fitter_dependencies : None
            The fixture mocking multiprocessing backends.
        tmp_path : pathlib.Path
            The built-in Pytest temporary directory fixture path.

        Returns
        -------
        AtlasFitter
            An initialized instance of AtlasFitter with a dummy atlas.
        """
        with patch.object(AtlasFitter, "_load_atlas") as mock_load:
            fitter = AtlasFitter(
                fit_instructions=sample_fit_instructions,
                atlas_path=tmp_path / "dummy_atlas.h5",
                out_path=tmp_path / "output",
                seed=42,
                overwrite=True,
            )
            return fitter

    def test_init_and_parameter_flattening(self, base_fitter: AtlasFitter) -> None:
        """
        Validate that the fit instructions are correctly parsed.

        Parameters
        ----------
        base_fitter : AtlasFitter
            The base initialized AtlasFitter fixture under test.
        """
        # Top level parameter tuple check
        assert "redshift" in base_fitter.params
        assert base_fitter.limits[base_fitter.params.index("redshift")] == (0.1, 2.0)

        # Nested dictionary parameter check
        assert "continuity:massformed" in base_fitter.params
        assert "dust:Av" in base_fitter.params

        # Verify hyperparameters assignment for non-uniform prior
        mass_idx = base_fitter.params.index("continuity:massformed")
        assert base_fitter.pdfs[mass_idx] == "gaussian"
        assert base_fitter.hyper_params[mass_idx] == {"mu": 10.0, "sigma": 0.5}

        # Context configurations
        assert base_fitter.ndim == 3
        assert base_fitter.overwrite is True
        assert isinstance(base_fitter.out_path, Path)

    def test_setup_vars_and_colnames(self, base_fitter: AtlasFitter) -> None:
        """
        Verify that the target catalogue columns match expectations.

        The output columns should depend on the `full_catalogue` flag and
        the presence of photometry in the observations.

        Parameters
        ----------
        base_fitter : AtlasFitter
            The base initialized AtlasFitter fixture under test.
        """
        # Baseline variables
        base_fitter._setup_vars()
        assert "stellar_mass" in base_fitter.vars
        assert "redshift" in base_fitter.vars
        assert "UV_colour" not in base_fitter.vars  # Requires full_catalogue flag

        colnames = base_fitter._setup_colnames()
        assert "#ID" in colnames
        assert "redshift_16" in colnames
        assert "redshift_50" in colnames
        assert "redshift_84" in colnames
        assert "log_evidence" in colnames

        # Full catalogue enabled tracking, no photometry
        base_fitter.full_catalogue = True
        base_fitter._setup_vars()
        colnames_full = base_fitter._setup_colnames()
        assert "UV_colour" in base_fitter.vars
        assert "chisq_phot" not in colnames_full

        # Full catalogue enabled tracking
        base_fitter.photometry_exists = True
        base_fitter._setup_vars()
        colnames_full = base_fitter._setup_colnames()
        assert "UV_colour" in base_fitter.vars
        assert "chisq_phot" in colnames_full

    def test_check_errs_dict(self, base_fitter: AtlasFitter) -> None:
        """
        Validate the additional uncertainties.

        Check that the uncertainties match dimensions and inject random
        distributions properly.

        Parameters
        ----------
        base_fitter : AtlasFitter
            The base initialized AtlasFitter fixture under test.
        """
        # When uncertainties dict is None
        base_fitter.check_errs_dict(min_uncertainties=None)
        assert base_fitter.add_errs is None

        # When standard errors are explicitly specified
        min_uncertainties = {"redshift": 0.05, "massformed": 0.01}
        base_fitter.check_errs_dict(min_uncertainties=min_uncertainties)

        assert base_fitter.add_errs.shape == (base_fitter.n_posterior, base_fitter.ndim)
        # Check that error mapping targets the matching index locations
        redshift_idx = base_fitter.params.index("redshift")
        assert not np.all(base_fitter.add_errs[:, redshift_idx] == 0)

    def test_context_manager_cleanup(
        self,
        sample_fit_instructions: dict,
        mock_fitter_dependencies: None,
        tmp_path: Path,
    ) -> None:
        """
        Test that the context manager functions correctly.

        Parameters
        ----------
        sample_fit_instructions : dict
            The base prior limits and distribution configurations fixture.
        mock_fitter_dependencies : None
            The fixture mocking multiprocessing backends.
        tmp_path : pathlib.Path
            The built-in Pytest temporary directory fixture path.
        """
        with patch.object(AtlasFitter, "_load_atlas"):
            with AtlasFitter(
                sample_fit_instructions, tmp_path / "a.h5", tmp_path / "out"
            ) as fitter:
                fitter._process_pool = MagicMock()
                mock_pool = fitter._process_pool
                mock_smm = fitter.smm

            # Once context closes, methods must invoke pool termination safely
            mock_pool.close.assert_called_once()
            mock_pool.join.assert_called_once()
            mock_smm.shutdown.assert_called_once()

    def test_process_pool_read_only_restriction(self, base_fitter: AtlasFitter) -> None:
        """
        Test if the read-only property prevents manual overwriting.

        Parameters
        ----------
        base_fitter : AtlasFitter
            The base initialized AtlasFitter fixture under test.

        Raises
        ------
        AttributeError
            If an attempt is made to directly overwrite the process pool
            property.
        """
        with pytest.raises(AttributeError, match="cannot be set directly"):
            base_fitter.process_pool = None

    @patch("h5py.File")
    def test_load_atlas_instruction_mismatch(
        self, mock_h5: MagicMock, base_fitter: AtlasFitter, tmp_path: Path
    ) -> None:
        """
        Check that fit instructions mismatches are correctly caught.

        Parameters
        ----------
        mock_h5 : unittest.mock.MagicMock
            Mock configuration wrapping the h5py File handler system.
        base_fitter : AtlasFitter
            The base initialized AtlasFitter fixture under test.
        tmp_path : pathlib.Path
            The built-in Pytest temporary directory fixture path.
        """
        # Simulated HDF5 contents carrying mismatched parameters
        mock_file = MagicMock()
        mock_file.attrs = {
            "fit_instructions": (
                "{'redshift': (5.0, 10.0)}"
            ),  # Incompatible range vs fixture
            "model_kwargs": "{}",
        }
        mock_h5.return_value.__enter__.return_value = mock_file

        with pytest.raises(ValueError, match="Fit instructions do not match"):
            base_fitter._load_atlas(tmp_path / "mismatch.h5")


class TestFunctionalFitHelpers:
    """
    Isolated checks wrapping the methods outside the object classes.

    This class groups functional verification procedures testing the
    isolated methods that run outside the `AtlasFitter` class environment.
    """

    @patch("bagpipes.galaxy")
    @patch("h5py.File")
    @patch("bagpipes_extended.sed.atlas_fitter.fit_single")
    @patch("bagpipes.fitting.posterior")
    def test_fit_object_saves_new_h5_and_returns_row(
        self,
        mock_post: MagicMock,
        mock_fit_single: MagicMock,
        mock_h5: MagicMock,
        mock_galaxy: MagicMock,
        tmp_path: Path,
    ) -> None:
        """
        Verify workflow orchestration inside fit_object including filesystem calls and formatting.

        Parameters
        ----------
        mock_post : unittest.mock.MagicMock
            Mocked object wrapping the bagpipes posterior function path.
        mock_fit_single : unittest.mock.MagicMock
            Mocked object wrapping the target execution step inside fit_single.
        mock_h5 : unittest.mock.MagicMock
            Mock tracking standard h5py file open routines.
        mock_galaxy : unittest.mock.MagicMock
            Mock tracking bagpipes galaxy building setups.
        tmp_path : pathlib.Path
            The built-in Pytest temporary directory fixture path.
        """
        # Setup mock behavior outputs
        mock_fit_single.return_value = {
            "samples2d": np.ones((5, 2)),
            "lnlike": np.zeros(5),
            "lnz": 12.5,
            "lnz_err": 0.2,
            "median": [1.0, 2.0],
            "conf_int": [[0.8, 1.8], [1.2, 2.2]],
        }
        # Mock structural components needed by posterior_obj.samples
        mock_post_instance = MagicMock()
        mock_post_instance.samples = {"redshift": np.array([1.1, 1.2, 1.3])}
        mock_post.return_value = mock_post_instance

        parent_path = tmp_path / "posterior"
        parent_path.mkdir()
        out_path = tmp_path / "out"
        out_path.mkdir()

        row = fit_object(
            ID="1234",
            load_data=MagicMock(),
            posterior_parent_path=parent_path,
            out_path=out_path,
            vars=["redshift"],
            z=1.0,
            params=["redshift"],
            rng_seed=42,
        )

        # Confirm the data structures map perfectly into row catalogues
        assert row["#ID"] == "1234"
        assert row["log_evidence"] == 12.5
        assert row["redshift_50"] == 1.2  # Median calculation check

    def test_temp_chdir(self, tmp_path: Path) -> None:
        """
        Ensure temp_chdir successfully updates and restores CWD.

        This test verifies that entering the context manager changes the
        current working directory to the target path, and exiting the context
        correctly restores the original working directory.

        Parameters
        ----------
        tmp_path : pathlib.Path
            The built-in Pytest temporary directory fixture path used to
            create the target destination.
        """
        original_cwd = os.getcwd()
        target_dir = tmp_path / "nested_dir"
        target_dir.mkdir()

        with temp_chdir(target_dir):
            assert os.getcwd() == os.path.abspath(target_dir)

        assert os.getcwd() == original_cwd

    @patch("bagpipes_extended.sed.atlas_fitter.shared_memory.SharedMemory")
    @patch("bagpipes_extended.sed.atlas_fitter.np.ndarray")
    def test_init_worker(
        self, mock_ndarray: MagicMock, mock_shared_memory: MagicMock
    ) -> None:
        """
        Ensure worker processes map shared memory segments correctly.

        This test verifies that the worker initialization routine attaches to
        pre-existing shared memory segments by name without triggering new
        allocations, and wraps those backing buffers into NumPy array structures.

        Parameters
        ----------
        mock_ndarray : unittest.mock.MagicMock
            Mock tracking standard numpy array buffer-wrapping instantiations.
        mock_shared_memory : unittest.mock.MagicMock
            Mock tracking underlying OS shared memory initialization handles.
        """
        mock_shm_instance = MagicMock()
        mock_shm_instance.buf = b"dummy_buffer"
        mock_shared_memory.return_value = mock_shm_instance

        init_worker(
            shared_param_samples_name="param_shm",
            shared_param_samples_shape=(2, 10),
            shared_model_atlas_name="atlas_shm",
            shared_model_atlas_shape=(10, 100),
        )

        # Verify both memory segments attach
        assert mock_shared_memory.call_count == 2
        mock_shared_memory.assert_any_call(name="param_shm", create=False)
        mock_shared_memory.assert_any_call(name="atlas_shm", create=False)

        # Verify numpy arrays are wrapped over buffers
        assert mock_ndarray.call_count == 2
