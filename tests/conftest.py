"""
Pytest configuration and common fixtures for PyCALQ tests.

This module provides common test fixtures and configuration
for all PyCALQ tests.
"""

import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_proj_files_handler(temp_dir):
    """Create a mock project files handler."""
    handler = Mock()
    handler.log_dir.return_value = temp_dir
    handler.plot_dir.return_value = temp_dir
    handler.corr_estimates_file.return_value = os.path.join(temp_dir, "test_corr.csv")
    handler.effen_estimates_file.return_value = os.path.join(temp_dir, "test_effen.csv")
    handler.spectrum_levels_file.return_value = os.path.join(temp_dir, "spectrum_levels.h5")
    handler.fit_estimates_file.return_value = os.path.join(temp_dir, "fit_estimates.csv")
    return handler


@pytest.fixture
def mock_sigmond_project_handler():
    """Create a mock Sigmond project handler."""
    handler = Mock()
    
    # Mock data handler
    data_handler = Mock()
    data_handler.raw_channels = ['test_channel_1', 'test_channel_2']
    data_handler.averaged_channels = ['test_channel_1', 'test_channel_2']
    data_handler.rotated_channels = ['test_channel_1', 'test_channel_2']
    data_handler.getChannelOperators.return_value = [Mock(), Mock()]
    data_handler.getChannelOperators2.return_value = [Mock(), Mock()]
    handler.data_handler = data_handler
    
    # Mock project info
    project_info = Mock()
    bins_info = Mock()
    sampling_info = Mock()
    bins_info.getNumberOfBins.return_value = 100
    bins_info.getRebinFactor.return_value = 1
    sampling_info.isJackknifeMode.return_value = True
    sampling_info.getSamplingMode.return_value = "Jackknife"
    sampling_info.getNumberOfReSamplings.return_value = 99
    project_info.bins_info = bins_info
    project_info.sampling_info = sampling_info
    handler.project_info = project_info
    
    # Mock other attributes
    handler.nodes = 1
    handler.subtract_vev = False
    handler.hermitian = True
    
    return handler


@pytest.fixture
def basic_general_configs():
    """Basic general configuration for tests."""
    return {
        'project_dir': '/test/project',
        'ensemble_id': 'test_ensemble'
    }


@pytest.fixture
def basic_task_configs():
    """Basic task configuration for tests."""
    return {
        'generate_estimates': True,
        'create_pdfs': True,
        'create_pickles': False,
        'create_summary': True,
        'plot': True
    }


@pytest.fixture
def mock_mcobs_handlers():
    """Create mock Monte Carlo observables handlers."""
    mcobs_handler = Mock()
    mcobs_get_handler = Mock()
    
    # Mock methods
    mcobs_handler.setToCorrelated.return_value = None
    mcobs_handler.setToUnCorrelated.return_value = None
    mcobs_handler.getFullAndSamplingValues.return_value = Mock()
    
    return mcobs_handler, mcobs_get_handler


@pytest.fixture
def sample_fit_config():
    """Sample fit configuration for testing."""
    return {
        'model': '1-exp',
        'tmin': 10,
        'tmax': 20,
        'exclude_times': [],
        'initial_params': {},
        'noise_cutoff': 0.0,
        'priors': {},
        'ratio': False,
        'sim_fit': False
    }


@pytest.fixture
def sample_spectrum_data():
    """Sample spectrum data for testing."""
    import numpy as np
    return {
        'energies': np.array([1.0, 1.5, 2.0, 2.5]),
        'errors': np.array([0.05, 0.08, 0.10, 0.12]),
        'channels': ['channel_1', 'channel_2', 'channel_3', 'channel_4'],
        'operators': ['op_1', 'op_2', 'op_3', 'op_4']
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration) 