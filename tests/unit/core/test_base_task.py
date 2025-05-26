"""
Unit tests for base task classes.

This module tests the abstract base classes and common functionality
defined in fvspectrum.core.base_task.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock

from fvspectrum.core.base_task import BaseSpectrumTask, CorrelatorAnalysisTask


class ConcreteSpectrumTask(BaseSpectrumTask):
    """Concrete implementation of BaseSpectrumTask for testing."""
    
    @property
    def info(self) -> str:
        return "Test task documentation"
    
    def _get_default_parameters(self):
        return {
            'test_param': 'default_value',
            'plot': True,
            'create_pdfs': True
        }
    
    def run(self):
        pass
    
    def plot(self):
        pass


class ConcreteCorrelatorTask(CorrelatorAnalysisTask):
    """Concrete implementation of CorrelatorAnalysisTask for testing."""
    
    @property
    def info(self) -> str:
        return "Test correlator task documentation"
    
    def _get_default_parameters(self):
        return {
            'test_param': 'default_value',
            'plot': True,
            'create_pdfs': True,
            'generate_estimates': True
        }
    
    def run(self):
        pass
    
    def plot(self):
        pass


class TestBaseSpectrumTask:
    """Test the BaseSpectrumTask abstract base class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock project files handler
        self.mock_proj_handler = Mock()
        self.mock_proj_handler.log_dir.return_value = self.temp_dir
        
        # Basic configurations
        self.general_configs = {
            'project_dir': '/test/project',
            'ensemble_id': 'test_ensemble'
        }
        
        self.task_configs = {
            'test_param': 'user_value',
            'new_param': 'new_value'
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test basic initialization of BaseSpectrumTask."""
        task = ConcreteSpectrumTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=self.task_configs
        )
        
        assert task.task_name == "test_task"
        assert task.proj_files_handler == self.mock_proj_handler
        assert task.general_configs == self.general_configs
        assert task.task_configs == self.task_configs
        
        # Check parameter merging
        assert task.params['test_param'] == 'user_value'  # User override
        assert task.params['new_param'] == 'new_value'    # User addition
        assert task.params['plot'] is True                # Default preserved
    
    def test_parameter_merging(self):
        """Test that user parameters override defaults correctly."""
        task_configs = {
            'plot': False,  # Override default
            'custom_param': 'custom_value'  # New parameter
        }
        
        task = ConcreteSpectrumTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=task_configs
        )
        
        assert task.params['plot'] is False
        assert task.params['custom_param'] == 'custom_value'
        assert task.params['test_param'] == 'default_value'  # Default preserved
        assert task.params['create_pdfs'] is True           # Default preserved
    
    def test_logging_setup(self):
        """Test that logging setup creates the expected log file."""
        task = ConcreteSpectrumTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=self.task_configs
        )
        
        # Check that log file was created
        log_file_path = os.path.join(self.temp_dir, 'full_input.yml')
        assert os.path.exists(log_file_path)
        
        # Check log file contents
        with open(log_file_path, 'r') as f:
            logged_data = yaml.safe_load(f)
        
        assert 'general' in logged_data
        assert 'test_task' in logged_data
        assert logged_data['general'] == self.general_configs
        assert logged_data['test_task'] == self.task_configs
    
    def test_should_create_plots(self):
        """Test the _should_create_plots method."""
        # Test with default settings (should create plots)
        task = ConcreteSpectrumTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs={}
        )
        assert task._should_create_plots() is True
        
        # Test with plot disabled
        task_configs = {'plot': False}
        task = ConcreteSpectrumTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=task_configs
        )
        assert task._should_create_plots() is False
        
        # Test with all plot types disabled
        task_configs = {
            'plot': True,
            'create_pdfs': False,
            'create_pickles': False,
            'create_summary': False
        }
        task = ConcreteSpectrumTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=task_configs
        )
        assert task._should_create_plots() is False
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseSpectrumTask(
                task_name="test",
                proj_files_handler=Mock(),
                general_configs={},
                task_configs={}
            )
    
    def test_info_property(self):
        """Test the info property."""
        task = ConcreteSpectrumTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=self.task_configs
        )
        
        assert task.info == "Test task documentation"
    
    @patch('logging.info')
    def test_logging_methods(self, mock_log):
        """Test the logging helper methods."""
        task = ConcreteSpectrumTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=self.task_configs
        )
        
        task._log_task_start("Testing operation")
        mock_log.assert_called_with("Testing operation for task 'test_task'...")
        
        task._log_task_complete("Testing operation")
        mock_log.assert_called_with("Testing operation for task 'test_task' completed.")


class TestCorrelatorAnalysisTask:
    """Test the CorrelatorAnalysisTask class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock project files handler
        self.mock_proj_handler = Mock()
        self.mock_proj_handler.log_dir.return_value = self.temp_dir
        
        # Mock sigmond project handler
        self.mock_sigmond_handler = Mock()
        self.mock_data_handler = Mock()
        self.mock_sigmond_handler.data_handler = self.mock_data_handler
        self.mock_data_handler.raw_channels = ['channel1', 'channel2', 'channel3']
        
        # Basic configurations
        self.general_configs = {
            'project_dir': '/test/project',
            'ensemble_id': 'test_ensemble'
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('fvspectrum.sigmond_util.check_raw_data_files')
    @patch('fvspectrum.sigmond_util.filter_channels')
    def test_initialization_with_raw_data(self, mock_filter, mock_check):
        """Test initialization with raw data files."""
        mock_check.return_value = ['/path/to/data1.h5', '/path/to/data2.h5']
        mock_filter.return_value = ['channel1', 'channel2']
        
        task_configs = {
            'raw_data_files': ['/path/to/data1.h5', '/path/to/data2.h5']
        }
        
        task = ConcreteCorrelatorTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=task_configs,
            sigmond_project_handler=self.mock_sigmond_handler
        )
        
        # Check that raw data validation was called
        mock_check.assert_called_once_with(
            ['/path/to/data1.h5', '/path/to/data2.h5'],
            '/test/project'
        )
        
        # Check that data was added to project handler
        self.mock_sigmond_handler.add_raw_data.assert_called_once()
        
        # Check that channels were filtered
        mock_filter.assert_called_once()
        assert task.channels == ['channel1', 'channel2']
    
    def test_initialization_without_raw_data(self):
        """Test initialization without raw data files."""
        task_configs = {}
        
        task = ConcreteCorrelatorTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=task_configs,
            sigmond_project_handler=self.mock_sigmond_handler
        )
        
        # Should not call add_raw_data
        self.mock_sigmond_handler.add_raw_data.assert_not_called()
    
    @patch('fvspectrum.sigmond_util.check_raw_data_files')
    @patch('logging.critical')
    def test_empty_raw_data_files(self, mock_log, mock_check):
        """Test handling of empty raw data files list."""
        task_configs = {
            'raw_data_files': []
        }
        
        task = ConcreteCorrelatorTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=task_configs,
            sigmond_project_handler=self.mock_sigmond_handler
        )
        
        # Should log critical error
        mock_log.assert_called_once()
        assert "No directory to view" in mock_log.call_args[0][0]
    
    def test_default_parameters(self):
        """Test default parameters for correlator analysis tasks."""
        task = ConcreteCorrelatorTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs={},
            sigmond_project_handler=self.mock_sigmond_handler
        )
        
        # Check that correlator analysis defaults are set
        assert task.params['generate_estimates'] is True
        assert task.params['create_pdfs'] is True
        assert task.params['create_pickles'] is True
        assert task.params['create_summary'] is True
        assert task.params['plot'] is True
        assert task.params['figwidth'] == 8
        assert task.params['figheight'] == 6
    
    @patch('fvspectrum.sigmond_util.filter_channels')
    def test_channel_filtering(self, mock_filter):
        """Test that channels are properly filtered."""
        mock_filter.return_value = ['channel1']  # Filter out some channels
        
        task = ConcreteCorrelatorTask(
            task_name="test_task",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs={},
            sigmond_project_handler=self.mock_sigmond_handler
        )
        
        # Check that remove_raw_data_channels was called with filtered channels
        expected_removed = {'channel2', 'channel3'}
        self.mock_sigmond_handler.remove_raw_data_channels.assert_called_once_with(expected_removed)
        
        # Check that final channels list is correct
        assert task.channels == ['channel1']


if __name__ == "__main__":
    pytest.main([__file__]) 