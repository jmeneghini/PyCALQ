"""
Integration tests for task implementations.

This module tests the complete workflow of the refactored tasks
to ensure they work correctly together and maintain compatibility.
"""

import pytest
import tempfile
import os
import yaml
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from fvspectrum.tasks.preview_correlators import PreviewCorrelatorsTask
from fvspectrum.tasks.average_correlators import AverageCorrelatorsTask
from fvspectrum.tasks.rotate_correlators import RotateCorrelatorsTask
from fvspectrum.tasks.fit_spectrum import FitSpectrumTask
from fvspectrum.tasks.compare_spectrums import CompareSpectrumsTask


class TestTaskIntegration:
    """Integration tests for the complete task workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock project files handler
        self.mock_proj_handler = Mock()
        self.mock_proj_handler.log_dir.return_value = self.temp_dir
        self.mock_proj_handler.plot_dir.return_value = self.temp_dir
        self.mock_proj_handler.corr_estimates_file.return_value = os.path.join(self.temp_dir, "test_corr.csv")
        self.mock_proj_handler.effen_estimates_file.return_value = os.path.join(self.temp_dir, "test_effen.csv")
        
        # Mock sigmond project handler
        self.mock_sigmond_handler = Mock()
        self.mock_data_handler = Mock()
        self.mock_sigmond_handler.data_handler = self.mock_data_handler
        self.mock_data_handler.raw_channels = ['test_channel']
        self.mock_data_handler.getChannelOperators.return_value = [Mock(), Mock()]
        
        # Mock project info
        self.mock_project_info = Mock()
        self.mock_bins_info = Mock()
        self.mock_sampling_info = Mock()
        self.mock_project_info.bins_info = self.mock_bins_info
        self.mock_project_info.sampling_info = self.mock_sampling_info
        self.mock_sigmond_handler.project_info = self.mock_project_info
        
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
    def test_preview_correlators_workflow(self, mock_filter, mock_check):
        """Test the complete preview correlators workflow."""
        mock_check.return_value = ['/path/to/data.h5']
        mock_filter.return_value = ['test_channel']
        
        task_configs = {
            'raw_data_files': ['/path/to/data.h5'],
            'generate_estimates': True,
            'plot': True
        }
        
        task = PreviewCorrelatorsTask(
            task_name="preview_corrs",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=task_configs,
            sigmond_project_handler=self.mock_sigmond_handler
        )
        
        # Test that task initializes correctly
        assert task.task_name == "preview_corrs"
        assert task.channels == ['test_channel']
        
        # Test run method (should not raise exceptions)
        with patch.object(task, '_process_correlators'):
            task.run()
        
        # Test plot method
        with patch.object(task, '_plot_correlators_sequential'):
            task.plot()
    
    @patch('fvspectrum.sigmond_util.get_mcobs_handlers')
    def test_average_correlators_workflow(self, mock_mcobs):
        """Test the complete average correlators workflow."""
        mock_mcobs.return_value = (Mock(), Mock())
        
        task_configs = {
            'raw_data_files': ['/path/to/data.h5'],
            'average_hadron_irrep_info': True,
            'average_hadron_spatial_info': True
        }
        
        with patch('fvspectrum.sigmond_util.check_raw_data_files') as mock_check, \
             patch('fvspectrum.sigmond_util.filter_channels') as mock_filter:
            
            mock_check.return_value = ['/path/to/data.h5']
            mock_filter.return_value = ['test_channel']
            
            task = AverageCorrelatorsTask(
                task_name="average_corrs",
                proj_files_handler=self.mock_proj_handler,
                general_configs=self.general_configs,
                task_configs=task_configs,
                sigmond_project_handler=self.mock_sigmond_handler
            )
            
            # Test initialization
            assert task.task_name == "average_corrs"
            
            # Test run method
            with patch.object(task, '_perform_averaging'):
                task.run()
    
    def test_rotate_correlators_workflow(self):
        """Test the complete rotate correlators workflow."""
        task_configs = {
            'averaged_input_correlators_dir': ['/path/to/averaged.h5'],
            't0': 5,
            'tN': 10,
            'tD': 15,
            'pivot_type': 0
        }
        
        with patch('fvspectrum.sigmond_util.filter_channels') as mock_filter:
            mock_filter.return_value = ['test_channel']
            
            task = RotateCorrelatorsTask(
                task_name="rotate_corrs",
                proj_files_handler=self.mock_proj_handler,
                general_configs=self.general_configs,
                task_configs=task_configs,
                sigmond_project_handler=self.mock_sigmond_handler
            )
            
            # Test initialization
            assert task.task_name == "rotate_corrs"
            assert task.t0 == 5
            assert task.tN == 10
            assert task.tD == 15
            
            # Test run method
            with patch.object(task, '_execute_rotation_tasks'):
                task.run()
    
    def test_fit_spectrum_workflow(self):
        """Test the complete fit spectrum workflow."""
        task_configs = {
            'default_corr_fit': {
                'model': '1-exp',
                'tmin': 10,
                'tmax': 20
            },
            'single_hadrons': {
                'pi': ['pi_op1', 'pi_op2']
            },
            'do_interacting_fits': True
        }
        
        with patch('fvspectrum.sigmond_util.get_mcobs_handlers') as mock_mcobs:
            mock_mcobs.return_value = (Mock(), Mock())
            
            task = FitSpectrumTask(
                task_name="fit_spectrum",
                proj_files_handler=self.mock_proj_handler,
                general_configs=self.general_configs,
                task_configs=task_configs,
                sigmond_project_handler=self.mock_sigmond_handler
            )
            
            # Test initialization
            assert task.task_name == "fit_spectrum"
            assert task.default_fit.model == '1-exp'
            assert task.default_fit.tmin == 10
            assert task.default_fit.tmax == 20
            
            # Test run method
            with patch.object(task, '_perform_single_hadron_fits'), \
                 patch.object(task, '_perform_interacting_fits'), \
                 patch.object(task, '_generate_estimates'):
                task.run()
    
    def test_compare_spectrums_workflow(self):
        """Test the complete compare spectrums workflow."""
        # Create mock spectrum data files
        spectrum_file1 = os.path.join(self.temp_dir, "spectrum1.h5")
        spectrum_file2 = os.path.join(self.temp_dir, "spectrum2.h5")
        
        # Create minimal HDF5 files for testing
        import h5py
        with h5py.File(spectrum_file1, 'w') as f:
            f.create_dataset('test_level', data=[1.0, 1.1, 0.9])
        
        with h5py.File(spectrum_file2, 'w') as f:
            f.create_dataset('test_level', data=[1.05, 1.15, 0.95])
        
        task_configs = {
            'spectrum_files': [spectrum_file1, spectrum_file2],
            'spectrum_labels': ['Spectrum 1', 'Spectrum 2']
        }
        
        task = CompareSpectrumsTask(
            task_name="compare_spectrums",
            proj_files_handler=self.mock_proj_handler,
            general_configs=self.general_configs,
            task_configs=task_configs
        )
        
        # Test initialization
        assert task.task_name == "compare_spectrums"
        assert len(task.spectrum_files) == 2
        
        # Test run method
        with patch.object(task, '_load_spectrum_data'), \
             patch.object(task, '_compare_spectra'):
            task.run()


class TestTaskCompatibility:
    """Test compatibility between old and new implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock handlers
        self.mock_proj_handler = Mock()
        self.mock_proj_handler.log_dir.return_value = self.temp_dir
        
        self.mock_sigmond_handler = Mock()
        self.mock_data_handler = Mock()
        self.mock_sigmond_handler.data_handler = self.mock_data_handler
        self.mock_data_handler.raw_channels = ['test_channel']
        
        self.general_configs = {
            'project_dir': '/test/project',
            'ensemble_id': 'test_ensemble'
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_yaml_interface_compatibility(self):
        """Test that YAML interface remains compatible."""
        # Test configuration that should work with both old and new implementations
        task_configs = {
            'raw_data_files': ['/path/to/data.h5'],
            'generate_estimates': True,
            'create_pdfs': True,
            'create_pickles': False,
            'create_summary': True,
            'plot': True,
            'figwidth': 10,
            'figheight': 8
        }
        
        with patch('fvspectrum.sigmond_util.check_raw_data_files') as mock_check, \
             patch('fvspectrum.sigmond_util.filter_channels') as mock_filter:
            
            mock_check.return_value = ['/path/to/data.h5']
            mock_filter.return_value = ['test_channel']
            
            # Test that new implementation accepts all old parameters
            task = PreviewCorrelatorsTask(
                task_name="preview_corrs",
                proj_files_handler=self.mock_proj_handler,
                general_configs=self.general_configs,
                task_configs=task_configs,
                sigmond_project_handler=self.mock_sigmond_handler
            )
            
            # Verify all parameters are correctly set
            assert task.params['generate_estimates'] is True
            assert task.params['create_pdfs'] is True
            assert task.params['create_pickles'] is False
            assert task.params['create_summary'] is True
            assert task.params['plot'] is True
            assert task.params['figwidth'] == 10
            assert task.params['figheight'] == 8
    
    def test_legacy_wrapper_compatibility(self):
        """Test that legacy wrappers maintain compatibility."""
        # Import legacy wrappers
        from fvspectrum.sigmond_view_corrs_new import SigmondViewCorrs
        from fvspectrum.sigmond_average_corrs_new import SigmondAverageCorrs
        from fvspectrum.sigmond_rotate_corrs_new import SigmondRotateCorrs
        from fvspectrum.sigmond_spectrum_fits_new import SigmondSpectrumFits
        from fvspectrum.compare_sigmond_levels_new import CompareLevels
        
        # Test that legacy classes are aliases to new implementations
        assert SigmondViewCorrs == PreviewCorrelatorsTask
        assert SigmondAverageCorrs == AverageCorrelatorsTask
        assert SigmondRotateCorrs == RotateCorrelatorsTask
        assert SigmondSpectrumFits == FitSpectrumTask
        assert CompareLevels == CompareSpectrumsTask
    
    def test_documentation_compatibility(self):
        """Test that documentation is accessible through legacy interface."""
        from fvspectrum.sigmond_view_corrs_new import doc as preview_doc
        from fvspectrum.sigmond_average_corrs_new import doc as average_doc
        from fvspectrum.sigmond_rotate_corrs_new import doc as rotate_doc
        from fvspectrum.sigmond_spectrum_fits_new import doc as fit_doc
        from fvspectrum.compare_sigmond_levels_new import doc as compare_doc
        
        # Test that documentation strings are available
        assert isinstance(preview_doc, str)
        assert isinstance(average_doc, str)
        assert isinstance(rotate_doc, str)
        assert isinstance(fit_doc, str)
        assert isinstance(compare_doc, str)
        
        # Test that documentation contains expected content
        assert "preview_correlators" in preview_doc.lower()
        assert "average_corrs" in average_doc.lower()
        assert "rotate_corrs" in rotate_doc.lower()
        assert "fit_spectrum" in fit_doc.lower()
        assert "compare_spectrums" in compare_doc.lower()


class TestErrorHandling:
    """Test error handling in the refactored tasks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.mock_proj_handler = Mock()
        self.mock_proj_handler.log_dir.return_value = self.temp_dir
        
        self.mock_sigmond_handler = Mock()
        self.mock_data_handler = Mock()
        self.mock_sigmond_handler.data_handler = self.mock_data_handler
        
        self.general_configs = {
            'project_dir': '/test/project',
            'ensemble_id': 'test_ensemble'
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_required_parameters(self):
        """Test handling of missing required parameters."""
        # Test fit spectrum task without required fit configuration
        task_configs = {
            'single_hadrons': {'pi': ['pi_op1']}
            # Missing default_corr_fit
        }
        
        with pytest.raises(Exception):  # Should raise validation error
            FitSpectrumTask(
                task_name="fit_spectrum",
                proj_files_handler=self.mock_proj_handler,
                general_configs=self.general_configs,
                task_configs=task_configs,
                sigmond_project_handler=self.mock_sigmond_handler
            )
    
    def test_invalid_file_paths(self):
        """Test handling of invalid file paths."""
        task_configs = {
            'raw_data_files': ['/nonexistent/path.h5']
        }
        
        with patch('fvspectrum.sigmond_util.check_raw_data_files') as mock_check:
            mock_check.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(FileNotFoundError):
                PreviewCorrelatorsTask(
                    task_name="preview_corrs",
                    proj_files_handler=self.mock_proj_handler,
                    general_configs=self.general_configs,
                    task_configs=task_configs,
                    sigmond_project_handler=self.mock_sigmond_handler
                )
    
    def test_empty_channels_list(self):
        """Test handling of empty channels list."""
        task_configs = {
            'raw_data_files': ['/path/to/data.h5']
        }
        
        with patch('fvspectrum.sigmond_util.check_raw_data_files') as mock_check, \
             patch('fvspectrum.sigmond_util.filter_channels') as mock_filter:
            
            mock_check.return_value = ['/path/to/data.h5']
            mock_filter.return_value = []  # Empty channels list
            
            self.mock_data_handler.raw_channels = []
            
            task = PreviewCorrelatorsTask(
                task_name="preview_corrs",
                proj_files_handler=self.mock_proj_handler,
                general_configs=self.general_configs,
                task_configs=task_configs,
                sigmond_project_handler=self.mock_sigmond_handler
            )
            
            # Should handle empty channels gracefully
            assert task.channels == []


if __name__ == "__main__":
    pytest.main([__file__]) 