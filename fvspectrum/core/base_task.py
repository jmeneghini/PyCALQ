"""
Base task class for spectrum analysis tasks.

This module provides the abstract base class that all spectrum analysis tasks inherit from,
defining the common interface and shared functionality.
"""

import logging
import os
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import fvspectrum.sigmond_util as sigmond_util


class BaseSpectrumTask(ABC):
    """
    Abstract base class for all spectrum analysis tasks.
    
    This class defines the common interface and shared functionality that all
    spectrum analysis tasks must implement. It handles common initialization,
    parameter validation, and logging setup.
    """
    
    def __init__(self, task_name: str, proj_files_handler, general_configs: Dict[str, Any], 
                 task_configs: Dict[str, Any], sigmond_project_handler=None):
        """
        Initialize the base spectrum task.
        
        Args:
            task_name: Name of the task
            proj_files_handler: Project file handler for managing output files
            general_configs: General configuration parameters
            task_configs: Task-specific configuration parameters
            sigmond_project_handler: Optional Sigmond project handler
        """
        self.task_name = task_name
        self.proj_files_handler = proj_files_handler
        self.general_configs = general_configs
        self.task_configs = task_configs
        self.project_handler = sigmond_project_handler
        
        # Initialize default parameters
        self.default_params = self._get_default_parameters()
        
        # Update parameters with user-provided values
        self.params = self._merge_parameters(self.default_params, task_configs)
        
        # Validate parameters
        self._validate_parameters()
        
        # Set up logging
        self._setup_logging()
    
    @property
    @abstractmethod
    def info(self) -> str:
        """Return documentation string for this task."""
        pass
    
    @abstractmethod
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for this task."""
        pass
    
    @abstractmethod
    def run(self) -> None:
        """Execute the main task logic."""
        pass
    
    @abstractmethod
    def plot(self) -> None:
        """Generate plots for this task."""
        pass
    
    def _merge_parameters(self, defaults: Dict[str, Any], user_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge default parameters with user-provided parameters.
        
        Args:
            defaults: Default parameter values
            user_params: User-provided parameter values
            
        Returns:
            Merged parameter dictionary
        """
        merged = defaults.copy()
        sigmond_util.update_params(merged, user_params)
        return merged
    
    def _validate_parameters(self) -> None:
        """Validate task parameters. Override in subclasses for specific validation."""
        pass
    
    def _setup_logging(self) -> None:
        """Set up logging for this task."""
        log_file_path = os.path.join(self.proj_files_handler.log_dir(), 'full_input.yml')
        logging.info(f"Full input written to '{log_file_path}'.")
        
        with open(log_file_path, 'w+') as log_file:
            yaml.dump({
                "general": self.general_configs,
                self.task_name: self.task_configs
            }, log_file)
    
    def _should_create_plots(self) -> bool:
        """Determine if plots should be created based on parameters."""
        return (self.params.get('plot', True) and 
                (self.params.get('create_pdfs', True) or 
                 self.params.get('create_pickles', True) or 
                 self.params.get('create_summary', True)))
    
    def _log_task_start(self, operation: str) -> None:
        """Log the start of a task operation."""
        logging.info(f"{operation} for task '{self.task_name}'...")
    
    def _log_task_complete(self, operation: str) -> None:
        """Log the completion of a task operation."""
        logging.info(f"{operation} for task '{self.task_name}' completed.")


class CorrelatorAnalysisTask(BaseSpectrumTask):
    """
    Base class for tasks that analyze correlator data.
    
    This class extends BaseSpectrumTask with functionality specific to
    correlator analysis, including data validation and channel filtering.
    """
    
    def __init__(self, task_name: str, proj_files_handler, general_configs: Dict[str, Any], 
                 task_configs: Dict[str, Any], sigmond_project_handler):
        """Initialize correlator analysis task."""
        super().__init__(task_name, proj_files_handler, general_configs, 
                        task_configs, sigmond_project_handler)
        
        # Validate raw data files if required
        if 'raw_data_files' in task_configs:
            self._validate_raw_data_files()
        
        # Set up data handlers
        self._setup_data_handlers()
    
    def _validate_raw_data_files(self) -> None:
        """Validate raw data files."""
        raw_data_files = self.task_configs.get('raw_data_files', [])
        if not raw_data_files:
            logging.critical(f"No directory to view. Add 'raw_data_files' to '{self.task_name}' task parameters.")
        
        validated_files = sigmond_util.check_raw_data_files(
            raw_data_files, self.general_configs['project_dir']
        )
        self.project_handler.add_raw_data(validated_files)
    
    def _setup_data_handlers(self) -> None:
        """Set up data handlers for correlator analysis."""
        if self.project_handler:
            self.data_handler = self.project_handler.data_handler
            self.channels = self.data_handler.raw_channels[:]
            
            # Filter channels based on task configuration
            final_channels = sigmond_util.filter_channels(self.task_configs, self.channels)
            remove_channels = set(self.channels) - set(final_channels)
            self.project_handler.remove_raw_data_channels(remove_channels)
            self.channels = final_channels
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for correlator analysis tasks."""
        return {
            'generate_estimates': True,
            'create_pdfs': True,
            'create_pickles': True,
            'create_summary': True,
            'plot': True,
            'figwidth': 8,
            'figheight': 6,
        } 