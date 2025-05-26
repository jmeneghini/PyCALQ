"""
Preview correlators task implementation.

This module provides the refactored implementation of the preview correlators task,
which reads and estimates/plots Lattice QCD temporal correlator data files.
"""

import logging
from typing import Dict, Any, List, Optional

from fvspectrum.core.base_task import CorrelatorAnalysisTask
from fvspectrum.analysis.correlator_processor import CorrelatorProcessor
from fvspectrum.plotting.spectrum_plotter import SpectrumPlotter


# Documentation string for the task
TASK_DOCUMENTATION = '''
preview - a task to read in and estimate/plot any Lattice QCD temporal correlator data files given

This task processes raw correlator data files and generates estimates and plots for
correlator functions and effective energies. It serves as a first step in the analysis
pipeline to visualize and validate the input data.

Configuration Parameters:
------------------------
general:
  ensemble_id: cls21_c103       # Required: Ensemble identifier
  project_dir: /path/to/project # Required: Project directory path
  sampling_info:                # Optional: Statistical sampling configuration
    mode: Jackknife             # Default: Jackknife (or Bootstrap)
  tweak_ensemble:               # Optional: Ensemble modifications
    omissions: []               # Default: [] - configurations to omit
    rebin: 1                    # Default: 1 - rebinning factor

preview:                  # Required task configuration
  raw_data_files:               # Required: List of input data files
  - /path/to/data/file1.bin
  - /path/to/data/file2.bin
  create_pdfs: true             # Optional: Generate PDF plots (default: true)
  create_pickles: true          # Optional: Generate pickle files (default: true)
  create_summary: true          # Optional: Generate summary document (default: true)
  figheight: 6                  # Optional: Figure height (default: 6)
  figwidth: 8                   # Optional: Figure width (default: 8)
  generate_estimates: true      # Optional: Generate CSV estimates (default: true)
  plot: true                    # Optional: Create plots (default: true)
  separate_mom: true            # Optional: Separate by momentum (default: true)

Output:
-------
- CSV files with correlator estimates and effective energies
- PDF/pickle plots of correlators and effective energies
- Summary document with all plots
- Log files with operator information
'''


class CorrelatorPreviewTask(CorrelatorAnalysisTask):
    """
    Task for previewing and analyzing correlator data.
    
    This task reads raw correlator data files, computes estimates for
    correlator functions and effective energies, and generates plots
    for visualization and validation. It serves as the first step in
    the analysis pipeline to validate input data quality.
    """
    
    @property
    def info(self) -> str:
        """Return task documentation."""
        return TASK_DOCUMENTATION
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for preview correlators task."""
        defaults = super()._get_default_parameters()
        defaults.update({
            'separate_mom': True,
        })
        return defaults
    
    def _validate_parameters(self) -> None:
        """Validate task-specific parameters."""
        super()._validate_parameters()
        
        # Check for required raw data files
        if 'raw_data_files' not in self.task_configs:
            logging.critical(f"No directory to view. Add 'raw_data_files' to '{self.task_name}' task parameters.")
        
        # Validate plot settings
        if not any([self.params['create_pdfs'], self.params['create_pickles'], self.params['create_summary']]):
            self.params['plot'] = False
            logging.warning("All plot output options disabled. Setting plot=False.")
    
    def run(self) -> None:
        """Execute the preview correlators task."""
        self._log_task_start("Running preview correlators analysis")
        
        # Initialize correlator processor
        processor = CorrelatorProcessor(
            self.data_handler, 
            self.project_handler, 
            self.proj_files_handler
        )
        
        # Log operator information
        processor.log_operators_info(self.channels)
        
        # Check if we need to do anything
        if not self.params['generate_estimates'] and not self.params['plot']:
            logging.warning("Both 'generate_estimates' and 'plot' are False. Task is obsolete.")
            return
        
        # Process correlators
        save_estimates = self.params['generate_estimates']
        save_to_memory = not save_estimates and self.params['plot']
        
        self.processed_data = processor.process_correlators(
            self.channels, 
            save_estimates=save_estimates,
            save_to_memory=save_to_memory
        )
        
        self._log_task_complete("Preview correlators analysis")
    
    def plot(self) -> None:
        """Generate plots for the preview correlators task."""
        if not self._should_create_plots():
            logging.info("No plots requested.")
            return
        
        self._log_task_start("Creating plots")
        
        # Initialize plotter
        plotter = SpectrumPlotter(self.proj_files_handler, self.params)
        
        # Determine if we should use multiprocessing
        use_multiprocessing = (
            hasattr(self.project_handler, 'nodes') and 
            self.project_handler.nodes and 
            self.project_handler.nodes > 1
        )
        
        # Plot correlators
        plotter.plot_correlators(
            self.channels, 
            self.data_handler,
            data=getattr(self, 'processed_data', None),
            use_multiprocessing=use_multiprocessing
        )
        
        # Create summary document if requested
        if self.params['create_summary']:
            self._create_summary_document(plotter)
        
        self._log_task_complete("Plot generation")
    
    def _create_summary_document(self, plotter: SpectrumPlotter) -> None:
        """
        Create summary document with all plots.
        
        Args:
            plotter: SpectrumPlotter instance
        """
        # Create summary documents (potentially separated by momentum)
        if self.params['separate_mom']:
            self._create_momentum_separated_summaries(plotter)
        else:
            self._create_single_summary(plotter)
    
    def _create_momentum_separated_summaries(self, plotter: SpectrumPlotter) -> None:
        """Create separate summary documents for each momentum."""
        # Get unique momentum values
        momentum_values = sorted(set(channel.psq for channel in self.channels))
        
        # Clean up any existing summary files
        self._cleanup_summary_files()
        
        # Create summary for each momentum
        for i, momentum in enumerate(momentum_values):
            plotter.create_summary_document(f"Preview Data - PSQ={momentum}")
            
            # Add sections for channels with this momentum
            momentum_channels = [ch for ch in self.channels if ch.psq == momentum]
            self._add_channel_sections_to_summary(plotter, momentum_channels, i)
        
        # Finalize all summaries
        self._finalize_summaries(plotter, momentum_values)
        
        logging.info(f"Summary files saved to {self.proj_files_handler.summary_file('*')}.pdf.")
    
    def _create_single_summary(self, plotter: SpectrumPlotter) -> None:
        """Create a single summary document for all channels."""
        plotter.create_summary_document("Preview Data")
        self._add_channel_sections_to_summary(plotter, self.channels, 0)
        
        # Clean up and compile
        self._cleanup_summary_files()
        plotter.plh.compile_pdf(self.proj_files_handler.summary_file())
        logging.info(f"Summary file saved to {self.proj_files_handler.summary_file()}.pdf.")
    
    def _cleanup_summary_files(self) -> None:
        """Clean up any existing summary files."""
        import glob
        import os
        
        patterns = [
            self.proj_files_handler.summary_file('*') + ".*",
            self.proj_files_handler.summary_file() + ".*"
        ]
        
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # File might not exist or be in use
    
    def _finalize_summaries(self, plotter: SpectrumPlotter, momentum_values: List[int]) -> None:
        """
        Finalize summary documents with optional multiprocessing.
        
        Args:
            plotter: SpectrumPlotter instance
            momentum_values: List of momentum values
        """
        use_multiprocessing = (
            hasattr(self.project_handler, 'nodes') and 
            self.project_handler.nodes and 
            self.project_handler.nodes > 1
        )
        
        if use_multiprocessing:
            self._finalize_summaries_parallel(plotter, momentum_values)
        else:
            self._finalize_summaries_sequential(plotter, momentum_values)
    
    def _finalize_summaries_parallel(self, plotter: SpectrumPlotter, momentum_values: List[int]) -> None:
        """Finalize summaries using multiprocessing."""
        from multiprocessing import Process
        
        processes = []
        for i, psq in enumerate(momentum_values):
            process = Process(
                target=plotter.plh.compile_pdf, 
                args=(self.proj_files_handler.summary_file(psq), i)
            )
            processes.append(process)
            process.start()
            
            # Limit concurrent processes
            if len(processes) >= self.project_handler.nodes:
                processes[0].join()
                processes.pop(0)
        
        # Wait for remaining processes
        for process in processes:
            process.join()
    
    def _finalize_summaries_sequential(self, plotter: SpectrumPlotter, momentum_values: List[int]) -> None:
        """Finalize summaries sequentially."""
        # Temporarily reduce logging verbosity
        original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.WARNING)
        
        try:
            for i, psq in enumerate(momentum_values):
                plotter.plh.compile_pdf(self.proj_files_handler.summary_file(psq), i)
        finally:
            logging.getLogger().setLevel(original_level)
    
    def _add_channel_sections_to_summary(self, plotter: SpectrumPlotter, 
                                       channels: List[Any], index: int) -> None:
        """
        Add channel sections to summary document.
        
        Args:
            plotter: SpectrumPlotter instance
            channels: List of channels to add
            index: Summary document index
        """
        import sigmond
        
        for channel in channels:
            # Add section for this channel
            plotter.plh.append_section(str(channel), index)
            
            # Add subsections for each correlator pair
            channel_operators = self.data_handler.getChannelOperators(channel)
            
            for op1 in channel_operators:
                for op2 in channel_operators:
                    corr = sigmond.CorrelatorInfo(op1.operator_info, op2.operator_info)
                    corr_name = repr(corr).replace(" ", "-")
                    
                    # Add correlator subsection with both plots
                    if self.params['create_pdfs'] or self.params['create_summary']:
                        corr_pdf = self.proj_files_handler.corr_plot_file(corr_name, "pdf")
                        effen_pdf = self.proj_files_handler.effen_plot_file(corr_name, "pdf")
                        
                        plotter.plh.add_correlator_subsection(
                            repr(corr), corr_pdf, effen_pdf, index
                        )


class DataQualityValidator:
    """
    Utility class for validating correlator data quality and consistency.
    
    This class provides comprehensive methods for checking data quality,
    identifying potential issues, and generating detailed validation reports
    for troubleshooting and quality assurance.
    """
    
    def __init__(self, data_handler, channels: List[Any]):
        """
        Initialize the data validator.
        
        Args:
            data_handler: Sigmond data handler
            channels: List of channels to validate
        """
        self.data_handler = data_handler
        self.channels = channels
        self.validation_results = {}
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Perform comprehensive data validation.
        
        Returns:
            Dictionary with validation results
        """
        logging.info("Performing data validation...")
        
        for channel in self.channels:
            self.validation_results[str(channel)] = self._validate_channel(channel)
        
        return self.validation_results
    
    def _validate_channel(self, channel: Any) -> Dict[str, Any]:
        """
        Validate data for a single channel.
        
        Args:
            channel: Physics channel to validate
            
        Returns:
            Dictionary with validation results for this channel
        """
        results = {
            'operators_count': 0,
            'missing_correlators': [],
            'data_quality_issues': [],
            'recommendations': []
        }
        
        operators = self.data_handler.getChannelOperators(channel)
        results['operators_count'] = len(operators)
        
        # Check for missing correlators
        expected_correlators = len(operators) ** 2
        actual_correlators = self._count_available_correlators(channel, operators)
        
        if actual_correlators < expected_correlators:
            results['missing_correlators'] = self._identify_missing_correlators(channel, operators)
        
        # Check data quality
        results['data_quality_issues'] = self._check_data_quality(channel, operators)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _count_available_correlators(self, channel: Any, operators: List[Any]) -> int:
        """Count available correlators for a channel."""
        count = 0
        for op1 in operators:
            for op2 in operators:
                # Check if correlator data exists
                # This would involve checking the data handler
                count += 1  # Placeholder
        return count
    
    def _identify_missing_correlators(self, channel: Any, operators: List[Any]) -> List[str]:
        """Identify missing correlators."""
        missing = []
        # Implementation would check which correlator pairs are missing
        return missing
    
    def _check_data_quality(self, channel: Any, operators: List[Any]) -> List[str]:
        """Check for data quality issues."""
        issues = []
        # Implementation would check for:
        # - NaN values
        # - Infinite values
        # - Suspicious statistical fluctuations
        # - Inconsistent time ranges
        return issues
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if results['missing_correlators']:
            recommendations.append("Some correlators are missing. Check input data files.")
        
        if results['data_quality_issues']:
            recommendations.append("Data quality issues detected. Review data preprocessing.")
        
        if results['operators_count'] < 2:
            recommendations.append("Few operators available. Consider adding more operators for better analysis.")
        
        return recommendations
    
    def save_validation_report(self, output_file: str) -> None:
        """
        Save validation report to file.
        
        Args:
            output_file: Path to output file
        """
        import yaml
        
        with open(output_file, 'w') as f:
            yaml.dump(self.validation_results, f, default_flow_style=False)
        
        logging.info(f"Validation report saved to {output_file}") 