"""
Spectrum plotting and visualization.

This module provides classes for creating plots and visualizations of
spectrum analysis results, including correlator plots, effective energy
plots, and spectrum summary plots.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from multiprocessing import Process
import tqdm

import general.plotting_handler as ph
import fvspectrum.sigmond_util as sigmond_util
from fvspectrum.core.data_structures import SpectrumLevel, FitResult


class SpectrumPlotter:
    """
    Handles plotting and visualization of spectrum analysis results.
    
    This class manages the creation of various plots including correlator
    plots, effective energy plots, fit quality plots, and spectrum summaries.
    """
    
    def __init__(self, proj_files_handler, plot_config: Dict[str, Any]):
        """
        Initialize the spectrum plotter.
        
        Args:
            proj_files_handler: Project file handler for output management
            plot_config: Configuration for plotting parameters
        """
        self.proj_files_handler = proj_files_handler
        self.plot_config = plot_config
        self.plh = ph.PlottingHandler()
        
        # Set up matplotlib style
        self._setup_plotting_style()
        
        # Create figure with specified dimensions
        self.plh.create_fig(
            plot_config.get('figwidth', 8), 
            plot_config.get('figheight', 6)
        )
    
    def _setup_plotting_style(self) -> None:
        """Set up matplotlib plotting style."""
        # Set LaTeX usage based on availability
        sigmond_util.set_latex_in_plots(matplotlib.style)
    
    def plot_correlators(self, channels: List[Any], data_handler, 
                        data: Optional[Dict] = None, use_multiprocessing: bool = False) -> None:
        """
        Plot correlator data for all channels.
        
        Args:
            channels: List of physics channels
            data_handler: Sigmond data handler
            data: Optional pre-computed data dictionary
            use_multiprocessing: Whether to use multiprocessing for plotting
        """
        if not self.plot_config.get('plot', True):
            logging.info("No plots requested.")
            return
        
        logging.info(f"Saving plots to directory {self.proj_files_handler.plot_dir()}...")
        
        if use_multiprocessing and self.plot_config.get('nodes', 1) > 1:
            self._plot_correlators_multiprocess(channels, data_handler, data)
        else:
            self._plot_correlators_sequential(channels, data_handler, data)
    
    def _plot_correlators_sequential(self, channels: List[Any], data_handler, 
                                   data: Optional[Dict] = None) -> None:
        """Plot correlators sequentially."""
        for channel in tqdm.tqdm(channels, desc="Plotting channels"):
            if data is not None:
                self._plot_channels_with_data(channel, data_handler, data)
            else:
                self._plot_channels_from_files(channel, data_handler)
    
    def _plot_correlators_multiprocess(self, channels: List[Any], data_handler, 
                                     data: Optional[Dict] = None) -> None:
        """Plot correlators using multiprocessing."""
        nodes = self.plot_config.get('nodes', 1)
        chunk_size = int(len(channels) / nodes) + 1
        channels_per_node = [channels[i:i + chunk_size] for i in range(0, len(channels), chunk_size)]
        
        processes = []
        for channel_chunk in channels_per_node:
            if data is not None:
                process = Process(target=self._plot_channels_with_data, 
                                args=(channel_chunk, data_handler, data))
            else:
                process = Process(target=self._plot_channels_from_files, 
                                args=(channel_chunk, data_handler))
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
    
    def _plot_channels_with_data(self, channel_chunk: List[Any], data_handler, data: Dict) -> None:
        """Plot channels using pre-computed data."""
        import fvspectrum.sigmond_util as sigmond_util
        
        if isinstance(channel_chunk, list):
            for channel in channel_chunk:
                if channel in data:
                    channel_operators = data_handler.getChannelOperators(channel)
                    sigmond_util.write_channel_plots(
                        channel_operators, self.plh, 
                        self.plot_config.get('create_pickles', True),
                        self.plot_config.get('create_pdfs', True) or self.plot_config.get('create_summary', True),
                        self.proj_files_handler, data[channel]
                    )
        else:
            # Single channel
            channel = channel_chunk
            if channel in data:
                channel_operators = data_handler.getChannelOperators(channel)
                sigmond_util.write_channel_plots(
                    channel_operators, self.plh, 
                    self.plot_config.get('create_pickles', True),
                    self.plot_config.get('create_pdfs', True) or self.plot_config.get('create_summary', True),
                    self.proj_files_handler, data[channel]
                )
    
    def _plot_channels_from_files(self, channel_chunk: List[Any], data_handler) -> None:
        """Plot channels loading data from files."""
        import fvspectrum.sigmond_util as sigmond_util
        
        if isinstance(channel_chunk, list):
            for channel in channel_chunk:
                channel_operators = data_handler.getChannelOperators(channel)
                sigmond_util.write_channel_plots(
                    channel_operators, self.plh,
                    self.plot_config.get('create_pickles', True),
                    self.plot_config.get('create_pdfs', True) or self.plot_config.get('create_summary', True),
                    self.proj_files_handler
                )
        else:
            # Single channel
            channel = channel_chunk
            channel_operators = data_handler.getChannelOperators(channel)
            sigmond_util.write_channel_plots(
                channel_operators, self.plh,
                self.plot_config.get('create_pickles', True),
                self.plot_config.get('create_pdfs', True) or self.plot_config.get('create_summary', True),
                self.proj_files_handler
            )
    
    def _plot_channel_correlators(self, channel: Any, data_handler, 
                                data: Optional[Dict] = None) -> None:
        """
        Plot correlators for a single channel.
        
        Args:
            channel: Physics channel
            data_handler: Sigmond data handler
            data: Optional pre-computed data
        """
        channel_operators = data_handler.getChannelOperators(channel)
        
        for op1 in channel_operators:
            for op2 in channel_operators:
                self._plot_correlator_pair(channel, op1, op2, data)
    
    def _plot_correlator_pair(self, channel: Any, op1: Any, op2: Any, 
                            data: Optional[Dict] = None) -> None:
        """
        Plot a single correlator pair.
        
        Args:
            channel: Physics channel
            op1: First operator
            op2: Second operator
            data: Optional pre-computed data
        """
        import sigmond
        corr = sigmond.CorrelatorInfo(op1.operator_info, op2.operator_info)
        corr_name = repr(corr).replace(" ", "-")
        
        if data is not None and channel in data and op1 in data[channel] and op2 in data[channel][op1]:
            # Use pre-computed data
            corr_data = data[channel][op1][op2]["corr"]
            effen_data = data[channel][op1][op2]["effen"]
        else:
            # Load data from files
            corr_data = pd.read_csv(self.proj_files_handler.corr_estimates_file(corr_name))
            effen_data = pd.read_csv(self.proj_files_handler.effen_estimates_file(corr_name))
        
        # Create correlator plot
        self._create_correlator_plot(corr_data, corr_name, "correlator")
        
        # Create effective energy plot
        self._create_correlator_plot(effen_data, corr_name, "effective_energy")
    
    def _create_correlator_plot(self, data: pd.DataFrame, corr_name: str, plot_type: str) -> None:
        """
        Create a correlator or effective energy plot.
        
        Args:
            data: DataFrame with correlator data
            corr_name: Name of the correlator
            plot_type: Type of plot ("correlator" or "effective_energy")
        """
        self.plh.clf()
        
        # Extract time values and estimates
        times = data['time'] if 'time' in data.columns else data.index
        values = data['value'] if 'value' in data.columns else data.iloc[:, 1]
        errors = data['error'] if 'error' in data.columns else data.iloc[:, 2]
        
        # Create the plot
        plt.errorbar(times, values, yerr=errors, fmt='o', capsize=3)
        plt.xlabel('Time')
        
        if plot_type == "correlator":
            plt.ylabel('Correlator')
            plt.yscale('log')
            plt.title(f'Correlator: {corr_name}')
            file_prefix = "corr"
        else:
            plt.ylabel('Effective Energy')
            plt.title(f'Effective Energy: {corr_name}')
            file_prefix = "effen"
        
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            pdf_file = getattr(self.proj_files_handler, f'{file_prefix}_plot_file')(corr_name, "pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
        
        if self.plot_config.get('create_pickles', True):
            pickle_file = getattr(self.proj_files_handler, f'{file_prefix}_plot_file')(corr_name, "pickle")
            self.plh.save_pickle(pickle_file)
    
    def plot_spectrum_fits(self, fit_results: Dict[str, Any], channels: List[Any]) -> None:
        """
        Plot spectrum fit results.
        
        Args:
            fit_results: Dictionary of fit results
            channels: List of physics channels
        """
        logging.info("Creating spectrum fit plots...")
        
        for channel in channels:
            if channel in fit_results:
                self._plot_channel_fits(channel, fit_results[channel])
    
    def _plot_channel_fits(self, channel: Any, channel_results: Dict[str, Any]) -> None:
        """
        Plot fit results for a single channel.
        
        Args:
            channel: Physics channel
            channel_results: Fit results for this channel
        """
        for operator, fit_result in channel_results.items():
            if hasattr(fit_result, 'success') and fit_result.success:
                self._plot_operator_fit(channel, operator, fit_result)
    
    def _plot_operator_fit(self, channel: Any, operator: Any, fit_result: FitResult) -> None:
        """
        Plot fit result for a single operator.
        
        Args:
            channel: Physics channel
            operator: Operator
            fit_result: Fit result
        """
        self.plh.clf()
        
        # This would involve plotting the correlator data with the fit overlay
        # Implementation depends on having access to the original correlator data
        # and the fit function
        
        plt.title(f'Fit: {channel} - {operator}')
        plt.xlabel('Time')
        plt.ylabel('Correlator')
        
        # Add fit information as text
        fit_text = f'Energy: {fit_result.energy_value:.4f} ± {fit_result.energy_error:.4f}\n'
        fit_text += f'χ²/dof: {fit_result.chisq_dof:.2f}\n'
        fit_text += f'Quality: {fit_result.quality:.3f}'
        
        plt.text(0.05, 0.95, fit_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save the plot
        plot_name = f"{channel}_{operator}_fit"
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), f"{plot_name}.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
    
    def plot_spectrum_summary(self, spectrum_levels: List[SpectrumLevel]) -> None:
        """
        Create a summary plot of the entire spectrum.
        
        Args:
            spectrum_levels: List of spectrum levels to plot
        """
        logging.info("Creating spectrum summary plot...")
        
        self.plh.clf()
        
        # Extract energies and errors
        energies = [level.fit_result.energy_value for level in spectrum_levels if level.fit_result.success]
        errors = [level.fit_result.energy_error for level in spectrum_levels if level.fit_result.success]
        
        # Create horizontal error bars for energy levels
        y_positions = range(len(energies))
        plt.errorbar(energies, y_positions, xerr=errors, fmt='o', capsize=3)
        
        plt.xlabel('Energy')
        plt.ylabel('Level')
        plt.title('Energy Spectrum')
        plt.grid(True, alpha=0.3)
        
        # Add level labels
        for i, level in enumerate([l for l in spectrum_levels if l.fit_result.success]):
            plt.text(level.fit_result.energy_value, i, f'  {level.channel}', 
                    verticalalignment='center')
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), "spectrum_summary.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
        
        if self.plot_config.get('create_pickles', True):
            pickle_file = os.path.join(self.proj_files_handler.plot_dir(), "spectrum_summary.pickle")
            self.plh.save_pickle(pickle_file)
    
    def plot_energy_shifts(self, energy_shifts: Dict[str, float], 
                          shift_errors: Dict[str, float]) -> None:
        """
        Plot energy shifts relative to non-interacting levels.
        
        Args:
            energy_shifts: Dictionary of energy shifts
            shift_errors: Dictionary of energy shift errors
        """
        logging.info("Creating energy shift plot...")
        
        self.plh.clf()
        
        channels = list(energy_shifts.keys())
        shifts = list(energy_shifts.values())
        errors = [shift_errors.get(ch, 0.0) for ch in channels]
        
        y_positions = range(len(channels))
        plt.errorbar(shifts, y_positions, xerr=errors, fmt='s', capsize=3)
        
        plt.xlabel('Energy Shift')
        plt.ylabel('Channel')
        plt.title('Energy Shifts from Non-Interacting Levels')
        plt.yticks(y_positions, channels)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), "energy_shifts.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
    
    def create_summary_document(self, title: str = "Spectrum Analysis Summary") -> None:
        """
        Create a summary document with all plots.
        
        Args:
            title: Title for the summary document
        """
        if not self.plot_config.get('create_summary', True):
            return
        
        logging.info("Creating summary document...")
        self.plh.create_summary_doc(title)
    
    def add_section_to_summary(self, section_title: str, plot_files: List[str], 
                             index: int = 0) -> None:
        """
        Add a section to the summary document.
        
        Args:
            section_title: Title of the section
            plot_files: List of plot files to include
            index: Index for multiple summary documents
        """
        self.plh.append_section(section_title, index)
        
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                self.plh.add_plot_to_section(plot_file, index)
    
    def finalize_summary(self, output_name: str = "summary") -> None:
        """
        Finalize and save the summary document.
        
        Args:
            output_name: Base name for the output file
        """
        if self.plot_config.get('create_summary', True):
            summary_file = os.path.join(self.proj_files_handler.plot_dir(), f"{output_name}.pdf")
            self.plh.finalize_summary_doc(summary_file)
            logging.info(f"Summary document saved to {summary_file}")


class FitQualityPlotter:
    """
    Specialized plotter for fit quality assessment.
    
    This class creates plots specifically for assessing the quality of
    correlator fits, including tmin/tmax dependence and stability plots.
    """
    
    def __init__(self, proj_files_handler, plot_config: Dict[str, Any]):
        """
        Initialize the fit quality plotter.
        
        Args:
            proj_files_handler: Project file handler
            plot_config: Plotting configuration
        """
        self.proj_files_handler = proj_files_handler
        self.plot_config = plot_config
        self.plh = ph.PlottingHandler()
    
    def plot_tmin_dependence(self, tmin_results: Dict[str, Any], 
                           operator: Any, channel: Any) -> None:
        """
        Plot the dependence of fit results on tmin.
        
        Args:
            tmin_results: Results from tmin variation
            operator: Operator being analyzed
            channel: Physics channel
        """
        self.plh.clf()
        
        # Extract tmin values and corresponding fit results
        tmin_values = []
        energies = []
        energy_errors = []
        chisq_values = []
        
        for tmin, result in tmin_results.items():
            if isinstance(tmin, int) and result is not None:
                tmin_values.append(tmin)
                energies.append(result['energy_value'])
                energy_errors.append(result['energy_error'])
                chisq_values.append(result.get('chisq_dof', 0))
        
        # Create subplot for energy vs tmin
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        ax1.errorbar(tmin_values, energies, yerr=energy_errors, fmt='o-', capsize=3)
        ax1.set_xlabel('tmin')
        ax1.set_ylabel('Energy')
        ax1.set_title(f'Energy vs tmin: {channel} - {operator}')
        ax1.grid(True, alpha=0.3)
        
        # Create subplot for chi-squared vs tmin
        ax2.plot(tmin_values, chisq_values, 'o-')
        ax2.set_xlabel('tmin')
        ax2.set_ylabel('χ²/dof')
        ax2.set_title('Fit Quality vs tmin')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='χ²/dof = 1')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_name = f"{channel}_{operator}_tmin_dependence"
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), f"{plot_name}.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
    
    def plot_stability_analysis(self, stability_results: Dict[str, Any]) -> None:
        """
        Plot stability analysis results.
        
        Args:
            stability_results: Results from stability analysis
        """
        self.plh.clf()
        
        # Extract parameter variations and corresponding results
        param_values = []
        energies = []
        energy_errors = []
        
        for param_val, result in stability_results.items():
            if result is not None and result.get('success', False):
                param_values.append(param_val)
                energies.append(result['energy_value'])
                energy_errors.append(result['energy_error'])
        
        if not param_values:
            logging.warning("No valid stability results to plot")
            return
        
        # Create stability plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(param_values, energies, yerr=energy_errors, fmt='o-', capsize=3)
        plt.xlabel('Parameter Value')
        plt.ylabel('Energy')
        plt.title('Fit Stability Analysis')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at mean value
        if len(energies) > 1:
            mean_energy = np.mean(energies)
            plt.axhline(y=mean_energy, color='r', linestyle='--', alpha=0.5, 
                       label=f'Mean: {mean_energy:.4f}')
            plt.legend()
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), "stability_analysis.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
            logging.info(f"Stability plot saved to {pdf_file}")


class ComparisonPlotter:
    """
    Plotter for comparing different analysis results.
    
    This class creates plots that compare results from different
    configurations, ensembles, or analysis methods.
    """
    
    def __init__(self, proj_files_handler, plot_config: Dict[str, Any]):
        """
        Initialize the comparison plotter.
        
        Args:
            proj_files_handler: Project file handler
            plot_config: Plotting configuration
        """
        self.proj_files_handler = proj_files_handler
        self.plot_config = plot_config
        self.plh = ph.PlottingHandler()
    
    def compare_spectra(self, spectrum_sets: Dict[str, List[SpectrumLevel]], 
                       labels: List[str]) -> None:
        """
        Compare multiple spectrum results.
        
        Args:
            spectrum_sets: Dictionary of spectrum sets to compare
            labels: Labels for each spectrum set
        """
        self.plh.clf()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(spectrum_sets)))
        
        for i, (label, spectrum) in enumerate(spectrum_sets.items()):
            energies = [level.fit_result.energy_value for level in spectrum if level.fit_result.success]
            errors = [level.fit_result.energy_error for level in spectrum if level.fit_result.success]
            y_positions = [j + i * 0.1 for j in range(len(energies))]
            
            plt.errorbar(energies, y_positions, xerr=errors, fmt='o', 
                        color=colors[i], label=label, capsize=3)
        
        plt.xlabel('Energy')
        plt.ylabel('Level')
        plt.title('Spectrum Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), "spectrum_comparison.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
    
    def generate_operator_overlaps_plots(self, channel: Any, overlap_data: Dict[str, Any]) -> None:
        """
        Generate operator overlap plots for a channel.
        
        Args:
            channel: Physics channel
            overlap_data: Dictionary containing overlap information
        """
        self.plh.clf()
        
        # This would create plots showing how original operators overlap
        # with the rotated eigenstates
        logging.info(f"Creating operator overlap plots for channel {channel}")
        
        # Implementation would depend on the specific overlap data structure
        # and would create bar charts or heatmaps showing overlap magnitudes
        
        plt.title(f'Operator Overlaps: {channel}')
        plt.xlabel('Eigenstate')
        plt.ylabel('Overlap Magnitude')
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), f"overlaps_{channel}.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
    
    def summary_spectrum_plot(self, spectrum_data: Dict[str, Any]) -> None:
        """
        Create a summary plot of the entire spectrum analysis.
        
        Args:
            spectrum_data: Dictionary containing spectrum information
        """
        self.plh.clf()
        
        logging.info("Creating summary spectrum plot")
        
        # Create comprehensive spectrum plot showing all channels and levels
        plt.figure(figsize=(12, 8))
        
        # Implementation would plot all energy levels across channels
        # with proper color coding and labeling
        
        plt.title('Complete Energy Spectrum')
        plt.xlabel('Channel')
        plt.ylabel('Energy')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), "summary_spectrum.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
    
    def summary_dElab_spectrum_plot(self, energy_shift_data: Dict[str, Any], 
                                  max_levels: Optional[int] = None, 
                                  color_coded: bool = False, 
                                  certainties: bool = False) -> None:
        """
        Create a summary plot of energy shifts in lab frame.
        
        Args:
            energy_shift_data: Dictionary containing energy shift information
            max_levels: Maximum number of levels to plot
            color_coded: Whether to use color coding
            certainties: Whether to include certainty information
        """
        self.plh.clf()
        
        logging.info("Creating energy shift summary plot")
        
        plt.figure(figsize=(12, 8))
        
        # Implementation would plot energy shifts relative to non-interacting levels
        # with proper error bars and color coding if requested
        
        plt.title('Energy Shifts from Non-Interacting Levels')
        plt.xlabel('Level')
        plt.ylabel('ΔE (lab frame)')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), "summary_energy_shifts.pdf")
            plt.savefig(pdf_file, bbox_inches='tight')
    
    def tmin_fit_plot(self, energy_type: str, results: Dict[str, Any], 
                     channel: Any, operator: Any, op_name: str, 
                     tmin_plot: bool = True) -> None:
        """
        Create tmin dependence plots for fit quality assessment.
        
        Args:
            energy_type: Type of energy being plotted
            results: Fit results dictionary
            channel: Physics channel
            operator: Operator being analyzed
            op_name: Operator name
            tmin_plot: Whether this is a tmin plot (vs tmax)
        """
        self.plh.clf()
        
        logging.info(f"Creating {'tmin' if tmin_plot else 'tmax'} fit plot for {op_name}")
        
        # Implementation would plot energy vs tmin/tmax with error bars
        # and chi-squared information
        
        plt.figure(figsize=(10, 8))
        
        # Create subplots for energy and chi-squared
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Energy plot
        ax1.set_xlabel('tmin' if tmin_plot else 'tmax')
        ax1.set_ylabel(f'Energy ({energy_type})')
        ax1.set_title(f'{"tmin" if tmin_plot else "tmax"} Dependence: {channel} - {op_name}')
        ax1.grid(True, alpha=0.3)
        
        # Chi-squared plot
        ax2.set_xlabel('tmin' if tmin_plot else 'tmax')
        ax2.set_ylabel('χ²/dof')
        ax2.set_title('Fit Quality')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='χ²/dof = 1')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the plot
        if self.plot_config.get('create_pdfs', True):
            plot_type = 'tmin' if tmin_plot else 'tmax'
            pdf_file = os.path.join(self.proj_files_handler.plot_dir(), 
                                  f"{plot_type}_dependence_{channel}_{op_name}.pdf")
            plt.savefig(pdf_file, bbox_inches='tight') 