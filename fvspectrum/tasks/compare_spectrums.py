"""
Compare spectrums task implementation.

This module provides the refactored implementation of the compare spectrums task,
which compares spectrum analysis results from different configurations and creates
comparison plots and summary documents.
"""

import logging
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

import general.task_manager as tm
import general.particles
import fvspectrum.sigmond_util as sigmond_util
import fvspectrum.spectrum_plotting_settings.settings as psettings
from fvspectrum.core.base_task import BaseSpectrumTask
from fvspectrum.plotting.spectrum_plotter import ComparisonPlotter


# Documentation string for the task
TASK_DOCUMENTATION = '''
compare - using the fit results of the fit task, plots
    the spectrums side by side for comparisons and puts in summary document

This task compares spectrum analysis results from different configurations,
such as different pivot parameters, rebin values, or user-defined tags.
It creates side-by-side comparison plots and summary documents.

Configuration Parameters:
------------------------
general:
  ensemble_id: cls21_c103       # Required: Ensemble identifier
  project_dir: /path/to/project # Required: Project directory path

compare:              # Required task configuration
  compare_plots:                # Required: List of comparison plot types
  - compare_gevp:               # Optional: Compare different GEVP configurations
      gevp_values:              # Required for "compare_gevp"
      - t0: 8                   # Required: Metric time
        tD: 16                  # Required: Diagonalize time
        tN: 5                   # Required: Normalize time
        pivot_type: 0           # Optional: Pivot type (default: 0)
      - t0: 8
        tD: 18
        tN: 5
      rebin: 1                  # Optional: Rebin factor (default: 1)
      run_tag: ''               # Optional: Run tag (default: '')
      sampling_mode: J          # Optional: Sampling mode (default: J)
  - compare_files: []           # Optional: Compare specific files (default: [])
  - compare_rebin:              # Optional: Compare different rebin values
      rebin_values: [1, 2, 4]   # Required: List of rebin values to compare
      run_tag: ''               # Optional: Run tag (default: '')
      sampling_mode: J          # Optional: Sampling mode (default: J)
      pivot_type: 0             # Optional: Pivot type (default: 0)
      t0: 8                     # Required: Metric time
      tN: 5                     # Required: Normalize time
      tD: 18                    # Required: Diagonalize time
  - compare_tags:               # Optional: Compare different user tags
      filetags: ['tag1', 'tag2'] # Required: List of file tags to compare
      sampling_mode: J          # Optional: Sampling mode (default: J)
      pivot_type: 0             # Optional: Pivot type (default: 0)
      t0: 8                     # Required: Metric time
      tN: 5                     # Required: Normalize time
      tD: 18                    # Required: Diagonalize time
      rebin: 1                  # Optional: Rebin factor (default: 1)
  figheight: 8                  # Optional: Figure height (default: 8)
  figwidth: 15                  # Optional: Figure width (default: 15)
  plot: true                    # Required: Create plots
  plot_deltaE: true             # Optional: Plot energy shifts (default: true)
  reference_particle: P         # Optional: Reference particle for normalization
  max_level: 1000               # Optional: Maximum level to include (default: 1000)

Output:
-------
- Comparison plots showing spectra side by side
- Error analysis plots for reference particles
- Energy shift plots (if requested)
- Summary document with all comparison plots
'''


class CompareSpectrumsTask(BaseSpectrumTask):
    """
    Task for comparing spectrum analysis results.
    
    This task compares spectrum results from different configurations,
    creating side-by-side comparison plots and analysis summaries.
    """
    
    @property
    def info(self) -> str:
        """Return task documentation."""
        return TASK_DOCUMENTATION
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for compare spectrums task."""
        defaults = super()._get_default_parameters()
        defaults.update({
            'figwidth': 15,
            'figheight': 8,
            'reference_particle': None,
            'plot_deltaE': True,
            'max_level': 1000,
        })
        return defaults
    
    def _validate_parameters(self) -> None:
        """Validate task-specific parameters."""
        super()._validate_parameters()
        
        # Check for required compare_plots
        if 'compare_plots' not in self.task_configs:
            logging.critical("No 'compare_plots' specified in task configuration.")
        
        if not self.params['plot']:
            logging.warning(f"No plots requested, task {self.task_name} does nothing.")
            return
        
        # Set energy key based on reference particle
        self.energy_key = 'ecm'
        if self.params['reference_particle'] is not None:
            self.energy_key = 'ecm_ref'
    
    def run(self) -> None:
        """Execute the compare spectrums task."""
        self._log_task_start("Running spectrum comparison analysis")
        
        # Set up comparison plots
        self.compare_plots = self._setup_comparison_plots()
        
        if not self.compare_plots:
            logging.warning("No valid comparison plots could be generated.")
            return
        
        self._log_task_complete("Spectrum comparison analysis")
    
    def _setup_comparison_plots(self) -> List[Dict[str, str]]:
        """Set up comparison plots from configuration."""
        compare_plots = []
        
        default_plot_configs = {
            "rebin": 1,
            "sampling_mode": 'J',
            "run_tag": "",
        }
        
        for compare_plot in self.task_configs['compare_plots']:
            root = list(compare_plot.keys())[0]
            plot_configs = compare_plot[root]
            sigmond_util.update_params(default_plot_configs, plot_configs)
            
            plot = {}
            
            if root == 'compare_files':
                # User-defined files comparison
                plot = self._setup_files_comparison(plot_configs)
            
            elif root == 'compare_rebin':
                # Compare different rebin values
                plot = self._setup_rebin_comparison(plot_configs)
            
            elif root == 'compare_gevp':
                # Compare different GEVP configurations
                plot = self._setup_gevp_comparison(plot_configs)
            
            elif root == 'compare_tags':
                # Compare different user tags
                plot = self._setup_tags_comparison(plot_configs)
            
            if plot:
                compare_plots.append(plot)
            else:
                logging.warning(f"Could not generate {root} plot.")
        
        return compare_plots
    
    def _setup_files_comparison(self, plot_configs: Dict[str, Any]) -> Dict[str, str]:
        """Set up comparison from user-defined files."""
        # This would be implemented based on user-provided file list
        return plot_configs
    
    def _setup_rebin_comparison(self, plot_configs: Dict[str, Any]) -> Dict[str, str]:
        """Set up comparison of different rebin values."""
        plot = {}
        
        for rebin in plot_configs['rebin_values']:
            dataset_key = rf"$N_{{\textup{{bin}}}}={rebin}$"
            file_tag = ''
            if plot_configs['run_tag']:
                file_tag = '-' + plot_configs['run_tag']
            
            tN = plot_configs['tN']
            t0 = plot_configs['t0']
            tD = plot_configs['tD']
            rotate_type = 'SP'
            if 'pivot_type' in plot_configs and plot_configs['pivot_type']:
                rotate_type = 'RP'
            
            sampling_mode = plot_configs['sampling_mode']
            
            # Try to find the spectrum file
            key = self.proj_files_handler.all_tasks[tm.Task.fit.name].filekey(
                None, rebin, sampling_mode + "-samplings", rotate_type, tN, t0, tD, file_tag
            )
            file = self.proj_files_handler.all_tasks[tm.Task.fit.name].estimates_file(key)
            
            if os.path.isfile(file):
                plot[dataset_key] = file
            else:
                # Try alternative file location
                file2 = self.proj_files_handler.all_tasks[tm.Task.fit.name].samplings_file(
                    False, None, None, rebin, sampling_mode, rotate_type, tN, t0, tD, 'levels' + file_tag
                )
                if os.path.isfile(file2):
                    plot[dataset_key] = file2
                else:
                    logging.warning(f"Could not find either '{file}' or '{file2}' for Nbin={rebin}.")
        
        return plot
    
    def _setup_gevp_comparison(self, plot_configs: Dict[str, Any]) -> Dict[str, str]:
        """Set up comparison of different GEVP configurations."""
        plot = {}
        
        for pivot_set in plot_configs['gevp_values']:
            tN = pivot_set['tN']
            t0 = pivot_set['t0']
            tD = pivot_set['tD']
            rotate_type = 'SP'
            if 'pivot_type' in pivot_set and pivot_set['pivot_type']:
                rotate_type = 'RP'
            
            dataset_key = f"{rotate_type}({tN},{t0},{tD})"
            file_tag = ''
            if plot_configs['run_tag']:
                file_tag = '-' + plot_configs['run_tag']
            
            sampling_mode = plot_configs['sampling_mode']
            rebin = plot_configs['rebin']
            
            # Try to find the spectrum file
            key = self.proj_files_handler.all_tasks[tm.Task.fit.name].filekey(
                None, rebin, sampling_mode + "-samplings", rotate_type, tN, t0, tD, file_tag
            )
            file = self.proj_files_handler.all_tasks[tm.Task.fit.name].estimates_file(key)
            
            if os.path.isfile(file):
                plot[dataset_key] = file
            else:
                # Try alternative file location
                file2 = self.proj_files_handler.all_tasks[tm.Task.fit.name].samplings_file(
                    False, None, None, rebin, sampling_mode, rotate_type, tN, t0, tD, 'levels' + file_tag
                )
                if os.path.isfile(file2):
                    plot[dataset_key] = file2
                else:
                    logging.warning(f"Could not find either '{file}' or '{file2}' for pivot_set={dataset_key}.")
        
        return plot
    
    def _setup_tags_comparison(self, plot_configs: Dict[str, Any]) -> Dict[str, str]:
        """Set up comparison of different user tags."""
        plot = {}
        
        for file_tag in plot_configs['filetags']:
            dataset_key = file_tag
            tN = plot_configs['tN']
            t0 = plot_configs['t0']
            tD = plot_configs['tD']
            rotate_type = 'SP'
            if 'pivot_type' in plot_configs and plot_configs['pivot_type']:
                rotate_type = 'RP'
            
            sampling_mode = plot_configs['sampling_mode']
            rebin = plot_configs['rebin']
            
            # Try to find the spectrum file
            key = self.proj_files_handler.all_tasks[tm.Task.fit.name].filekey(
                None, rebin, sampling_mode + "-samplings", rotate_type, tN, t0, tD, '-' + file_tag
            )
            file = self.proj_files_handler.all_tasks[tm.Task.fit.name].estimates_file(key)
            
            if os.path.isfile(file):
                plot[dataset_key] = file
            else:
                # Try alternative file location
                file2 = self.proj_files_handler.all_tasks[tm.Task.fit.name].samplings_file(
                    False, None, None, rebin, sampling_mode, rotate_type, tN, t0, tD, 'levels-' + file_tag
                )
                if os.path.isfile(file2):
                    plot[dataset_key] = file2
                else:
                    logging.warning(f"Could not find either '{file}' or '{file2}' for run_tag={file_tag}.")
        
        return plot
    
    def plot(self) -> None:
        """Generate comparison plots."""
        if not self._should_create_plots():
            logging.info("No plots requested.")
            return
        
        self._log_task_start("Creating comparison plots")
        
        # Initialize comparison plotter
        plotter = ComparisonPlotter(self.proj_files_handler, self.params)
        plotter.create_summary_document("Compare Spectrums")
        
        # Load datasets
        datasets = self._load_datasets()
        
        # Create comparison plots
        for iplot, plot in enumerate(self.compare_plots):
            self._create_comparison_plot(plotter, plot, datasets, iplot)
        
        # Finalize summary document
        plotter.finalize_summary("comparison_summary")
        
        self._log_task_complete("Comparison plot generation")
    
    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets needed for comparison."""
        datasets = {}
        
        for plot in self.compare_plots:
            for dataset in plot:
                if plot[dataset].endswith(".csv") and plot[dataset] not in datasets:
                    datasets[plot[dataset]] = pd.read_csv(plot[dataset])
        
        return datasets
    
    def _create_comparison_plot(self, plotter: ComparisonPlotter, plot: Dict[str, str], 
                              datasets: Dict[str, pd.DataFrame], iplot: int) -> None:
        """Create a single comparison plot."""
        # Collect particles/channels involved
        particles = self._get_particles_from_plot(plot, datasets)
        
        # Create summary plots section
        plotter.add_section_to_summary(f"Summaries {iplot}", [], 0)
        
        for particle in particles:
            self._create_particle_comparison(plotter, particle, plot, datasets, iplot)
        
        # Create energy shift plots if requested
        if self.params['plot_deltaE']:
            self._create_energy_shift_plots(plotter, particles, plot, datasets, iplot)
    
    def _get_particles_from_plot(self, plot: Dict[str, str], 
                                datasets: Dict[str, pd.DataFrame]) -> List[tuple]:
        """Get list of particles involved in the plot."""
        particles = []
        
        for dataset in plot:
            if plot[dataset].endswith(".csv"):
                df = datasets[plot[dataset]]
                for i, row in df.iterrows():
                    particle = (row['isospin'], row['strangeness'])
                    particles.append(particle)
        
        return list(set(particles))
    
    def _create_particle_comparison(self, plotter: ComparisonPlotter, particle: tuple, 
                                  plot: Dict[str, str], datasets: Dict[str, pd.DataFrame], 
                                  iplot: int) -> None:
        """Create comparison plot for a single particle."""
        plotter.add_section_to_summary(f"I={particle[0]} S={particle[1]}", [], 0)
        
        # Set up error analysis data (for rebin analysis)
        study_particle = False
        error_analysis = {}
        if self.params['reference_particle']:
            if (particle[0] == general.particles.data[self.params['reference_particle']]["isospin"] and
                particle[1] == general.particles.data[self.params['reference_particle']]["strangeness"]):
                study_particle = True
                error_analysis = {
                    "x": [], "val": [], "err": [], "chisqrdof": []
                }
        
        # Create spectrum comparison
        spectrum_data = self._collect_spectrum_data(particle, plot, datasets, error_analysis, study_particle)
        
        if spectrum_data['has_data']:
            # Create and save the comparison plot
            plot_file = self._create_spectrum_plot(plotter, spectrum_data, particle, iplot)
            plotter.add_section_to_summary(f"I={particle[0]} S={particle[1]}", [plot_file], 0)
        
        # Generate error analysis plot if applicable
        if study_particle and error_analysis['x']:
            error_plot_file = self._create_error_analysis_plot(plotter, error_analysis, iplot)
            plotter.add_section_to_summary("Error Analysis", [error_plot_file], 0)
    
    def _collect_spectrum_data(self, particle: tuple, plot: Dict[str, str], 
                             datasets: Dict[str, pd.DataFrame], error_analysis: Dict, 
                             study_particle: bool) -> Dict[str, Any]:
        """Collect spectrum data for plotting."""
        spectrum_data = {
            'datasets': [],
            'has_data': False,
            'energy_key': self.energy_key
        }
        
        # Determine energy key
        this_energy_key = self.energy_key
        for dataset in plot:
            df = datasets[plot[dataset]]
            if f'{self.energy_key} value' not in df:
                this_energy_key = 'ecm'
        spectrum_data['energy_key'] = this_energy_key
        
        # Collect irreps
        irreps = []
        for dataset in plot:
            df = datasets[plot[dataset]]
            for i, row in df[(df['isospin'] == particle[0]) & (df['strangeness'] == particle[1])].iterrows():
                irrep = (row['irrep'], row['momentum'])
                if irrep not in irreps:
                    irreps.append(irrep)
        
        # Sort irreps
        irreps.sort(key=lambda x: x[1] + psettings.alphabetical[x[0]])
        spectrum_data['irreps'] = irreps
        
        # Collect data for each dataset
        for id, dataset in enumerate(plot):
            df = datasets[plot[dataset]]
            
            # Determine levels key
            levels_key = 'fit level'
            if 'fit level' in df and isinstance(df['fit level'].iloc[0], np.float64):
                levels_key = 'rotate level'
            
            # Collect spectrum data
            indexes, levels, errs = [], [], []
            for i, row in df[(df['isospin'] == particle[0]) & (df['strangeness'] == particle[1])].iterrows():
                if (not np.isnan(row[f'{this_energy_key} value']) and 
                    row[levels_key] <= self.params['max_level']):
                    irrep = (row['irrep'], row['momentum'])
                    indexes.append(irreps.index(irrep))
                    levels.append(row[f'{this_energy_key} value'])
                    errs.append(row[f'{this_energy_key} error'])
                    spectrum_data['has_data'] = True
                    
                    # Collect error analysis data
                    if (study_particle and row['momentum'] == 0 and row[levels_key] == 0):
                        error_analysis['x'].append(dataset)
                        error_analysis['val'].append(row[f'{this_energy_key} value'])
                        error_analysis['err'].append(row[f'{this_energy_key} error'])
                        error_analysis['chisqrdof'].append(row['chisqrdof'])
            
            spectrum_data['datasets'].append({
                'name': dataset,
                'indexes': indexes,
                'levels': levels,
                'errors': errs,
                'id': id,
                'total': len(plot)
            })
        
        return spectrum_data
    
    def _create_spectrum_plot(self, plotter: ComparisonPlotter, spectrum_data: Dict[str, Any], 
                            particle: tuple, iplot: int) -> str:
        """Create and save spectrum comparison plot."""
        plotter.plh.clf()
        
        # Plot each dataset
        for dataset_info in spectrum_data['datasets']:
            if dataset_info['levels']:
                if spectrum_data['energy_key'] == "ecm":
                    plotter.plh.summary_plot(
                        dataset_info['indexes'], dataset_info['levels'], dataset_info['errors'],
                        spectrum_data['irreps'], None, [], dataset_info['name'], 
                        dataset_info['id'], dataset_info['total']
                    )
                else:
                    plotter.plh.summary_plot(
                        dataset_info['indexes'], dataset_info['levels'], dataset_info['errors'],
                        spectrum_data['irreps'], self.params['reference_particle'], [], 
                        dataset_info['name'], dataset_info['id'], dataset_info['total']
                    )
        
        # Save plot
        strangeness = particle[1]
        if particle[1] < 0:
            strangeness = f"m{-particle[1]}"
        
        filekey = f"{iplot}-I{particle[0]}_S{strangeness}"
        plot_file = self.proj_files_handler.summary_plot_file("pdf", filekey)
        plotter.plh.save_pdf(plot_file)
        logging.info(f"Comparison plot saved to '{plot_file}'.")
        
        return plot_file
    
    def _create_error_analysis_plot(self, plotter: ComparisonPlotter, 
                                  error_analysis: Dict, iplot: int) -> str:
        """Create error analysis plot."""
        plotter.plh.clf()
        
        relerrs = np.array(error_analysis['err']) / np.array(error_analysis['val'])
        relerrs /= relerrs[0]
        
        plotter.plh.show_trend(error_analysis['x'], relerrs, "$R_N/R_1$")
        plotter.plh.show_trend(error_analysis['x'], error_analysis['chisqrdof'], 
                              r"$\chi^2/\textup{dof}$", True)
        
        plot_file = os.path.join(self.proj_files_handler.plot_dir("pdfs"), 
                                f"error_analysis-{iplot}.pdf")
        plotter.plh.save_pdf(plot_file)
        
        return plot_file
    
    def _create_energy_shift_plots(self, plotter: ComparisonPlotter, particles: List[tuple], 
                                 plot: Dict[str, str], datasets: Dict[str, pd.DataFrame], 
                                 iplot: int) -> None:
        """Create energy shift comparison plots."""
        plotter.add_section_to_summary(f"Energy shifts {iplot}", [], 0)
        
        for particle in particles:
            plotter.add_section_to_summary(f"I={particle[0]} S={particle[1]}", [], 0)
            
            # Collect irreps for this particle
            irreps = self._collect_irreps_for_shifts(particle, plot, datasets)
            
            # Create plots for each momentum
            for mom in irreps:
                shift_plot_file = self._create_momentum_shift_plot(
                    plotter, particle, mom, irreps[mom], plot, datasets, iplot
                )
                if shift_plot_file:
                    plotter.add_section_to_summary(f"PSQ={mom}", [shift_plot_file], 0)
    
    def _collect_irreps_for_shifts(self, particle: tuple, plot: Dict[str, str], 
                                 datasets: Dict[str, pd.DataFrame]) -> Dict[int, List[str]]:
        """Collect irreps for energy shift plots."""
        irreps = {}
        
        for dataset in plot:
            df = datasets[plot[dataset]]
            levels_key = 'fit level'
            if 'fit level' in df and isinstance(df['fit level'].iloc[0], np.float64):
                levels_key = 'rotate level'
            
            if 'dElab value' not in df:
                continue
            
            for i, row in df[(df['isospin'] == particle[0]) & (df['strangeness'] == particle[1])].iterrows():
                if row['momentum'] not in irreps:
                    irreps[row['momentum']] = []
                
                if row[levels_key] > self.params['max_level']:
                    continue
                
                if row['irrep'] not in irreps[row['momentum']]:
                    irreps[row['momentum']].append(row['irrep'])
                elif (irreps[row['momentum']].index(row['irrep']) + row[levels_key] >= 
                      len(irreps[row['momentum']])):
                    irreps[row['momentum']].append(row['irrep'])
                elif (irreps[row['momentum']][irreps[row['momentum']].index(row['irrep']) + 
                                              int(row[levels_key])] != row['irrep']):
                    irreps[row['momentum']].insert(
                        irreps[row['momentum']].index(row['irrep']), row['irrep']
                    )
        
        # Sort irreps
        for irrep in irreps:
            irreps[irrep].sort(key=lambda x: psettings.alphabetical[x])
        
        return irreps
    
    def _create_momentum_shift_plot(self, plotter: ComparisonPlotter, particle: tuple, 
                                  mom: int, irrep_list: List[str], plot: Dict[str, str], 
                                  datasets: Dict[str, pd.DataFrame], iplot: int) -> Optional[str]:
        """Create energy shift plot for a specific momentum."""
        plotter.plh.clf()
        nothing = True
        
        for id, dataset in enumerate(plot):
            df = datasets[plot[dataset]]
            levels_key = 'fit level'
            if 'fit level' in df and isinstance(df['fit level'].iloc[0], np.float64):
                levels_key = 'rotate level'
            
            if 'dElab value' not in df:
                continue
            
            indexes, levels, errs, split_irreps = [], [], [], []
            
            for i, row in df[(df['isospin'] == particle[0]) & 
                           (df['strangeness'] == particle[1]) & 
                           (df['momentum'] == mom)].iterrows():
                if (not np.isnan(row['dElab value']) and 
                    row[levels_key] <= self.params['max_level']):
                    irrep = row['irrep']
                    fit_level = row[levels_key]
                    split_irreps.append((irrep, mom, fit_level))
                    indexes.append(irrep_list.index(irrep) + fit_level)
                    levels.append(row['dElab value'])
                    errs.append(row['dElab error'])
                    nothing = False
            
            if levels:
                plotter.plh.summary_plot(
                    indexes, levels, errs, split_irreps, None, [], 
                    dataset, id, len(plot), True
                )
        
        if not nothing:
            strangeness = particle[1]
            if particle[1] < 0:
                strangeness = f"m{-particle[1]}"
            
            filekey = f"{iplot}-I{particle[0]}_S{strangeness}_PSQ{mom}"
            plot_file = self.proj_files_handler.summary_plot_file("pdf", filekey)
            plotter.plh.save_pdf(plot_file)
            logging.info(f"Comparison plot saved to '{plot_file}'.")
            return plot_file
        
        return None 