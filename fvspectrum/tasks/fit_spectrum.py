"""
Fit spectrum task implementation.

This module provides the refactored implementation of the fit spectrum task,
which fits single hadron and/or rotated correlators to determine the energy
spectrum of each channel of interest.
"""

import logging
import os
import yaml
import pandas as pd
import numpy as np
import h5py
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from ordered_set import OrderedSet
import regex
import tqdm
import time
from datetime import datetime
from multiprocessing import Process

import sigmond
import fvspectrum.spectrum_plotting_settings.settings as psettings
import general.plotting_handler as ph
import fvspectrum.sigmond_util as sigmond_util
from fvspectrum.core.base_task import CorrelatorAnalysisTask
from fvspectrum.core.data_structures import ObservableType, FitConfiguration, FitResult
from fvspectrum.analysis.correlator_processor import CorrelatorProcessor
from fvspectrum.fitting.spectrum_fitter import SpectrumFitter, SingleHadronFitter, InteractingFitter
from fvspectrum.plotting.spectrum_plotter import SpectrumPlotter
from sigmond_scripts import util as utils
from sigmond_scripts import operator
from sigmond_scripts import fit_info, sigmond_info, sigmond_input
import general.task_manager as tm


# Documentation string for the task
TASK_DOCUMENTATION = '''
fit - task for fitting the single hadron and/or rotated correlators in order to determine
    energy spectrum of each channel of interest. If rotated, operator overlaps on the original
    operators are computed and plotted as well.

Configuration Parameters:
------------------------
general:
  ensemble_id: cls21_c103               # Required: Ensemble identifier
  project_dir: /path/to/project         # Required: Project directory path
  sampling_info:                        # Optional: Statistical sampling configuration
    mode: Jackknife                     # Default: Jackknife (or Bootstrap)
  tweak_ensemble:                       # Optional: Ensemble modifications
    omissions: []                       # Default: [] - configurations to omit
    rebin: 1                            # Default: 1 - rebinning factor

fit:                           # Required task configuration
  default_corr_fit:                     # Required unless both default_interacting_corr_fit and default_noninteracting_corr_fit are specified
    model: 1-exp                        # Required: Fit model
    tmin: 15                            # Required: Minimum time for fit
    tmax: 25                            # Required: Maximum time for fit
    exclude_times: []                   # Optional: Times to exclude (default: [])
    initial_params: {}                  # Optional: Initial parameter values (default: {})
    noise_cutoff: 0.0                   # Optional: Noise cutoff (default: 0.0)
    priors: {}                          # Optional: Prior constraints (default: {})
    ratio: false                        # Optional: Use ratio fits (default: false)
    tmin_plots: []                      # Optional: tmin variation plots
    tmax_plots: []                      # Optional: tmax variation plots
  reference_particle: pi                # Optional: Reference particle for normalization
  default_noninteracting_corr_fit: null # Optional: Default for non-interacting fits
  default_interacting_corr_fit: null    # Optional: Default for interacting fits
  correlator_fits: {}                   # Optional: Operator-specific fit configurations
  single_hadrons:                       # Required for ratio fits
    pi:                                 # Single hadron name
    - operator_name                     # List of operators ordered by momentum
  single_hadrons_ratio: {}              # Optional: Override for ratio fits
  non_interacting_levels: {}            # Optional: Non-interacting level definitions
  averaged_input_correlators_dir: null # Optional: Input directory for averaged data
  compute_overlaps: true                # Optional: Compute operator overlaps (default: true)
  correlated: true                      # Optional: Use correlated fits (default: true)
  create_pdfs: true                     # Optional: Generate PDF plots (default: true)
  create_pickles: true                  # Optional: Generate pickle files (default: true)
  create_summary: true                  # Optional: Generate summary document (default: true)
  do_interacting_fits: true             # Optional: Perform interacting fits (default: true)
  figheight: 6                          # Optional: Figure height (default: 6)
  figwidth: 8                           # Optional: Figure width (default: 8)
  generate_estimates: true              # Optional: Generate CSV estimates (default: true)
  minimizer_info:                       # Optional: Minimizer configuration
    chisquare_rel_tol: 0.0001           # Default: 0.0001
    max_iterations: 2000                # Default: 2000
    minimizer: lmder                    # Default: lmder
    parameter_rel_tol: 1.0e-06          # Default: 1.0e-06
    verbosity: low                      # Default: low
  non_interacting_energy_sums: false    # Optional: Use energy sums (default: false)
  pivot_file: null                      # Optional: Pivot file path
  plot: true                            # Optional: Create plots (default: true)
  precompute: true                      # Optional: Precompute correlators (default: true)
  rotated_input_correlators_dir: null   # Optional: Input directory for rotated data
  run_tag: ""                           # Optional: Unique run identifier (default: "")
  rotate_run_tag: ""                    # Optional: Rotation run tag (default: "")
  thresholds: []                        # Optional: Threshold definitions
  use_rotated_samplings: true           # Optional: Use rotated samplings (default: true)
  used_averaged_bins: true              # Optional: Use averaged bins (default: true)
  tN: null                              # Optional: Normalize time (auto-detected)
  t0: null                              # Optional: Metric time (auto-detected)
  tD: null                              # Optional: Diagonalize time (auto-detected)
  pivot_type: null                      # Optional: Pivot type (auto-detected)

Output:
-------
- Fitted spectrum energy levels (HDF5 format)
- Fit parameter files with detailed fit results
- CSV files with spectrum estimates and effective energies
- PDF/pickle plots of fitted spectra and fit quality
- Operator overlap plots (if computed)
- Summary document with all plots and fit information
'''


class ObservableType(Enum):
    """Enumeration of observable types for spectrum fitting."""
    DELAB = "dElab"          # Energy shift in lab frame
    ELAB = "elab"            # Energy in lab frame
    ECM = "ecm"              # Energy in center of momentum frame
    ECM_REF = "ecm_ref"      # Energy normalized by reference particle
    AMP = "amp"              # Amplitude
    DELAB_REF = "dElab_ref"  # Energy shift normalized by reference particle


class FitSpectrumTask(CorrelatorAnalysisTask):
    """
    Task for fitting correlator spectra.
    
    This task fits single hadron and/or rotated correlators to determine
    energy spectra, including support for ratio fits and operator overlap calculations.
    """
    
    @property
    def info(self) -> str:
        """Return task documentation."""
        return TASK_DOCUMENTATION
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for fit spectrum task."""
        defaults = super()._get_default_parameters()
        defaults.update({
            'used_averaged_bins': True,
            'use_rotated_samplings': True,
            'precompute': True,
            'correlated': True,
            'compute_overlaps': True,
            'do_interacting_fits': True,
            'non_interacting_energy_sums': False,
            'reference_particle': None,
            'run_tag': "",
            'rotate_run_tag': "",
            'pivot_type': None,
            'tN': None,
            't0': None,
            'tD': None,
            'minimizer_info': {
                'minimizer': 'lmder',
                'parameter_rel_tol': 1e-6,
                'chisquare_rel_tol': 1e-4,
                'max_iterations': 2000,
                'verbosity': 'low',
            },
            'correlator_fits': {},
            'single_hadrons': {},
            'single_hadrons_ratio': {},
            'non_interacting_levels': {},
            'thresholds': [],
        })
        return defaults
    
    def _validate_parameters(self) -> None:
        """Validate task-specific parameters."""
        super()._validate_parameters()
        
        # Check for required fit configurations
        has_default = 'default_corr_fit' in self.task_configs
        has_interacting = 'default_interacting_corr_fit' in self.task_configs
        has_noninteracting = 'default_noninteracting_corr_fit' in self.task_configs
        
        if not has_default and not (has_interacting and has_noninteracting):
            logging.critical(
                "Must specify either 'default_corr_fit' or both "
                "'default_interacting_corr_fit' and 'default_noninteracting_corr_fit'"
            )
        
        # Set up fit configurations
        if has_interacting and has_noninteracting:
            self.default_interacting_fit = self._validate_fit_config(
                self.task_configs['default_interacting_corr_fit'], 'default_interacting_corr_fit'
            )
            self.default_noninteracting_fit = self._validate_fit_config(
                self.task_configs['default_noninteracting_corr_fit'], 'default_noninteracting_corr_fit'
            )
            self.default_fit = None
        else:
            self.default_fit = self._validate_fit_config(
                self.task_configs['default_corr_fit'], 'default_corr_fit'
            )
            self.default_interacting_fit = None
            self.default_noninteracting_fit = None
        
        # Validate plot settings
        if not any([self.params['create_pdfs'], self.params['create_pickles'], self.params['create_summary']]):
            self.params['plot'] = False
            logging.warning("All plot output options disabled. Setting plot=False.")
    
    def _validate_fit_config(self, fit_config: Dict[str, Any], config_name: str) -> FitConfiguration:
        """Validate and convert fit configuration to FitConfiguration object."""
        required_params = ['model', 'tmin', 'tmax']
        for param in required_params:
            if param not in fit_config:
                logging.critical(f"Missing required parameter '{param}' in '{config_name}'")
        
        return FitConfiguration(
            model=fit_config['model'],
            tmin=fit_config['tmin'],
            tmax=fit_config['tmax'],
            exclude_times=fit_config.get('exclude_times', []),
            initial_params=fit_config.get('initial_params', {}),
            noise_cutoff=fit_config.get('noise_cutoff', 0.0),
            priors=fit_config.get('priors', {}),
            ratio=fit_config.get('ratio', False),
            tmin_plots=fit_config.get('tmin_plots', []),
            tmax_plots=fit_config.get('tmax_plots', [])
        )
    
    def _setup_data_handlers(self) -> None:
        """Set up data handlers for spectrum fitting."""
        # Set up sampling mode
        if self.project_handler.project_info.sampling_info.isJackknifeMode():
            sampling_mode = 'J'
        else:
            sampling_mode = 'B'
        
        # Set up averaged data
        averaged_data_files = self._get_averaged_data_files(sampling_mode)
        self.project_handler.add_averaged_data(averaged_data_files)
        
        # Set up rotated data if needed
        if self.params['do_interacting_fits']:
            rotated_data_files = self._get_rotated_data_files(sampling_mode)
            self.project_handler.add_rotated_data(rotated_data_files)
            self._setup_pivot_info(sampling_mode)
        
        # Set up data handler and channels
        self.data_handler = self.project_handler.data_handler
        self._setup_channels()
    
    def _get_averaged_data_files(self, sampling_mode: str) -> List[str]:
        """Get list of averaged data files."""
        if 'averaged_input_correlators_dir' in self.task_configs:
            if isinstance(self.task_configs['averaged_input_correlators_dir'], list):
                return self.task_configs['averaged_input_correlators_dir']
            else:
                return [self.task_configs['averaged_input_correlators_dir']]
        else:
            return self.proj_files_handler.get_averaged_data(
                self.params['used_averaged_bins'],
                self.project_handler.project_info.bins_info.getRebinFactor(),
                sampling_mode
            )
    
    def _get_rotated_data_files(self, sampling_mode: str) -> List[str]:
        """Get list of rotated data files."""
        if 'rotated_input_correlators_dir' in self.task_configs:
            if isinstance(self.task_configs['rotated_input_correlators_dir'], list):
                return self.task_configs['rotated_input_correlators_dir']
            else:
                return [self.task_configs['rotated_input_correlators_dir']]
        else:
            # Auto-detect rotation parameters
            rotate_type = self.params['pivot_type']
            if self.params['pivot_type'] is not None:
                rotate_type = 'SP' if self.params['pivot_type'] == 0 else 'RP'
            else:
                rotate_type = "*"
            
            rotated_data_files = self.proj_files_handler.get_rotated_data(
                not self.params['use_rotated_samplings'],
                self.project_handler.project_info.bins_info.getRebinFactor(),
                rotate_type, self.params['tN'], self.params['t0'], self.params['tD'],
                sampling_mode, self.params['rotate_run_tag']
            )
            
            if rotated_data_files:
                rotated_data_files.sort(key=os.path.getmtime)
                return [rotated_data_files[-1]]
            else:
                logging.critical("No rotated data files found.")
                return []
    
    def _setup_pivot_info(self, sampling_mode: str) -> None:
        """Set up pivot information from rotated data files."""
        # Extract pivot info from filename if not provided
        if (self.params['pivot_type'] is None or self.params['tN'] is None or 
            self.params['t0'] is None or self.params['tD'] is None):
            
            rotated_files = self.data_handler.rotated_data_files
            if rotated_files:
                pattern = self.proj_files_handler.all_tasks[tm.Task.rotate.name].samplings_file(
                    not self.params['use_rotated_samplings'], None, None,
                    self.project_handler.project_info.bins_info.getRebinFactor(),
                    sampling_mode, "(?P<pivot>\\S+)", "(?P<tN>[0-9]+)", 
                    "(?P<t0>[0-9]+)", "(?P<tD>[0-9]+)", self.params['rotate_run_tag']
                )
                
                match = regex.search(pattern, rotated_files[0])
                if match:
                    match_dict = match.groupdict()
                    self.params['tN'] = int(match_dict['tN'])
                    self.params['t0'] = int(match_dict['t0'])
                    self.params['tD'] = int(match_dict['tD'])
                    self.params['pivot_type'] = 0 if match_dict['pivot'] == 'SP' else 1
                else:
                    logging.critical(f"Could not extract pivot info from '{rotated_files[0]}'")
        
        # Set up pivot file
        if 'pivot_file' not in self.task_configs or self.task_configs['pivot_file'] is None:
            rotate_type = 'SP' if self.params['pivot_type'] == 0 else 'RP'
            self.params['pivot_file'] = self.proj_files_handler.pivot_file(
                rotate_type, self.params['tN'], self.params['t0'], self.params['tD'],
                self.params['rotate_run_tag'],
                self.project_handler.project_info.bins_info.getRebinFactor(),
                sampling_mode
            )
    
    def _setup_channels(self) -> None:
        """Set up and filter channels for fitting."""
        # Get all channels
        self.averaged_channels = self.data_handler.averaged_channels[:]
        self.rotated_channels = self.data_handler.rotated_channels[:] if hasattr(self.data_handler, 'rotated_channels') else []
        
        # Set up single hadron channels
        self.single_hadron_channels = []
        for sh in self.params['single_hadrons']:
            for op in self.params['single_hadrons'][sh]:
                self.single_hadron_channels.append(operator.Operator(op).channel)
        
        # Filter rotated channels
        if self.rotated_channels:
            final_channels = sigmond_util.filter_channels(self.task_configs, self.rotated_channels)
            remove_channels = list(set(self.rotated_channels) - set(final_channels))
            self.project_handler.remove_rotated_data_channels(remove_channels)
            self.rotated_channels = final_channels
        
        # Filter averaged channels
        rm_channels = []
        for channel in self.averaged_channels:
            num_hadrons = self._count_hadrons_in_channel(channel)
            operators = self.data_handler.getAveragedOperators(channel)
            
            # Remove multi-hadron channels that need rotation
            if len(operators) > 1 and num_hadrons > 1:
                rm_channels.append(channel)
            elif num_hadrons > 1:
                # Add single-operator multi-hadron channels to rotated
                if channel not in self.rotated_channels:
                    self.rotated_channels.append(channel)
            elif num_hadrons == 1:
                # Keep only single hadrons that are in our list
                if channel not in self.single_hadron_channels:
                    rm_channels.append(channel)
        
        # Remove unqualified channels
        self.project_handler.remove_averaged_data_channels(rm_channels)
        self.averaged_channels = list(set(self.averaged_channels) - set(rm_channels))
        
        # Final filter for rotated channels
        if self.rotated_channels:
            final_channels = sigmond_util.filter_channels(self.task_configs, self.rotated_channels)
            remove_channels = list(set(self.rotated_channels) - set(final_channels))
            self.project_handler.remove_averaged_data_channels(remove_channels)
            self.rotated_channels = final_channels
        
        # Log final channel sets
        self.task_configs['fitted_channels'] = []
        for channel in self.averaged_channels + self.rotated_channels:
            self.task_configs['fitted_channels'].append(str(channel))
    
    def _count_hadrons_in_channel(self, channel: Any) -> int:
        """Count number of hadrons in a channel."""
        operators = self.data_handler.getAveragedOperators(channel)
        if not operators:
            return 0
        
        op = operators[0]
        if op.operator_info.isGenIrrep():
            opname = op.operator_info.getGenIrrep().getIDName()
            return self._count_hadrons_in_name(opname)
        else:
            return op.operator_info.getBasicLapH().getNumberOfHadrons()
    
    def _count_hadrons_in_name(self, opname: str) -> int:
        """Count hadrons in operator name."""
        hadron_names = ['N', 'X', 'k', 'S', 'L', 'pi', 'P', 'K']
        hadron_tags = ['(', "-", "["]
        
        temp_opname = opname
        count = 0
        for hadron in hadron_names:
            for tag in hadron_tags:
                count += temp_opname.count(hadron + tag)
                temp_opname = temp_opname.replace(hadron + tag, "")
        return count
    
    def run(self) -> None:
        """Execute the fit spectrum task."""
        self._log_task_start("Running spectrum fitting analysis")
        
        # Set up Monte Carlo observables handlers
        self.mcobs_handler, self.mcobs_get_handler = sigmond_util.get_mcobs_handlers(
            self.data_handler, self.project_handler.project_info
        )
        
        # Set correlation mode
        if self.params['correlated']:
            self.mcobs_handler.setToCorrelated()
        else:
            self.mcobs_handler.setToUnCorrelated()
        
        # Initialize result storage
        self.single_hadron_results = {}
        self.single_hadron_info = {}
        self.spectrum_results = {}
        self.tmin_results = {}
        self.tmax_results = {}
        
        # Perform single hadron fits
        self._perform_single_hadron_fits()
        
        # Perform interacting spectrum fits
        if self.params['do_interacting_fits']:
            self._perform_interacting_fits()
        
        # Generate estimates if requested
        if self.params['generate_estimates']:
            self._generate_estimates()
        
        self._log_task_complete("Spectrum fitting analysis")
    
    def _perform_single_hadron_fits(self) -> None:
        """Perform fits for single hadron correlators."""
        logging.info("Fitting single hadron correlators...")
        
        # Initialize single hadron fitter
        fitter = SingleHadronFitter(
            self.mcobs_handler, self.project_handler, self.proj_files_handler, self.params
        )
        
        # Create levels file
        levels_file = self._get_spectrum_levels_file()
        file_created = False
        
        with h5py.File(levels_file, 'w') as final_levels:
            sh_levels = final_levels.create_group('single_hadrons')
            
            for channel in self.single_hadron_channels:
                logging.info(f"Fitting single hadron channel '{str(channel)}'...")
                
                operators = self.data_handler.getChannelOperators(channel)
                if not operators:
                    continue
                
                self.single_hadron_results[channel] = {}
                self.tmin_results[channel] = {}
                self.tmax_results[channel] = {}
                
                for i, op in enumerate(operators):
                    # Determine the operator to fit
                    if len(operators) == 1:
                        fit_op = op
                    else:
                        fit_op = operator.Operator(channel.getRotatedOp(i))
                    
                    # Get single hadron info
                    single_hadron, sh_index = self._get_single_hadron(str(fit_op))
                    if not single_hadron:
                        continue
                    
                    # Set up fit configuration
                    fit_config = self._get_fit_config_for_operator(fit_op, is_interacting=False)
                    
                    # Perform the fit
                    fit_result = fitter.fit_single_hadron(
                        channel, fit_op, fit_config, 
                        file_created, single_hadron, sh_index
                    )
                    
                    # Store results
                    self.single_hadron_results[channel][fit_op] = fit_result
                    self.single_hadron_info[f"{single_hadron}({channel.psq})"] = {
                        'mom': channel.psq,
                        'energy_obs': fit_result.energy_obs,
                        'amp_obs': fit_result.amp_obs,
                        'ecm': fit_result.ecm_estimate,
                        'ecm_ref': fit_result.ecm_estimate
                    }
                    
                    # Write to HDF5 if successful
                    if fit_result.success:
                        hadron_string = f"{single_hadron}({channel.psq})"
                        samplings = self.mcobs_handler.getFullAndSamplingValues(
                            fit_result.energy_obs,
                            self.project_handler.project_info.sampling_info.getSamplingMode()
                        )
                        sh_levels.create_dataset(hadron_string, data=np.array(samplings.array()))
                    
                    file_created = True
        
        logging.info(f"Single hadron fit results written to {levels_file}")
    
    def _perform_interacting_fits(self) -> None:
        """Perform fits for interacting (rotated) correlators."""
        logging.info("Fitting interacting correlators...")
        
        # Initialize interacting fitter
        fitter = InteractingFitter(
            self.mcobs_handler, self.project_handler, self.proj_files_handler, 
            self.params, self.single_hadron_info
        )
        
        file_created = False
        self.interacting_channels = []
        
        for channel in self.rotated_channels:
            operators = self.data_handler.getChannelOperators2(channel)
            if not operators:
                continue
            
            # Check if this is an interacting channel
            if not self._is_interacting_channel(channel, operators):
                continue
            
            self.interacting_channels.append(channel)
            self.spectrum_results[channel] = {}
            
            if channel not in self.tmin_results:
                self.tmin_results[channel] = {}
            if channel not in self.tmax_results:
                self.tmax_results[channel] = {}
            
            logging.info(f"Fitting interacting channel '{str(channel)}'...")
            
            for i, op in enumerate(operators):
                # Determine the operator to fit
                if len(operators) == 1:
                    fit_op = op
                else:
                    fit_op = operator.Operator(channel.getRotatedOp(i))
                
                logging.info(f"\tFitting operator '{str(fit_op)}'...")
                
                # Set up fit configuration
                fit_config = self._get_fit_config_for_operator(fit_op, is_interacting=True)
                
                # Set up non-interacting levels for ratio fits
                ni_level = self._get_non_interacting_level(channel, i)
                
                # Perform the fit
                fit_result = fitter.fit_interacting_correlator(
                    channel, fit_op, fit_config, ni_level, file_created
                )
                
                # Store results
                self.spectrum_results[channel][fit_op] = fit_result
                file_created = True
    
    def _is_interacting_channel(self, channel: Any, operators: List[Any]) -> bool:
        """Check if a channel represents an interacting system."""
        if not operators:
            return False
        
        op = operators[0]
        
        # Count hadrons
        if op.operator_info.isBasicLapH():
            hadrons = op.operator_info.getBasicLapH().getNumberOfHadrons()
        else:
            opname = op.operator_info.getGenIrrep().getIDName()
            if "ROT" in opname:
                hadrons = 2
            else:
                hadrons = self._count_hadrons_in_name(opname)
        
        # Determine operator to check
        if len(operators) == 1:
            check_op = op
        else:
            check_op = operator.Operator(channel.getRotatedOp(0))
        
        # Check if it's a single hadron
        single_hadron, _ = self._get_single_hadron(str(check_op))
        
        # It's interacting if it has multiple hadrons and is not a single hadron
        return (not single_hadron and hadrons >= 2) or (not single_hadron and len(operators) >= 2)
    
    def _get_single_hadron(self, corr_str: str) -> Tuple[Optional[str], Optional[int]]:
        """Get single hadron name and index from correlator string."""
        for hadron in self.params['single_hadrons']:
            if corr_str in self.params['single_hadrons'][hadron]:
                return hadron, self.params['single_hadrons'][hadron].index(corr_str)
        return None, None
    
    def _get_fit_config_for_operator(self, operator: Any, is_interacting: bool) -> FitConfiguration:
        """Get fit configuration for a specific operator."""
        # Start with default configuration
        if is_interacting and self.default_interacting_fit:
            base_config = self.default_interacting_fit
        elif not is_interacting and self.default_noninteracting_fit:
            base_config = self.default_noninteracting_fit
        else:
            base_config = self.default_fit
        
        # Override with operator-specific settings
        op_str = str(operator)
        if op_str in self.params['correlator_fits']:
            op_config = self.params['correlator_fits'][op_str]
            
            # Create new configuration with overrides
            config_dict = base_config.__dict__.copy()
            config_dict.update(op_config)
            
            return FitConfiguration(**config_dict)
        
        return base_config
    
    def _get_non_interacting_level(self, channel: Any, operator_index: int) -> Optional[List[str]]:
        """Get non-interacting level for ratio fits."""
        channel_str = str(channel)
        if channel_str not in self.params['non_interacting_levels']:
            return None
        
        ni_levels = self.params['non_interacting_levels'][channel_str]
        if len(ni_levels) <= operator_index:
            logging.warning(f"Not enough non-interacting levels defined for channel {channel_str}")
            return None
        
        return ni_levels[operator_index]
    
    def _generate_estimates(self) -> None:
        """Generate CSV estimates for fitted spectra."""
        logging.info("Generating spectrum estimates...")
        
        # Generate estimates for single hadrons
        for channel in self.single_hadron_results:
            for operator in self.single_hadron_results[channel]:
                result = self.single_hadron_results[channel][operator]
                if result.success:
                    self._write_estimates_for_result(result, f"sh_{str(operator).replace(' ', '_')}")
        
        # Generate estimates for interacting systems
        for channel in self.spectrum_results:
            for operator in self.spectrum_results[channel]:
                result = self.spectrum_results[channel][operator]
                if result.success:
                    self._write_estimates_for_result(result, f"int_{str(operator).replace(' ', '_')}")
    
    def _write_estimates_for_result(self, result: FitResult, name_prefix: str) -> None:
        """Write estimates for a fit result."""
        if not result.success:
            return
        
        # Create CSV data for the fit result
        fit_data = {
            'energy_value': result.energy_value,
            'energy_error': result.energy_error,
            'chisq_dof': result.chisq_dof,
            'quality': result.quality,
            'dof': result.dof,
            'success': result.success
        }
        
        if result.amplitude_value is not None:
            fit_data['amplitude_value'] = result.amplitude_value
            fit_data['amplitude_error'] = result.amplitude_error
        
        # Add parameter values if available
        if result.parameters:
            for param_name, param_value in result.parameters.items():
                fit_data[f'param_{param_name}'] = param_value
        
        # Write to CSV file
        import pandas as pd
        df = pd.DataFrame([fit_data])
        
        output_file = self.proj_files_handler.fit_estimates_file(name_prefix)
        df.to_csv(output_file, index=False)
        
        logging.info(f"Fit estimates written to {output_file}")
    
    def _get_spectrum_levels_file(self) -> str:
        """Get filename for spectrum levels output."""
        return self.proj_files_handler.spectrum_levels_file()
    
    def plot(self) -> None:
        """Generate plots for the fit spectrum task."""
        if not self._should_create_plots():
            logging.info("No plots requested.")
            return
        
        self._log_task_start("Creating spectrum plots")
        
        # Initialize plotter
        plotter = SpectrumPlotter(self.proj_files_handler, self.params)
        
        # Create spectrum plots
        self._create_spectrum_plots(plotter)
        
        # Create fit quality plots
        self._create_fit_quality_plots(plotter)
        
        # Create operator overlap plots if computed
        if self.params['compute_overlaps']:
            self._create_overlap_plots(plotter)
        
        # Create summary document
        if self.params['create_summary']:
            self._create_summary_document(plotter)
        
        self._log_task_complete("Spectrum plot generation")
    
    def _create_spectrum_plots(self, plotter: SpectrumPlotter) -> None:
        """Create spectrum plots."""
        logging.info("Creating spectrum plots...")
        
        # Plot single hadron spectra
        for channel in self.single_hadron_results:
            for operator in self.single_hadron_results[channel]:
                result = self.single_hadron_results[channel][operator]
                if result.success:
                    plot_name = f"sh_{str(operator).replace(' ', '_')}"
                    plotter.create_spectrum_plot(result, plot_name, "single_hadron")
        
        # Plot interacting spectra
        for channel in self.spectrum_results:
            for operator in self.spectrum_results[channel]:
                result = self.spectrum_results[channel][operator]
                if result.success:
                    plot_name = f"int_{str(operator).replace(' ', '_')}"
                    plotter.create_spectrum_plot(result, plot_name, "interacting")
    
    def _create_fit_quality_plots(self, plotter: SpectrumPlotter) -> None:
        """Create fit quality plots."""
        logging.info("Creating fit quality plots...")
        
        # Create tmin/tmax variation plots if requested
        for channel in self.tmin_results:
            for operator in self.tmin_results[channel]:
                if self.tmin_results[channel][operator]:
                    plot_name = f"tmin_{str(operator).replace(' ', '_')}"
                    plotter.create_fit_quality_plot(
                        self.tmin_results[channel][operator], plot_name, "tmin_variation"
                    )
        
        for channel in self.tmax_results:
            for operator in self.tmax_results[channel]:
                if self.tmax_results[channel][operator]:
                    plot_name = f"tmax_{str(operator).replace(' ', '_')}"
                    plotter.create_fit_quality_plot(
                        self.tmax_results[channel][operator], plot_name, "tmax_variation"
                    )
    
    def _create_overlap_plots(self, plotter: SpectrumPlotter) -> None:
        """Create operator overlap plots."""
        if not self.params['compute_overlaps']:
            return
            
        logging.info("Creating operator overlap plots...")
        
        # Create overlap plots for each interacting channel
        for channel in self.interacting_channels:
            if hasattr(self, 'overlap_results') and channel in self.overlap_results:
                plot_name = f"overlaps_{str(channel).replace(' ', '_')}"
                plotter.create_overlap_plot(
                    self.overlap_results[channel], plot_name, str(channel)
                )
    
    def _create_summary_document(self, plotter: SpectrumPlotter) -> None:
        """Create summary document with all plots."""
        logging.info("Creating summary document...")
        
        plotter.create_summary_document("Spectrum Fitting Results")
        
        # Add single hadron section
        if self.single_hadron_results:
            plotter.add_section_to_summary("Single Hadron Fits", [], 0)
            for channel in self.single_hadron_results:
                for operator in self.single_hadron_results[channel]:
                    result = self.single_hadron_results[channel][operator]
                    if result.success:
                        plot_name = f"sh_{str(operator).replace(' ', '_')}"
                        plot_files = [
                            self.proj_files_handler.spectrum_plot_file(plot_name, "pdf"),
                            self.proj_files_handler.fit_quality_plot_file(plot_name, "pdf")
                        ]
                        plotter.add_section_to_summary(str(operator), plot_files, 0)
        
        # Add interacting fits section
        if self.spectrum_results:
            plotter.add_section_to_summary("Interacting Fits", [], 0)
            for channel in self.spectrum_results:
                for operator in self.spectrum_results[channel]:
                    result = self.spectrum_results[channel][operator]
                    if result.success:
                        plot_name = f"int_{str(operator).replace(' ', '_')}"
                        plot_files = [
                            self.proj_files_handler.spectrum_plot_file(plot_name, "pdf"),
                            self.proj_files_handler.fit_quality_plot_file(plot_name, "pdf")
                        ]
                        if self.params['compute_overlaps']:
                            overlap_plot = self.proj_files_handler.overlap_plot_file(
                                str(channel).replace(' ', '_'), "pdf"
                            )
                            plot_files.append(overlap_plot)
                        
                        plotter.add_section_to_summary(f"{str(channel)} - {str(operator)}", plot_files, 0)
        
        # Finalize summary
        plotter.finalize_summary("spectrum_fitting_summary") 