"""
Average correlators task implementation.

This module provides the refactored implementation of the average correlators task,
which automatically averages Lattice QCD temporal correlator data files within the
same irrep row and total momentum.
"""

import logging
import itertools
from typing import Dict, Any, List, Optional

import sigmond
import fvspectrum.sigmond_util as sigmond_util
from fvspectrum.core.base_task import CorrelatorAnalysisTask
from fvspectrum.analysis.correlator_processor import CorrelatorAverager
from fvspectrum.plotting.spectrum_plotter import SpectrumPlotter


# Documentation string for the task
TASK_DOCUMENTATION = '''
average - a task to read in and automatically average over any Lattice QCD temporal correlator data files 
                given within the same irrep row and total momentum

This task processes raw correlator data files and automatically averages correlators
that belong to the same irreducible representation and momentum configuration.
This reduces statistical noise and simplifies the analysis.

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

average:                  # Required task configuration
  raw_data_files:               # Required: List of input data files
  - /path/to/data/file1.bin
  - /path/to/data/file2.bin
  average_by_bins: false        # Optional: Average by bins vs samplings (default: true)
  average_hadron_irrep_info: true    # Optional: Average hadron irrep info (default: true)
  average_hadron_spatial_info: true  # Optional: Average hadron spatial info (default: true)
  create_pdfs: true             # Optional: Generate PDF plots (default: true)
  create_pickles: true          # Optional: Generate pickle files (default: true)
  create_summary: true          # Optional: Generate summary document (default: true)
  erase_original_matrix_from_memory: true  # Optional: Erase original data (default: true)
  figheight: 6                  # Optional: Figure height (default: 6)
  figwidth: 8                   # Optional: Figure width (default: 8)
  generate_estimates: true      # Optional: Generate CSV estimates (default: true)
  ignore_missing_correlators: true  # Optional: Ignore missing correlators (default: true)
  plot: true                    # Optional: Create plots (default: true)
  separate_mom: false           # Optional: Separate by momentum (default: false)
  tmax: 64                      # Optional: Maximum time for averaging (default: 64)
  tmin: 0                       # Optional: Minimum time for averaging (default: 0)

Output:
-------
- Averaged correlator data files (bins or samplings format)
- CSV files with averaged correlator estimates and effective energies
- PDF/pickle plots of averaged correlators and effective energies
- Summary document with all plots
- Log files with channel and operator averaging information
'''


class AverageCorrelatorsTask(CorrelatorAnalysisTask):
    """
    Task for averaging correlator data.
    
    This task reads raw correlator data files, automatically groups correlators
    by irreducible representation and momentum, performs averaging, and generates
    plots for the averaged data.
    """
    
    @property
    def info(self) -> str:
        """Return task documentation."""
        return TASK_DOCUMENTATION
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for average correlators task."""
        defaults = super()._get_default_parameters()
        defaults.update({
            'average_by_bins': True,
            'average_hadron_irrep_info': True,
            'average_hadron_spatial_info': True,
            'erase_original_matrix_from_memory': True,
            'ignore_missing_correlators': True,
            'separate_mom': False,
            'tmin': 0,
            'tmax': 64,
        })
        return defaults
    
    def _validate_parameters(self) -> None:
        """Validate task-specific parameters."""
        super()._validate_parameters()
        
        # Validate time range
        if self.params['tmin'] < 0:
            logging.warning("tmin < 0, setting to 0")
            self.params['tmin'] = 0
        
        if self.params['tmax'] <= self.params['tmin']:
            logging.critical(f"tmax ({self.params['tmax']}) must be greater than tmin ({self.params['tmin']})")
        
        # Validate plot settings
        if not any([self.params['create_pdfs'], self.params['create_pickles'], self.params['create_summary']]):
            self.params['plot'] = False
            logging.warning("All plot output options disabled. Setting plot=False.")
    
    def run(self) -> None:
        """Execute the average correlators task."""
        self._log_task_start("Running correlator averaging analysis")
        
        # Get Monte Carlo observables handlers
        mcobs_handler, mcobs_get_handler = sigmond_util.get_mcobs_handlers(
            self.data_handler, self.project_handler.project_info
        )
        
        # Analyze channels for averaging
        self.averaged_channels, self.averaged_operators = self._analyze_channels_for_averaging()
        
        # Log averaging information
        self._log_averaging_info()
        
        # Check if we need to do anything
        if not self.params['generate_estimates'] and not self.params['plot']:
            logging.warning("Both 'generate_estimates' and 'plot' are False. Task is obsolete.")
            return
        
        # Perform the averaging
        self._perform_averaging(mcobs_handler)
        
        # Generate estimates if requested
        if self.params['generate_estimates'] or (not self.params['generate_estimates'] and self.params['plot']):
            self._generate_estimates(mcobs_handler)
        
        self._log_task_complete("Correlator averaging analysis")
    
    def _analyze_channels_for_averaging(self) -> tuple:
        """
        Analyze channels to determine averaging groups.
        
        Returns:
            Tuple of (averaged_channels, averaged_operators) dictionaries
        """
        averaged_channels = {}
        averaged_operators = {}
        
        # Group channels by their averaged representation
        for channel in self.channels:
            if channel.is_averaged:
                logging.warning(f"Channel {str(channel)} is averaged already.")
            
            averaged_channel = channel.averaged
            if averaged_channel not in averaged_channels:
                averaged_channels[averaged_channel] = []
                averaged_operators[averaged_channel] = {}
            
            averaged_channels[averaged_channel].append(channel)
        
        # Generate operator averaging maps
        for avchannel, rawchannels in averaged_channels.items():
            for rawchannel in rawchannels:
                operators = [op for op in self.data_handler.getChannelOperators(rawchannel)]
                ops_map = self._get_operators_map(
                    operators, avchannel,
                    self.params['average_hadron_spatial_info'],
                    self.params['average_hadron_irrep_info']
                )
                
                for avop, rawops in ops_map.items():
                    if avop not in averaged_operators[avchannel]:
                        averaged_operators[avchannel][avop] = []
                    
                    if isinstance(rawops, list):
                        averaged_operators[avchannel][avop].extend(rawops)
                    else:
                        averaged_operators[avchannel][avop].append(rawops)
        
        return averaged_channels, averaged_operators
    
    def _log_averaging_info(self) -> None:
        """Log information about channel and operator averaging."""
        import yaml
        import os
        
        # Log channel averaging information
        channel_log = {}
        for avchannel, rawchannels in self.averaged_channels.items():
            channel_log[str(avchannel)] = []
            for channel in rawchannels:
                if channel.is_averaged:
                    channel_log[str(avchannel)].append("WARNING: is averaged already.")
                channel_log[str(avchannel)].append(str(channel))
        
        log_path = os.path.join(self.proj_files_handler.log_dir(), 'channels_combined_log.yml')
        logging.info(f"List of averaged channels written to '{log_path}'.")
        with open(log_path, 'w+') as log_file:
            yaml.dump(channel_log, log_file)
        
        # Log operator averaging information
        operator_log = {}
        for avchannel, operators in self.averaged_operators.items():
            operator_log[str(avchannel)] = {}
            for avop, rawops in operators.items():
                operator_log[str(avchannel)][str(avop)] = [str(op) for op in rawops]
        
        log_path = os.path.join(self.proj_files_handler.log_dir(), 'operators_combined_log.yml')
        logging.info(f"List of averaged operators written to '{log_path}'.")
        with open(log_path, 'w+') as log_file:
            yaml.dump(operator_log, log_file)
    
    def _perform_averaging(self, mcobs_handler) -> None:
        """
        Perform the actual correlator averaging.
        
        Args:
            mcobs_handler: Monte Carlo observables handler
        """
        import tqdm
        
        # Set up file creation tracking
        if self.params['separate_mom']:
            file_created = [False] * 10  # Support up to 10 different momenta
            logging.info(f"Saving averaged correlators to {self._averaged_file(self.params['average_by_bins'], '*')}...")
        else:
            file_created = [False]
            logging.info(f"Saving averaged correlators to {self._averaged_file(self.params['average_by_bins'])}...")
        
        self.moms = []
        
        for avchannel in tqdm.tqdm(self.averaged_operators, desc="Averaging channels"):
            # Determine file index for momentum separation
            index = 0
            mom_key = None
            if self.params['separate_mom']:
                index = avchannel.momentum_squared
                mom_key = index
                self.moms.append(index)
            
            # Determine write mode
            if file_created[index]:
                wmode = sigmond.WriteMode.Update
            else:
                wmode = sigmond.WriteMode.Overwrite
            
            # Set up operators for averaging
            result_ops = [op.operator_info for op in self.averaged_operators[avchannel]]
            input_ops = []
            
            if self.averaged_operators[avchannel]:
                for i in range(len(list(self.averaged_operators[avchannel].values())[0])):
                    an_item = []
                    for op2 in self.averaged_operators[avchannel]:
                        an_item.append((self.averaged_operators[avchannel][op2][i].operator_info, 1.0))
                    input_ops.append(an_item)
            
            # Perform the averaging
            if input_ops:
                if self.params['average_by_bins']:
                    result_obs = sigmond.doCorrelatorMatrixSuperpositionByBins(
                        mcobs_handler, input_ops, result_ops, 
                        self.project_handler.hermitian,
                        self.params['tmin'], self.params['tmax'],
                        self.params['erase_original_matrix_from_memory'],
                        self.params['ignore_missing_correlators']
                    )
                    decoy = sigmond.XMLHandler()
                    mcobs_handler.writeBinsToFile(
                        result_obs, 
                        self._averaged_file(self.params['average_by_bins'], mom_key, repr(avchannel)),
                        decoy, wmode, 'H'
                    )
                else:
                    result_obs = sigmond.doCorrelatorMatrixSuperpositionBySamplings(
                        mcobs_handler, input_ops, result_ops,
                        self.project_handler.hermitian,
                        self.params['tmin'], self.params['tmax'],
                        self.params['erase_original_matrix_from_memory'],
                        self.params['ignore_missing_correlators']
                    )
                    decoy = sigmond.XMLHandler()
                    mcobs_handler.writeSamplingValuesToFile(
                        result_obs,
                        self._averaged_file(self.params['average_by_bins'], mom_key, repr(avchannel)),
                        decoy, wmode, 'H'
                    )
                file_created[index] = True
        
        logging.info(f"Saved averaged correlators to {self._averaged_file(self.params['average_by_bins'])}.")
    
    def _generate_estimates(self, mcobs_handler) -> None:
        """
        Generate correlator and effective energy estimates.
        
        Args:
            mcobs_handler: Monte Carlo observables handler
        """
        save_to_memory = not self.params['generate_estimates'] and self.params['plot']
        if save_to_memory:
            self.processed_data = {}
        
        for avchannel in self.averaged_operators:
            if save_to_memory:
                self.processed_data[avchannel] = {}
            
            for avop1, avop2 in itertools.product(self.averaged_operators[avchannel], 
                                                 self.averaged_operators[avchannel]):
                corr = sigmond.CorrelatorInfo(avop1.operator_info, avop2.operator_info)
                corr_name = repr(corr).replace(" ", "-")
                
                # Compute correlator estimates
                corr_estimates = sigmond.getCorrelatorEstimates(
                    mcobs_handler, corr, self.project_handler.hermitian,
                    self.project_handler.subtract_vev, sigmond.ComplexArg.RealPart,
                    self.project_handler.project_info.sampling_info.getSamplingMode()
                )
                
                # Compute effective energy estimates
                effen_estimates = sigmond.getEffectiveEnergy(
                    mcobs_handler, corr, self.project_handler.hermitian,
                    self.project_handler.subtract_vev, sigmond.ComplexArg.RealPart,
                    self.project_handler.project_info.sampling_info.getSamplingMode(),
                    self.project_handler.time_separation,
                    self.project_handler.effective_energy_type,
                    self.project_handler.vev_const
                )
                
                # Save or store the results
                if save_to_memory:
                    if avop1 not in self.processed_data[avchannel]:
                        self.processed_data[avchannel][avop1] = {}
                    self.processed_data[avchannel][avop1][avop2] = {
                        "corr": sigmond_util.estimates_to_df(corr_estimates),
                        "effen": sigmond_util.estimates_to_df(effen_estimates)
                    }
                else:
                    sigmond_util.estimates_to_csv(
                        corr_estimates, 
                        self.proj_files_handler.corr_estimates_file(corr_name)
                    )
                    sigmond_util.estimates_to_csv(
                        effen_estimates, 
                        self.proj_files_handler.effen_estimates_file(corr_name)
                    )
        
        if self.params['generate_estimates']:
            logging.info(f"Estimates generated in directory {self.proj_files_handler.data_dir('estimates')}.")
    
    def plot(self) -> None:
        """Generate plots for the average correlators task."""
        if not self._should_create_plots():
            logging.info("No plots requested.")
            return
        
        self._log_task_start("Creating plots")
        
        # Initialize plotter
        plotter = SpectrumPlotter(self.proj_files_handler, self.params)
        
        # Get list of averaged channels
        avchannels = list(self.averaged_operators.keys())
        
        # Determine if we should use multiprocessing
        use_multiprocessing = (
            hasattr(self.project_handler, 'nodes') and 
            self.project_handler.nodes and 
            self.project_handler.nodes > 1
        )
        
        # Plot correlators
        if use_multiprocessing:
            self._plot_correlators_multiprocess(plotter, avchannels)
        else:
            self._plot_correlators_sequential(plotter, avchannels)
        
        # Create summary document if requested
        if self.params['create_summary']:
            self._create_summary_document(plotter, avchannels)
        
        self._log_task_complete("Plot generation")
    
    def _plot_correlators_sequential(self, plotter: SpectrumPlotter, avchannels: List[Any]) -> None:
        """Plot correlators sequentially."""
        import tqdm
        
        for channel in tqdm.tqdm(avchannels, desc="Plotting channels"):
            self._plot_channel_correlators(plotter, channel)
    
    def _plot_correlators_multiprocess(self, plotter: SpectrumPlotter, avchannels: List[Any]) -> None:
        """Plot correlators using multiprocessing."""
        from multiprocessing import Process
        
        nodes = self.project_handler.nodes
        chunk_size = int(len(avchannels) / nodes) + 1
        channels_per_node = [avchannels[i:i + chunk_size] for i in range(0, len(avchannels), chunk_size)]
        
        processes = []
        for channel_chunk in channels_per_node:
            if hasattr(self, 'processed_data'):
                process = Process(target=self._plot_channels_with_data, 
                                args=(plotter, channel_chunk))
            else:
                process = Process(target=self._plot_channels_from_files, 
                                args=(plotter, channel_chunk))
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
    
    def _plot_channel_correlators(self, plotter: SpectrumPlotter, channel: Any) -> None:
        """Plot correlators for a single averaged channel."""
        operators = self.averaged_operators[channel]
        
        for avop1, avop2 in itertools.product(operators, operators):
            corr = sigmond.CorrelatorInfo(avop1.operator_info, avop2.operator_info)
            corr_name = repr(corr).replace(" ", "-")
            
            if hasattr(self, 'processed_data') and channel in self.processed_data:
                # Use pre-computed data
                corr_data = self.processed_data[channel][avop1][avop2]["corr"]
                effen_data = self.processed_data[channel][avop1][avop2]["effen"]
            else:
                # Load data from files
                import pandas as pd
                corr_data = pd.read_csv(self.proj_files_handler.corr_estimates_file(corr_name))
                effen_data = pd.read_csv(self.proj_files_handler.effen_estimates_file(corr_name))
            
            # Create plots using the plotter
            plotter._create_correlator_plot(corr_data, corr_name, "correlator")
            plotter._create_correlator_plot(effen_data, corr_name, "effective_energy")
    
    def _create_summary_document(self, plotter: SpectrumPlotter, avchannels: List[Any]) -> None:
        """Create summary document with all plots."""
        if self.params['separate_mom']:
            self._create_momentum_separated_summaries(plotter, avchannels)
        else:
            self._create_single_summary(plotter, avchannels)
    
    def _create_momentum_separated_summaries(self, plotter: SpectrumPlotter, avchannels: List[Any]) -> None:
        """Create separate summary documents for each momentum."""
        self.moms = list(set(self.moms))
        
        for i, momentum in enumerate(self.moms):
            plotter.create_summary_document(f"Average Data - PSQ={momentum}")
            
            # Add sections for channels with this momentum
            momentum_channels = [ch for ch in avchannels if ch.momentum_squared == momentum]
            self._add_channel_sections_to_summary(plotter, momentum_channels, i)
        
        # Finalize all summaries
        for i, momentum in enumerate(self.moms):
            plotter.finalize_summary(f"average_summary_psq{momentum}")
    
    def _create_single_summary(self, plotter: SpectrumPlotter, avchannels: List[Any]) -> None:
        """Create a single summary document for all channels."""
        plotter.create_summary_document("Average Data")
        self._add_channel_sections_to_summary(plotter, avchannels, 0)
        plotter.finalize_summary("average_summary")
    
    def _add_channel_sections_to_summary(self, plotter: SpectrumPlotter, 
                                       channels: List[Any], index: int) -> None:
        """Add channel sections to summary document."""
        for channel in channels:
            # Add section for this channel
            plotter.add_section_to_summary(str(channel), [], index)
            
            # Add subsections for each correlator pair
            operators = self.averaged_operators[channel]
            plot_files = []
            
            for avop1, avop2 in itertools.product(operators, operators):
                corr = sigmond.CorrelatorInfo(avop1.operator_info, avop2.operator_info)
                corr_name = repr(corr).replace(" ", "-")
                
                # Check if plot files exist and add them
                corr_pdf = self.proj_files_handler.corr_plot_file(corr_name, "pdf")
                effen_pdf = self.proj_files_handler.effen_plot_file(corr_name, "pdf")
                
                if self.params['create_pdfs']:
                    plot_files.extend([corr_pdf, effen_pdf])
            
            # Add all plot files for this channel
            plotter.add_section_to_summary(str(channel), plot_files, index)
    
    def _averaged_file(self, binned: bool, mom: Optional[int] = None, 
                      channel: Optional[str] = None) -> str:
        """
        Get the filename for averaged data.
        
        Args:
            binned: Whether data is binned
            mom: Optional momentum value
            channel: Optional channel string
            
        Returns:
            Filename for averaged data
        """
        if self.project_handler.project_info.sampling_info.isJackknifeMode():
            sampling_mode = 'J'
        else:
            sampling_mode = 'B'
        
        return self.proj_files_handler.samplings_file(
            binned, channel, mom,
            self.project_handler.project_info.bins_info.getRebinFactor(),
            sampling_mode
        )
    
    def _get_operators_map(self, operators: List[Any], averaged_channel: Any,
                          get_had_spat: bool = False, get_had_irrep: bool = False) -> Dict[Any, Any]:
        """
        Create a mapping of averaged operators to raw operators.
        
        Args:
            operators: List of raw operators
            averaged_channel: Averaged channel
            get_had_spat: Whether to include hadron spatial info
            get_had_irrep: Whether to include hadron irrep info
            
        Returns:
            Dictionary mapping averaged operators to raw operators
        """
        op_map = {}
        for operator in operators:
            averaged_op = self._get_averaged_operator(
                operator, averaged_channel, get_had_spat, get_had_irrep
            )
            if averaged_op in op_map:
                logging.critical(f"Conflicting operators {operator} and {op_map[averaged_op]}")
            elif averaged_op is None:
                continue
            
            op_map[averaged_op] = operator
        
        return op_map
    
    def _get_averaged_operator(self, operator: Any, averaged_channel: Any,
                              get_had_spat: bool = False, get_had_irrep: bool = False) -> Optional[Any]:
        """
        Get the averaged operator corresponding to a raw operator.
        
        Args:
            operator: Raw operator
            averaged_channel: Averaged channel
            get_had_spat: Whether to include hadron spatial info
            get_had_irrep: Whether to include hadron irrep info
            
        Returns:
            Averaged operator or None if not supported
        """
        from sigmond_scripts import operator as operator_lib
        
        if operator.operator_type is sigmond.OpKind.GenIrrep:
            logging.warning("Averaging of GIOperators not currently supported.")
            return None
        
        op_info = operator.operator_info.getBasicLapH()
        
        # Define name mapping for hadrons
        NAME_MAP = {
            'pion': 'P', 'eta': 'e', 'phi': 'p', 'kaon': 'K', 'kbar': 'k',
            'nucleon': 'N', 'delta': 'D', 'sigma': 'S', 'lambda': 'L',
            'xi': 'X', 'omega': 'O',
        }
        
        if op_info.getNumberOfHadrons() == 1:
            obs_name = f"{NAME_MAP[op_info.getFlavor()]}-{op_info.getHadronSpatialType(1)}_{op_info.getHadronSpatialIdNumber(1)}"
            obs_id = 0
        else:
            obs_name = ""
            for had_num in range(1, op_info.getNumberOfHadrons() + 1):
                had_name = NAME_MAP[op_info.getHadronFlavor(had_num)]
                had_psq = (op_info.getHadronXMomentum(had_num)**2 + 
                          op_info.getHadronYMomentum(had_num)**2 + 
                          op_info.getHadronZMomentum(had_num)**2)
                had_str = str(had_psq)
                
                if get_had_spat:
                    had_spat_type = op_info.getHadronSpatialType(had_num)
                    had_spat_id = op_info.getHadronSpatialIdNumber(had_num)
                    had_str += f"_{had_spat_type}{had_spat_id}"
                
                if get_had_irrep:
                    had_irrep_str = op_info.getHadronLGIrrep(had_num)
                    had_str += f"{had_irrep_str}"
                
                obs_name += f"{had_name}({had_str})"
            
            obs_id = op_info.getLGClebschGordonIdNum()
        
        return operator_lib.Operator(averaged_channel.getGIOperator(obs_name, obs_id)) 