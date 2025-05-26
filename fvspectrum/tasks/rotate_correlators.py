"""
Rotate correlators task implementation.

This module provides the refactored implementation of the rotate correlators task,
which pivots correlator matrices using GEVP (Generalized Eigenvalue Problem) to
return time-dependent eigenvalues.
"""

import logging
import os
import math
import itertools
from typing import Dict, Any, List, Optional
import h5py
import tqdm
from multiprocessing import Process

import sigmond
import fvspectrum.sigmond_util as sigmond_util
from fvspectrum.core.base_task import CorrelatorAnalysisTask
from fvspectrum.analysis.correlator_processor import CorrelatorRotator
from fvspectrum.plotting.spectrum_plotter import SpectrumPlotter
from sigmond_scripts import sigmond_info, sigmond_input, operator


# Documentation string for the task
TASK_DOCUMENTATION = '''
rotate - a task to pivot a given correlator matrix and return the time-dependent eigenvalues

This task performs GEVP (Generalized Eigenvalue Problem) rotation on correlator matrices
to extract energy eigenvalues and operator overlaps. The rotation diagonalizes the
correlator matrix at a specific time to isolate individual energy states.

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

rotate:                   # Required task configuration
  t0: 5                         # Required: Metric time for pivot
  tD: 10                        # Required: Diagonalize time for pivot
  tN: 5                         # Required: Normalize time for pivot
  averaged_input_correlators_dir: {project_dir}/1average/data/bins  # Optional: Input directory
  create_pdfs: true             # Optional: Generate PDF plots (default: true)
  create_pickles: true          # Optional: Generate pickle files (default: true)
  create_summary: true          # Optional: Generate summary document (default: true)
  figheight: 6                  # Optional: Figure height (default: 6)
  figwidth: 8                   # Optional: Figure width (default: 8)
  generate_estimates: true      # Optional: Generate CSV estimates (default: true)
  max_condition_number: 50      # Optional: Maximum condition number (default: 150)
  omit_operators: []            # Optional: Operators to omit (default: [])
  pivot_type: 0                 # Optional: 0=single pivot, 1=rolling pivot (default: 0)
  plot: true                    # Optional: Create plots (default: true)
  precompute: true              # Optional: Precompute correlators (default: true)
  rotate_by_samplings: true     # Optional: Rotate by samplings vs bins (default: true)
  run_tag: "unique"             # Optional: Unique run identifier (default: "")
  tmax: 25                      # Optional: Maximum time for analysis (default: 25)
  tmin: 2                       # Optional: Minimum time for analysis (default: 2)
  used_averaged_bins: true      # Optional: Use averaged bins vs samplings (default: true)

Output:
-------
- Rotated correlator data files (HDF5 format)
- Pivot matrix files with rotation information
- CSV files with rotated correlator estimates and effective energies
- PDF/pickle plots of rotated correlators and effective energies
- Summary document with all plots and pivot information
- XML log files with rotation details
'''


class RotateCorrelatorsTask(CorrelatorAnalysisTask):
    """
    Task for rotating correlator matrices using GEVP.
    
    This task performs the Generalized Eigenvalue Problem (GEVP) rotation
    on correlator matrices to extract energy eigenvalues and operator overlaps.
    """
    
    @property
    def info(self) -> str:
        """Return task documentation."""
        return TASK_DOCUMENTATION
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for rotate correlators task."""
        defaults = super()._get_default_parameters()
        defaults.update({
            'pivot_type': 0,  # 0 = single pivot, 1 = rolling pivot
            'precompute': True,
            'max_condition_number': 150,
            'rotate_by_samplings': True,
            'used_averaged_bins': True,
            'omit_operators': [],
            'run_tag': "",
            'tmin': 2,
            'tmax': 25,
        })
        return defaults
    
    def _validate_parameters(self) -> None:
        """Validate task-specific parameters."""
        super()._validate_parameters()
        
        # Check required rotation parameters
        for param in ['t0', 'tN', 'tD']:
            if param not in self.task_configs:
                logging.critical(f"No default '{param}' set for '{self.task_name}'. Ending task.")
        
        self.t0 = self.task_configs['t0']
        self.tN = self.task_configs['tN']
        self.tD = self.task_configs['tD']
        
        # Validate pivot type
        if self.params['pivot_type'] not in [0, 1]:
            logging.critical("Parameter 'pivot_type' must be 0 (single pivot) or 1 (rolling pivot).")
        
        # Validate time range
        if self.params['tmin'] is not None and self.params['tmax'] is not None:
            if self.params['tmax'] <= self.params['tmin']:
                logging.critical(f"tmax ({self.params['tmax']}) must be greater than tmin ({self.params['tmin']})")
        
        # Validate plot settings
        if not any([self.params['create_pdfs'], self.params['create_pickles'], self.params['create_summary']]):
            self.params['plot'] = False
            logging.warning("All plot output options disabled. Setting plot=False.")
    
    def _setup_data_handlers(self) -> None:
        """Set up data handlers for rotation analysis."""
        # Get averaged data files
        averaged_data_files = self._get_averaged_data_files()
        self.project_handler.add_averaged_data(averaged_data_files)
        
        # Set up data handler and channels
        self.data_handler = self.project_handler.data_handler
        self.channels = self.data_handler.averaged_channels[:]
        
        # Filter channels
        final_channels = sigmond_util.filter_channels(self.task_configs, self.channels)
        remove_channels = list(set(self.channels) - set(final_channels))
        
        # Remove channels with insufficient operators
        unqual_channels = []
        for channel in final_channels:
            if len(self.data_handler.getAveragedOperators(channel)) < 2:
                unqual_channels.append(channel)
                logging.info(f"Skipping {str(channel)} because there is an insufficient number of operators.")
        
        remove_channels += unqual_channels
        final_channels = list(set(final_channels) - set(unqual_channels))
        
        self.project_handler.remove_averaged_data_channels(remove_channels)
        self.channels = final_channels
        self.channels.sort(key=sigmond_util.channel_sort)
        
        # Log rotated channels
        self.task_configs['rotated_channels'] = [str(channel) for channel in self.channels]
    
    def _get_averaged_data_files(self) -> List[str]:
        """Get list of averaged data files."""
        if 'averaged_input_correlators_dir' in self.task_configs:
            if isinstance(self.task_configs['averaged_input_correlators_dir'], list):
                return self.task_configs['averaged_input_correlators_dir']
            else:
                return [self.task_configs['averaged_input_correlators_dir']]
        else:
            # Use default location
            if self.project_handler.project_info.sampling_info.isJackknifeMode():
                sampling_mode = 'J'
            else:
                sampling_mode = 'B'
            
            return self.proj_files_handler.get_averaged_data(
                self.params['used_averaged_bins'],
                self.project_handler.project_info.bins_info.getRebinFactor(),
                sampling_mode,
                sigmond_util.get_selected_mom(self.task_configs)
            )
    
    def run(self) -> None:
        """Execute the rotate correlators task."""
        self._log_task_start("Running correlator rotation analysis")
        
        # Set up rotation parameters
        pivot_string = 'single_pivot' if self.params['pivot_type'] == 0 else 'rolling_pivot'
        
        # Set up multiprocessing
        nodes = max(1, self.project_handler.nodes - 1) if self.project_handler.nodes else 1
        chunked_channels = self._chunk_channels_for_processing(nodes)
        
        # Set up rotation tasks
        task_inputs, skip_channels = self._setup_rotation_tasks(chunked_channels, pivot_string)
        
        # Remove channels with insufficient operators
        for channel in skip_channels:
            if channel in self.channels:
                self.channels.remove(channel)
        
        # Execute rotation tasks
        self._execute_rotation_tasks(task_inputs, nodes)
        
        # Combine output files
        self._combine_output_files(len(task_inputs))
        
        # Generate estimates if requested
        if self._should_generate_estimates():
            self._generate_rotation_estimates()
        
        self._log_task_complete("Correlator rotation analysis")
    
    def _chunk_channels_for_processing(self, nodes: int) -> List[List[Any]]:
        """Chunk channels for multiprocessing."""
        if nodes == 1:
            return [self.channels]
        
        initial_channel = self.channels[0]
        final_channels = self.channels[1:]
        
        if len(final_channels) == 0:
            return [[initial_channel]]
        
        chunk_size = math.ceil(len(final_channels) / nodes)
        chunk_size = min(chunk_size, len(final_channels))
        chunked_channels = [final_channels[i:i + chunk_size] 
                           for i in range(0, len(final_channels), chunk_size)]
        chunked_channels.insert(0, [initial_channel])
        
        return chunked_channels
    
    def _setup_rotation_tasks(self, chunked_channels: List[List[Any]], 
                            pivot_string: str) -> tuple:
        """Set up rotation tasks for Sigmond."""
        task_inputs = []
        skip_channels = []
        self.nlevels = {}
        
        logging.info("Setting up rotation inputs...")
        
        for il, channels in enumerate(tqdm.tqdm(chunked_channels, desc="Setting up tasks")):
            # Create Sigmond input for this chunk
            task_input = sigmond_input.SigmondInput(
                os.path.basename(self.project_handler.project_info.project_dir),
                self.project_handler.project_info.bins_info,
                self.project_handler.project_info.sampling_info,
                self.project_handler.project_info.ensembles_file,
                self.data_handler.averaged_data_files,
                "temp1.xml",
                self._sigmond_rotation_log(il),
                self.params['precompute'],
                None,
            )
            
            file_created = False
            for channel in channels:
                # Set time range if not specified
                if self.params['tmax'] is None and self.params['tmin'] is None:
                    self.params['tmin'], self.params['tmax'] = self.data_handler.getChannelsLargestTRange(channel)
                elif self.params['tmax'] is None:
                    _, self.params['tmax'] = self.data_handler.getChannelsLargestTRange(channel)
                elif self.params['tmin'] is None:
                    self.params['tmin'], _ = self.data_handler.getChannelsLargestTRange(channel)
                
                # Set up operators
                initial_operators = self.data_handler.getChannelOperators(channel)
                initial_operator_strings = [str(op) for op in initial_operators]
                
                # Remove omitted operators
                for op in self.params['omit_operators']:
                    if op in initial_operator_strings:
                        idx = initial_operator_strings.index(op)
                        initial_operators.pop(idx)
                        initial_operator_strings.pop(idx)
                
                operators = [op.operator_info for op in initial_operators]
                self.nlevels[channel] = len(operators)
                
                if len(operators) < 2:
                    logging.info(f"Skipping {str(channel)} because there is an insufficient number of operators.")
                    skip_channels.append(channel)
                    continue
                
                # Set up rotation mode and parameters
                rotate_mode = "samplings_all" if self.params['rotate_by_samplings'] else "bins"
                wmode = sigmond.WriteMode.Update if file_created else sigmond.WriteMode.Overwrite
                
                # Configure rotation task
                task_input.doCorrMatrixRotation(
                    sigmond_info.PivotInfo(
                        pivot_string, norm_time=self.tN, metric_time=self.t0,
                        diagonalize_time=self.tD, 
                        max_condition_number=self.params['max_condition_number']
                    ),
                    sigmond_info.RotateMode(rotate_mode),
                    sigmond.CorrelatorMatrixInfo(
                        operators, self.project_handler.hermitian, 
                        self.project_handler.subtract_vev
                    ),
                    operator.Operator(channel.getRotatedOp()),
                    self.params['tmin'],
                    self.params['tmax'],
                    rotated_corrs_filename=self._rotated_corrs_file(
                        not self.params['rotate_by_samplings'], repr(channel), il
                    ),
                    file_mode=wmode,
                    pivot_filename=self._pivot_file(repr(channel), il),
                    pivot_overwrite=not file_created,
                )
                file_created = True
            
            task_inputs.append(task_input)
        
        return task_inputs, skip_channels
    
    def _execute_rotation_tasks(self, task_inputs: List[Any], nodes: int) -> None:
        """Execute the rotation tasks."""
        logging.info("Starting the rotation tasks...")
        
        processes = []
        write_method = 'w+'
        taskhandlers = []
        
        for task_input in tqdm.tqdm(task_inputs, desc="Executing rotations"):
            # Finalize Sigmond task inputs
            setuptaskhandler = sigmond.XMLHandler()
            setuptaskhandler.set_from_string(task_input.to_str())
            
            # Write input file
            with open(os.path.join(self.proj_files_handler.log_dir(), 'sigmond_rotation_input.xml'), write_method) as f:
                f.write(setuptaskhandler.output(1))
            
            # Execute rotation
            taskhandlers.append(sigmond.TaskHandler(setuptaskhandler))
            if nodes > 1:
                processes.append(Process(target=taskhandlers[-1].do_batch_tasks, args=(setuptaskhandler,)))
                processes[-1].start()
            else:
                taskhandlers[-1].do_batch_tasks(setuptaskhandler)
            
            if len(processes) == 1:
                processes[0].join()  # First process must complete first
                write_method = 'a'
        
        # Wait for all processes to complete
        for process in tqdm.tqdm(processes[1:], desc="Collecting results"):
            process.join()
        
        del taskhandlers
        logging.info("Rotation tasks completed.")
    
    def _combine_output_files(self, num_tasks: int) -> None:
        """Combine output files from multiple tasks."""
        logging.info("Combining output files...")
        
        # Combine rotated correlator files
        with h5py.File(self._rotated_corrs_file(not self.params['rotate_by_samplings']), 'w') as datafile:
            with h5py.File(self._pivot_file(), 'w') as pivotfile:
                for i in range(num_tasks):
                    # Combine data files
                    data_file_i = self._rotated_corrs_file(not self.params['rotate_by_samplings'], None, i)
                    if os.path.exists(data_file_i):
                        with h5py.File(data_file_i, 'r') as datafilei:
                            if i == 0:
                                datafilei.copy(datafilei["Info"], datafile["/"], "Info")
                            for channel in self.channels:
                                if repr(channel) in datafilei.keys():
                                    datafilei.copy(datafilei[repr(channel)], datafile["/"], repr(channel))
                        os.remove(data_file_i)
                    
                    # Combine pivot files
                    pivot_file_i = self._pivot_file(None, i)
                    if os.path.exists(pivot_file_i):
                        with h5py.File(pivot_file_i, 'r') as pivotfilei:
                            if i == 0:
                                pivotfilei.copy(pivotfilei["Info"], pivotfile["/"], "Info")
                            for channel in self.channels:
                                if repr(channel) in pivotfilei.keys():
                                    pivotfilei.copy(pivotfilei[repr(channel)], pivotfile["/"], repr(channel))
                        os.remove(pivot_file_i)
        
        # Log output file locations
        self._log_output_files()
        
        # Add header information to files
        self._add_header_information()
    
    def _log_output_files(self) -> None:
        """Log the locations of output files."""
        logging.info(f"Sigmond input file written to {os.path.join(self.proj_files_handler.log_dir(), 'sigmond_rotation_input.xml')}")
        
        if os.path.isfile(self._pivot_file()):
            logging.info(f"Pivot matrix written to {self._pivot_file()}.")
        
        if os.path.isfile(self._rotated_corrs_file(not self.params['rotate_by_samplings'])):
            logging.info(f"Rotated correlators written to {self._rotated_corrs_file(not self.params['rotate_by_samplings'])}.")
        else:
            self.params['plot'] = False
        
        logging.info(f"Log file(s) written to {self._sigmond_rotation_log('*')}")
    
    def _add_header_information(self) -> None:
        """Add header information to output files."""
        rotated_file = self._rotated_corrs_file(not self.params['rotate_by_samplings'])
        pivot_file = self._pivot_file()
        
        if os.path.isfile(rotated_file) and os.path.isfile(pivot_file):
            with h5py.File(rotated_file, 'r+') as datafile:
                with h5py.File(pivot_file, 'r+') as pivotfile:
                    if self.channels:
                        dataheader = datafile[repr(self.channels[0])]['Header'][()]
                        pivotheader = pivotfile[repr(self.channels[0])]['Header'][()]
                        
                        datainfo = datafile['Info']
                        pivotinfo = pivotfile['Info']
                        
                        datainfo.create_dataset('Header', data=dataheader + pivotheader)
                        pivotinfo.create_dataset('Header', data=dataheader + pivotheader)
    
    def _should_generate_estimates(self) -> bool:
        """Determine if estimates should be generated."""
        return (self.params['plot'] or self.params['generate_estimates']) and \
               os.path.isfile(self._rotated_corrs_file(not self.params['rotate_by_samplings']))
    
    def _generate_rotation_estimates(self) -> None:
        """Generate estimates for rotated correlators."""
        # Add rotated data to project handler
        self.project_handler.add_rotated_data([self._rotated_corrs_file(not self.params['rotate_by_samplings'])])
        mcobs_handler, mcobs_get_handler = sigmond_util.get_mcobs_handlers(
            self.data_handler, self.project_handler.project_info
        )
        
        # Set up data storage for plotting
        save_to_memory = self.params['plot'] and not self.params['generate_estimates']
        if save_to_memory:
            self.rotated_estimates = {}
        
        logging.info(f"Generating estimates for {self.proj_files_handler.data_dir('estimates')}...")
        
        for channel in self.channels:
            if save_to_memory:
                self.rotated_estimates[str(channel)] = {}
            
            logging.info(f"\tGenerating estimates for channel {str(channel)}...")
            
            for i in range(self.nlevels[channel]):
                for j in range(self.nlevels[channel]):
                    rop1 = operator.Operator(channel.getRotatedOp(i))
                    rop2 = operator.Operator(channel.getRotatedOp(j))
                    corr = sigmond.CorrelatorInfo(rop1.operator_info, rop2.operator_info)
                    corr_name = repr(corr).replace(" ", "-")
                    
                    if save_to_memory:
                        self.rotated_estimates[str(channel)][corr] = {}
                    
                    # Generate correlator estimates
                    corr_estimates = sigmond.getCorrelatorEstimates(
                        mcobs_handler, corr, self.project_handler.hermitian,
                        self.project_handler.subtract_vev, sigmond.ComplexArg.RealPart,
                        self.project_handler.project_info.sampling_info.getSamplingMode()
                    )
                    
                    # Generate effective energy estimates
                    effen_estimates = sigmond.getEffectiveEnergy(
                        mcobs_handler, corr, self.project_handler.hermitian,
                        self.project_handler.subtract_vev, sigmond.ComplexArg.RealPart,
                        self.project_handler.project_info.sampling_info.getSamplingMode(),
                        self.project_handler.time_separation,
                        self.project_handler.effective_energy_type,
                        self.project_handler.vev_const
                    )
                    
                    # Save or store estimates
                    if self.params['generate_estimates']:
                        if corr_estimates:
                            sigmond_util.estimates_to_csv(
                                corr_estimates, 
                                self.proj_files_handler.corr_estimates_file(corr_name)
                            )
                        else:
                            # Remove existing file if no data
                            corr_file = self.proj_files_handler.corr_estimates_file(corr_name)
                            if os.path.exists(corr_file):
                                os.remove(corr_file)
                        
                        if effen_estimates:
                            sigmond_util.estimates_to_csv(
                                effen_estimates, 
                                self.proj_files_handler.effen_estimates_file(corr_name)
                            )
                        else:
                            # Remove existing file if no data
                            effen_file = self.proj_files_handler.effen_estimates_file(corr_name)
                            if os.path.exists(effen_file):
                                os.remove(effen_file)
                    
                    elif save_to_memory:
                        self.rotated_estimates[str(channel)][corr]["corr"] = sigmond_util.estimates_to_df(corr_estimates)
                        self.rotated_estimates[str(channel)][corr]["effen"] = sigmond_util.estimates_to_df(effen_estimates)
                    
                    if not corr_estimates:
                        logging.warning(f"No data found for {repr(corr)}.")
    
    def plot(self) -> None:
        """Generate plots for the rotate correlators task."""
        if not self._should_create_plots():
            logging.info("No plots requested.")
            return
        
        self._log_task_start("Creating plots")
        
        # Initialize plotter
        plotter = SpectrumPlotter(self.proj_files_handler, self.params)
        
        # Generate plots
        if self.project_handler.nodes and self.project_handler.nodes > 1:
            self._plot_correlators_multiprocess(plotter)
        else:
            self._plot_correlators_sequential(plotter)
        
        # Create summary document if requested
        if self.params['create_summary']:
            self._create_summary_document(plotter)
        
        self._log_task_complete("Plot generation")
    
    def _plot_correlators_sequential(self, plotter: SpectrumPlotter) -> None:
        """Plot correlators sequentially."""
        for channel in self.channels:
            self._plot_channel_correlators(plotter, channel)
    
    def _plot_correlators_multiprocess(self, plotter: SpectrumPlotter) -> None:
        """Plot correlators using multiprocessing."""
        chunk_size = math.ceil(len(self.channels) / self.project_handler.nodes)
        chunk_size = min(chunk_size, len(self.channels))
        chunked_channels = [self.channels[i:i + chunk_size] 
                           for i in range(0, len(self.channels), chunk_size)]
        
        processes = []
        for channels in chunked_channels:
            process = Process(target=self._plot_channels_chunk, args=(plotter, channels))
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
    
    def _plot_channels_chunk(self, plotter: SpectrumPlotter, channels: List[Any]) -> None:
        """Plot a chunk of channels."""
        for channel in channels:
            self._plot_channel_correlators(plotter, channel)
    
    def _plot_channel_correlators(self, plotter: SpectrumPlotter, channel: Any) -> None:
        """Plot correlators for a single channel."""
        import pandas as pd
        
        logging.info(f"\tGenerating plots for channel {str(channel)}...")
        
        # Define correlation order (diagonal first, then off-diagonal)
        corr_order = [(i, i) for i in range(self.nlevels[channel])]
        for i in range(self.nlevels[channel]):
            for j in range(self.nlevels[channel]):
                if j != i:
                    corr_order.append((i, j))
        
        for i, j in corr_order:
            rop1 = operator.Operator(channel.getRotatedOp(i))
            rop2 = operator.Operator(channel.getRotatedOp(j))
            corr = sigmond.CorrelatorInfo(rop1.operator_info, rop2.operator_info)
            corr_name = repr(corr).replace(" ", "-")
            
            # Clean up existing plot files
            plot_files = [
                self.proj_files_handler.corr_plot_file(corr_name, "pickle"),
                self.proj_files_handler.corr_plot_file(corr_name, "pdf"),
                self.proj_files_handler.effen_plot_file(corr_name, "pickle"),
                self.proj_files_handler.effen_plot_file(corr_name, "pdf")
            ]
            for file in plot_files:
                if os.path.exists(file):
                    os.remove(file)
            
            # Get correlator data
            if hasattr(self, 'rotated_estimates'):
                corr_data = self.rotated_estimates[str(channel)][corr]["corr"]
                if corr_data.empty:
                    continue
            else:
                corr_file = self.proj_files_handler.corr_estimates_file(corr_name)
                if not os.path.exists(corr_file):
                    continue
                corr_data = pd.read_csv(corr_file)
            
            # Create correlator plot
            plotter._create_correlator_plot(corr_data, corr_name, "correlator")
            
            # Get effective energy data and create plot
            try:
                if hasattr(self, 'rotated_estimates'):
                    effen_data = self.rotated_estimates[str(channel)][corr]["effen"]
                else:
                    effen_file = self.proj_files_handler.effen_estimates_file(corr_name)
                    effen_data = pd.read_csv(effen_file)
                
                plotter._create_correlator_plot(effen_data, corr_name, "effective_energy")
            except (FileNotFoundError, KeyError):
                pass  # Skip if effective energy data not available
    
    def _create_summary_document(self, plotter: SpectrumPlotter) -> None:
        """Create summary document with all plots and pivot information."""
        plotter.create_summary_document("Rotated Correlators")
        
        # Get pivot information
        pivot_info = sigmond_util.get_pivot_info([
            self._sigmond_rotation_log(il) for il in range(self.project_handler.nodes)
        ])
        
        logging.info("\tGenerating summary document...")
        
        for channel in self.channels:
            plotter.add_section_to_summary(str(channel), [], 0)
            
            # Add pivot information table if available
            if channel in pivot_info:
                # This would require extending the plotter to handle tables
                # For now, we'll skip the table functionality
                pass
            
            # Add correlator plots in order
            corr_order = [(i, i) for i in range(self.nlevels[channel])]
            for i in range(self.nlevels[channel]):
                for j in range(self.nlevels[channel]):
                    if j != i:
                        corr_order.append((i, j))
            
            plot_files = []
            for i, j in corr_order:
                rop1 = operator.Operator(channel.getRotatedOp(i))
                rop2 = operator.Operator(channel.getRotatedOp(j))
                corr = sigmond.CorrelatorInfo(rop1.operator_info, rop2.operator_info)
                corr_name = repr(corr).replace(" ", "-")
                
                if self.params['create_pdfs']:
                    corr_pdf = self.proj_files_handler.corr_plot_file(corr_name, "pdf")
                    effen_pdf = self.proj_files_handler.effen_plot_file(corr_name, "pdf")
                    plot_files.extend([corr_pdf, effen_pdf])
            
            plotter.add_section_to_summary(str(channel), plot_files, 0)
        
        # Finalize summary
        plotter.finalize_summary("rotation_summary")
    
    def _rotated_corrs_file(self, binned: bool, channel: Optional[str] = None, 
                           index: Optional[int] = None) -> str:
        """Get filename for rotated correlators data."""
        run_tag = self.params['run_tag']
        if index is not None:
            run_tag += f"-{index}"
        
        if self.project_handler.project_info.sampling_info.isJackknifeMode():
            sampling_mode = 'J'
        else:
            sampling_mode = 'B'
        
        rotate_type = 'SP' if self.params['pivot_type'] == 0 else 'RP'
        
        return self.proj_files_handler.samplings_file(
            binned, channel, None,
            self.project_handler.project_info.bins_info.getRebinFactor(),
            sampling_mode, rotate_type, self.tN, self.t0, self.tD, run_tag
        )
    
    def _pivot_file(self, channel: Optional[str] = None, 
                   index: Optional[int] = None) -> str:
        """Get filename for pivot data."""
        run_tag = self.params['run_tag']
        if index is not None:
            run_tag += f"-{index}"
        
        if self.project_handler.project_info.sampling_info.isJackknifeMode():
            sampling_mode = 'J'
        else:
            sampling_mode = 'B'
        
        rotate_type = 'SP' if self.params['pivot_type'] == 0 else 'RP'
        
        return self.proj_files_handler.pivot_file(
            rotate_type, self.tN, self.t0, self.tD, run_tag,
            self.project_handler.project_info.bins_info.getRebinFactor(),
            sampling_mode, channel
        )
    
    def _sigmond_rotation_log(self, il: int) -> str:
        """Get filename for Sigmond rotation log."""
        return os.path.join(self.proj_files_handler.log_dir(), f"sigmond_rotation_log-{il}.xml") 