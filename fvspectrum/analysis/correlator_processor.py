"""
Correlator data processing and analysis.

This module provides classes for processing correlator data, including
estimation calculation, effective energy computation, and data management.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import tqdm
from multiprocessing import Process

import sigmond
import fvspectrum.sigmond_util as sigmond_util
from fvspectrum.core.data_structures import ChannelInfo, OperatorInfo


class CorrelatorProcessor:
    """
    Processes correlator data to extract estimates and effective energies.
    
    This class handles the computation of correlator estimates and effective
    energies from raw lattice QCD data, managing the statistical sampling
    and data output.
    """
    
    def __init__(self, data_handler, project_handler, proj_files_handler):
        """
        Initialize the correlator processor.
        
        Args:
            data_handler: Sigmond data handler
            project_handler: Project handler for configuration
            proj_files_handler: File handler for output management
        """
        self.data_handler = data_handler
        self.project_handler = project_handler
        self.proj_files_handler = proj_files_handler
        self.mcobs_handler = None
        self.mcobs_get_handler = None
        
    def setup_mcobs_handlers(self) -> Tuple[Any, Any]:
        """
        Set up Monte Carlo observables handlers.
        
        Returns:
            Tuple of (mcobs_handler, mcobs_get_handler)
        """
        self.mcobs_handler, self.mcobs_get_handler = sigmond_util.get_mcobs_handlers(
            self.data_handler, self.project_handler.project_info
        )
        return self.mcobs_handler, self.mcobs_get_handler
    
    def process_correlators(self, channels: List[Any], save_estimates: bool = True, 
                          save_to_memory: bool = False) -> Optional[Dict]:
        """
        Process correlators for all channels and operators.
        
        Args:
            channels: List of channels to process
            save_estimates: Whether to save estimates to files
            save_to_memory: Whether to save data to memory for plotting
            
        Returns:
            Dictionary of processed data if save_to_memory is True
        """
        if not self.mcobs_handler:
            self.setup_mcobs_handlers()
        
        data = {} if save_to_memory else None
        
        if save_estimates:
            logging.info(f"Saving correlator estimates to directory {self.proj_files_handler.data_dir()}...")
        
        for channel in channels:
            if save_to_memory:
                data[channel] = {}
            
            channel_operators = self.data_handler.getChannelOperators(channel)
            
            for op1 in channel_operators:
                if save_to_memory:
                    data[channel][op1] = {}
                
                for op2 in channel_operators:
                    self._process_correlator_pair(channel, op1, op2, save_estimates, 
                                                save_to_memory, data)
        
        return data
    
    def _process_correlator_pair(self, channel: Any, op1: Any, op2: Any, 
                               save_estimates: bool, save_to_memory: bool, 
                               data: Optional[Dict]) -> None:
        """
        Process a single correlator pair.
        
        Args:
            channel: Physics channel
            op1: First operator
            op2: Second operator
            save_estimates: Whether to save estimates to files
            save_to_memory: Whether to save to memory
            data: Data dictionary for memory storage
        """
        corr = sigmond.CorrelatorInfo(op1.operator_info, op2.operator_info)
        corr_name = repr(corr).replace(" ", "-")
        
        # Compute correlator estimates
        corr_estimates = self._compute_correlator_estimates(corr)
        
        # Compute effective energy estimates
        effen_estimates = self._compute_effective_energy_estimates(corr)
        
        # Save or store the results
        if save_to_memory:
            data[channel][op1][op2] = {
                "corr": sigmond_util.estimates_to_df(corr_estimates),
                "effen": sigmond_util.estimates_to_df(effen_estimates)
            }
        
        if save_estimates:
            sigmond_util.estimates_to_csv(
                corr_estimates, 
                self.proj_files_handler.corr_estimates_file(corr_name)
            )
            sigmond_util.estimates_to_csv(
                effen_estimates, 
                self.proj_files_handler.effen_estimates_file(corr_name)
            )
    
    def _compute_correlator_estimates(self, corr: Any) -> Any:
        """
        Compute correlator estimates.
        
        Args:
            corr: Correlator info object
            
        Returns:
            Correlator estimates
        """
        return sigmond.getCorrelatorEstimates(
            self.mcobs_handler, corr, 
            self.project_handler.hermitian,
            self.project_handler.subtract_vev,
            sigmond.ComplexArg.RealPart, 
            self.project_handler.project_info.sampling_info.getSamplingMode()
        )
    
    def _compute_effective_energy_estimates(self, corr: Any) -> Any:
        """
        Compute effective energy estimates.
        
        Args:
            corr: Correlator info object
            
        Returns:
            Effective energy estimates
        """
        return sigmond.getEffectiveEnergy(
            self.mcobs_handler, corr,
            self.project_handler.hermitian,
            self.project_handler.subtract_vev,
            sigmond.ComplexArg.RealPart, 
            self.project_handler.project_info.sampling_info.getSamplingMode(),
            self.project_handler.time_separation,
            self.project_handler.effective_energy_type,
            self.project_handler.vev_const
        )
    
    def log_operators_info(self, channels: List[Any]) -> None:
        """
        Log information about operators in each channel.
        
        Args:
            channels: List of channels to log
        """
        log_path = os.path.join(self.proj_files_handler.log_dir(), 'ops_log.yml')
        
        ops_list = {
            "channels": {
                str(channel): {
                    "operators": [str(op) for op in self.data_handler.getChannelOperators(channel)]
                } for channel in channels
            }
        }
        
        logging.info(f"Channels and operators list written to '{log_path}'.")
        with open(log_path, 'w+') as log_file:
            import yaml
            yaml.dump(ops_list, log_file)


class CorrelatorAverager:
    """
    Handles averaging of correlator data across different configurations.
    
    This class manages the averaging process for correlators, including
    averaging by bins or by resampling, and handling different momentum
    and irrep configurations.
    """
    
    def __init__(self, data_handler, project_handler):
        """
        Initialize the correlator averager.
        
        Args:
            data_handler: Sigmond data handler
            project_handler: Project handler for configuration
        """
        self.data_handler = data_handler
        self.project_handler = project_handler
    
    def average_correlators(self, channels: List[Any], average_config: Dict[str, Any]) -> None:
        """
        Average correlators according to configuration.
        
        Args:
            channels: List of channels to average
            average_config: Configuration for averaging process
        """
        logging.info("Starting correlator averaging process...")
        
        for channel in tqdm.tqdm(channels, desc="Averaging channels"):
            self._average_channel_correlators(channel, average_config)
        
        logging.info("Correlator averaging completed.")
    
    def _average_channel_correlators(self, channel: Any, config: Dict[str, Any]) -> None:
        """
        Average correlators for a single channel.
        
        Args:
            channel: Physics channel to average
            config: Averaging configuration
        """
        operators = self.data_handler.getChannelOperators(channel)
        
        if config.get('average_hadron_irrep_info', True):
            operators = self._group_by_irrep(operators)
        
        if config.get('average_hadron_spatial_info', True):
            operators = self._group_by_spatial_config(operators)
        
        # Perform the actual averaging
        self._perform_averaging(channel, operators, config)
    
    def _group_by_irrep(self, operators: List[Any]) -> List[Any]:
        """
        Group operators by irreducible representation.
        
        Args:
            operators: List of operators to group
            
        Returns:
            Grouped operators
        """
        # Implementation for grouping by irrep
        # This would involve analyzing the operator quantum numbers
        return operators
    
    def _group_by_spatial_config(self, operators: List[Any]) -> List[Any]:
        """
        Group operators by spatial configuration.
        
        Args:
            operators: List of operators to group
            
        Returns:
            Grouped operators
        """
        # Implementation for grouping by spatial configuration
        return operators
    
    def _perform_averaging(self, channel: Any, operators: List[Any], 
                         config: Dict[str, Any]) -> None:
        """
        Perform the actual averaging calculation.
        
        Args:
            channel: Physics channel
            operators: List of operators to average
            config: Averaging configuration
        """
        if config.get('average_by_bins', False):
            self._average_by_bins(channel, operators)
        else:
            self._average_by_resampling(channel, operators)
    
    def _average_by_bins(self, channel: Any, operators: List[Any]) -> None:
        """
        Average correlators bin by bin.
        
        Args:
            channel: Physics channel
            operators: List of operators
        """
        logging.info(f"Averaging correlators by bins for channel {channel}")
        
        # Get bin information
        bins_info = self.project_handler.project_info.bins_info
        num_bins = bins_info.getNumberOfBins()
        
        for i, op1 in enumerate(operators):
            for j, op2 in enumerate(operators[i:], i):
                correlator_info = sigmond.CorrelatorInfo(op1.operator_info, op2.operator_info)
                
                # Average across bins
                averaged_data = []
                for bin_idx in range(num_bins):
                    bin_data = self.data_handler.getBinData(correlator_info, bin_idx)
                    if bin_data is not None:
                        averaged_data.append(bin_data)
                
                if averaged_data:
                    # Store averaged result
                    self.data_handler.putAveragedData(correlator_info, averaged_data)
    
    def _average_by_resampling(self, channel: Any, operators: List[Any]) -> None:
        """
        Average correlators using resampling.
        
        Args:
            channel: Physics channel
            operators: List of operators
        """
        logging.info(f"Averaging correlators by resampling for channel {channel}")
        
        # Get sampling information
        sampling_info = self.project_handler.project_info.sampling_info
        num_resamplings = sampling_info.getNumberOfReSamplings(
            self.project_handler.project_info.bins_info
        )
        
        for i, op1 in enumerate(operators):
            for j, op2 in enumerate(operators[i:], i):
                correlator_info = sigmond.CorrelatorInfo(op1.operator_info, op2.operator_info)
                
                # Average across resamplings
                resampled_data = []
                for resample_idx in range(num_resamplings):
                    resample_data = self.data_handler.getResamplingData(correlator_info, resample_idx)
                    if resample_data is not None:
                        resampled_data.append(resample_data)
                
                if resampled_data:
                    # Store averaged result
                    self.data_handler.putAveragedData(correlator_info, resampled_data)


class CorrelatorRotator:
    """
    Handles rotation of correlator matrices using GEVP (Generalized Eigenvalue Problem).
    
    This class manages the rotation process that diagonalizes correlator matrices
    to extract energy eigenvalues and operator overlaps.
    """
    
    def __init__(self, data_handler, project_handler):
        """
        Initialize the correlator rotator.
        
        Args:
            data_handler: Sigmond data handler
            project_handler: Project handler for configuration
        """
        self.data_handler = data_handler
        self.project_handler = project_handler
    
    def rotate_correlators(self, channels: List[Any], rotation_config: Dict[str, Any]) -> None:
        """
        Rotate correlator matrices for all channels.
        
        Args:
            channels: List of channels to rotate
            rotation_config: Configuration for rotation process
        """
        logging.info("Starting correlator rotation process...")
        
        for channel in tqdm.tqdm(channels, desc="Rotating channels"):
            self._rotate_channel_correlators(channel, rotation_config)
        
        logging.info("Correlator rotation completed.")
    
    def _rotate_channel_correlators(self, channel: Any, config: Dict[str, Any]) -> None:
        """
        Rotate correlators for a single channel.
        
        Args:
            channel: Physics channel to rotate
            config: Rotation configuration
        """
        # Set up pivot parameters
        t0 = config['t0']
        tN = config['tN'] 
        tD = config['tD']
        pivot_type = config.get('pivot_type', 0)
        
        # Perform the rotation
        self._perform_rotation(channel, t0, tN, tD, pivot_type, config)
    
    def _perform_rotation(self, channel: Any, t0: int, tN: int, tD: int, 
                        pivot_type: int, config: Dict[str, Any]) -> None:
        """
        Perform the actual rotation calculation.
        
        Args:
            channel: Physics channel
            t0: Metric time for pivot
            tN: Normalize time for pivot
            tD: Diagonalize time for pivot
            pivot_type: Type of pivot (0=single, 1=rolling)
            config: Rotation configuration
        """
        logging.info(f"Performing GEVP rotation for channel {channel}")
        
        # Get operators for this channel
        operators = self.data_handler.getChannelOperators(channel)
        if len(operators) < 2:
            logging.warning(f"Channel {channel} has fewer than 2 operators, skipping rotation")
            return
        
        # Set up pivot information
        if pivot_type == 0:
            pivot_info = sigmond.PivotInfo(sigmond.PivotType.SinglePivot, t0, True)
        else:
            pivot_info = sigmond.PivotInfo(sigmond.PivotType.RollingPivot, t0, True)
        
        # Set up correlator matrix info
        matrix_info = sigmond.CorrelatorMatrixInfo(
            [op.operator_info for op in operators], 
            config.get('hermitian', True), 
            config.get('subtract_vev', False)
        )
        
        # Create rotation task
        rotation_task = sigmond.RotateCorrsTask(
            matrix_info, pivot_info, tN, tD,
            config.get('max_condition_number', 1e12),
            config.get('off_diagonal_rescale', True)
        )
        
        # Perform the rotation
        try:
            rotation_task.initiate(self.data_handler.mcobs_handler)
            
            # Get the rotated operators
            rotated_ops = rotation_task.getRotatedOperators()
            
            # Store rotation results
            for i, rotated_op in enumerate(rotated_ops):
                rotated_channel = sigmond.OperatorInfo(channel.getRotatedOp(i))
                self.data_handler.addRotatedOperator(rotated_channel, rotated_op)
            
            logging.info(f"Successfully rotated {len(rotated_ops)} operators for channel {channel}")
            
        except Exception as e:
            logging.error(f"Rotation failed for channel {channel}: {e}")
            raise 