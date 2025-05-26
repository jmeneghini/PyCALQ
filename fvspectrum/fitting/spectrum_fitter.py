"""
Spectrum fitting and analysis.

This module provides classes for fitting correlator data to extract energy
spectra, including single hadron fits, interacting fits, and ratio fits.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

import sigmond
import fvspectrum.sigmond_util as sigmond_util
from fvspectrum.core.data_structures import (
    FitConfiguration, FitResult, SpectrumLevel, MinimizerConfiguration,
    ObservableType, HadronNames
)
from sigmond_scripts import fit_info, sigmond_info, sigmond_input


class SpectrumFitter:
    """
    Handles fitting of correlator data to extract energy spectra.
    
    This class manages the fitting process for both single hadron and
    interacting correlators, including ratio fits and simultaneous fits.
    """
    
    def __init__(self, mcobs_handler, project_handler, ensemble_info):
        """
        Initialize the spectrum fitter.
        
        Args:
            mcobs_handler: Monte Carlo observables handler
            project_handler: Project handler for configuration
            ensemble_info: Ensemble information
        """
        self.mcobs_handler = mcobs_handler
        self.project_handler = project_handler
        self.ensemble_info = ensemble_info
        self.fit_results = {}
        self.single_hadron_results = {}
        self.interacting_results = {}
    
    def fit_spectrum(self, channels: List[Any], operators: Dict[Any, List[Any]], 
                    fit_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit spectrum for all channels and operators.
        
        Args:
            channels: List of physics channels
            operators: Dictionary mapping channels to operators
            fit_configs: Fitting configuration parameters
            
        Returns:
            Dictionary of fit results
        """
        logging.info("Starting spectrum fitting process...")
        
        # Set up fitting configurations
        self._setup_fit_configurations(fit_configs)
        
        # Fit single hadrons first
        if fit_configs.get('single_hadrons'):
            self._fit_single_hadrons(fit_configs['single_hadrons'], fit_configs)
        
        # Fit interacting correlators
        if fit_configs.get('do_interacting_fits', True):
            self._fit_interacting_correlators(channels, operators, fit_configs)
        
        logging.info("Spectrum fitting completed.")
        return self.fit_results
    
    def _setup_fit_configurations(self, fit_configs: Dict[str, Any]) -> None:
        """
        Set up fitting configurations from user input.
        
        Args:
            fit_configs: User-provided fitting configurations
        """
        # Set up default fit configuration
        self.default_fit_config = self._create_fit_configuration(
            fit_configs.get('default_corr_fit', {})
        )
        
        # Set up minimizer configuration
        self.minimizer_config = MinimizerConfiguration(
            **fit_configs.get('minimizer_info', {})
        )
        
        # Set up specific correlator fit configurations
        self.correlator_fit_configs = {}
        for op_name, config in fit_configs.get('correlator_fits', {}).items():
            self.correlator_fit_configs[op_name] = self._create_fit_configuration(config)
    
    def _create_fit_configuration(self, config_dict: Dict[str, Any]) -> FitConfiguration:
        """
        Create a FitConfiguration object from dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters
            
        Returns:
            FitConfiguration object
        """
        return FitConfiguration(
            model=config_dict.get('model', '1-exp'),
            tmin=config_dict.get('tmin', 10),
            tmax=config_dict.get('tmax', 20),
            exclude_times=config_dict.get('exclude_times', []),
            initial_params=config_dict.get('initial_params', {}),
            noise_cutoff=config_dict.get('noise_cutoff', 0.0),
            priors=config_dict.get('priors', {}),
            ratio=config_dict.get('ratio', False),
            sim_fit=config_dict.get('sim_fit', False)
        )
    
    def _fit_single_hadrons(self, single_hadron_config: Dict[str, List[str]], 
                          fit_configs: Dict[str, Any]) -> None:
        """
        Fit single hadron correlators.
        
        Args:
            single_hadron_config: Configuration for single hadrons
            fit_configs: General fitting configuration
        """
        logging.info("Fitting single hadron correlators...")
        
        for hadron_name, operator_list in single_hadron_config.items():
            for operator_name in operator_list:
                self._fit_single_hadron_operator(hadron_name, operator_name, fit_configs)
    
    def _fit_single_hadron_operator(self, hadron_name: str, operator_name: str, 
                                  fit_configs: Dict[str, Any]) -> None:
        """
        Fit a single hadron operator.
        
        Args:
            hadron_name: Name of the hadron
            operator_name: Name of the operator
            fit_configs: Fitting configuration
        """
        # Get fit configuration for this operator
        fit_config = self.correlator_fit_configs.get(operator_name, self.default_fit_config)
        
        # Perform the fit
        fit_result = self._perform_correlator_fit(operator_name, fit_config)
        
        # Store the result
        if hadron_name not in self.single_hadron_results:
            self.single_hadron_results[hadron_name] = {}
        self.single_hadron_results[hadron_name][operator_name] = fit_result
    
    def _fit_interacting_correlators(self, channels: List[Any], 
                                   operators: Dict[Any, List[Any]], 
                                   fit_configs: Dict[str, Any]) -> None:
        """
        Fit interacting correlators.
        
        Args:
            channels: List of physics channels
            operators: Dictionary mapping channels to operators
            fit_configs: Fitting configuration
        """
        logging.info("Fitting interacting correlators...")
        
        for channel in channels:
            if channel not in operators:
                continue
                
            channel_operators = operators[channel]
            self._fit_channel_correlators(channel, channel_operators, fit_configs)
    
    def _fit_channel_correlators(self, channel: Any, operators: List[Any], 
                               fit_configs: Dict[str, Any]) -> None:
        """
        Fit correlators for a single channel.
        
        Args:
            channel: Physics channel
            operators: List of operators in the channel
            fit_configs: Fitting configuration
        """
        if channel not in self.interacting_results:
            self.interacting_results[channel] = {}
        
        for operator in operators:
            operator_name = str(operator)
            
            # Get fit configuration for this operator
            fit_config = self.correlator_fit_configs.get(operator_name, self.default_fit_config)
            
            # Perform the fit
            fit_result = self._perform_correlator_fit(operator, fit_config, channel)
            
            # Store the result
            self.interacting_results[channel][operator] = fit_result
    
    def _perform_correlator_fit(self, operator: Any, fit_config: FitConfiguration, 
                              channel: Any = None) -> FitResult:
        """
        Perform a correlator fit.
        
        Args:
            operator: Operator to fit
            fit_config: Fit configuration
            channel: Optional channel information
            
        Returns:
            FitResult object with fit results
        """
        try:
            # Set up Sigmond input for fitting
            task_input = sigmond_input.SigmondInput(
                self.ensemble_info, self.mcobs_handler, 
                self.project_handler.project_info.bins_info,
                self.project_handler.project_info.sampling_info
            )
            
            # Convert fit configuration to Sigmond format
            fit_input = self._convert_fit_config_to_sigmond(fit_config)
            
            # Perform the fit using appropriate method
            if self.minimizer_config.minimizer == 'lmder':
                fit_info_obj, fit_results, chisqr, qual, dof = sigmond_util.sigmond_fit(
                    task_input, operator, self.minimizer_config.__dict__, fit_input,
                    self.mcobs_handler, 
                    self.project_handler.project_info.sampling_info.getSamplingMode(),
                    self.project_handler.subtract_vev, "fit_log.xml", False
                )
            else:
                # Use scipy fitting
                nsamplings = self.project_handler.project_info.sampling_info.getNumberOfReSamplings(
                    self.project_handler.project_info.bins_info
                )
                fit_info_obj, fit_results, chisqr, qual, dof = sigmond_util.scipy_fit(
                    operator, self.minimizer_config.__dict__, fit_input,
                    self.mcobs_handler, self.project_handler.subtract_vev,
                    self.project_handler.hermitian, self.ensemble_info.getLatticeTimeExtent(),
                    self.project_handler.nodes, nsamplings, False
                )
            
            # Extract energy and amplitude from fit results
            energy_value = fit_results[fit_info_obj.energy_index].getFullEstimate()
            energy_error = fit_results[fit_info_obj.energy_index].getSymmetricError()
            
            amplitude_value = None
            amplitude_error = None
            if hasattr(fit_info_obj, 'amplitude_index') and fit_info_obj.amplitude_index is not None:
                amplitude_value = fit_results[fit_info_obj.amplitude_index].getFullEstimate()
                amplitude_error = fit_results[fit_info_obj.amplitude_index].getSymmetricError()
            
            return FitResult(
                success=True,
                energy_value=energy_value,
                energy_error=energy_error,
                amplitude_value=amplitude_value,
                amplitude_error=amplitude_error,
                chisq_dof=float(chisqr),
                quality=float(qual),
                dof=int(dof)
            )
            
        except Exception as e:
            logging.warning(f"Fit failed for operator {operator}: {e}")
            return FitResult(
                success=False,
                energy_value=0.0,
                energy_error=0.0
            )
    
    def _convert_fit_config_to_sigmond(self, fit_config: FitConfiguration) -> Dict[str, Any]:
        """
        Convert FitConfiguration to Sigmond format.
        
        Args:
            fit_config: FitConfiguration object
            
        Returns:
            Dictionary in Sigmond format
        """
        return {
            'model': fit_config.model,
            'tmin': fit_config.tmin,
            'tmax': fit_config.tmax,
            'exclude_times': fit_config.exclude_times,
            'initial_params': fit_config.initial_params,
            'noise_cutoff': fit_config.noise_cutoff,
            'priors': fit_config.priors,
            'ratio': fit_config.ratio,
            'sim_fit': fit_config.sim_fit
        }
    
    def get_spectrum_levels(self, sort_by_energy: bool = True) -> List[SpectrumLevel]:
        """
        Get all spectrum levels from fit results.
        
        Args:
            sort_by_energy: Whether to sort levels by energy
            
        Returns:
            List of SpectrumLevel objects
        """
        levels = []
        
        # Add single hadron levels
        for hadron_name, hadron_results in self.single_hadron_results.items():
            for operator_name, fit_result in hadron_results.items():
                if fit_result.success:
                    # Create dummy channel and operator info for single hadrons
                    channel_info = None  # Would need to extract from operator
                    operator_info = None  # Would need to extract from operator
                    
                    level = SpectrumLevel(
                        channel=channel_info,
                        operator=operator_info,
                        fit_result=fit_result
                    )
                    levels.append(level)
        
        # Add interacting levels
        for channel, channel_results in self.interacting_results.items():
            for operator, fit_result in channel_results.items():
                if fit_result.success:
                    # Extract channel and operator info
                    channel_info = None  # Would need to extract from channel
                    operator_info = None  # Would need to extract from operator
                    
                    level = SpectrumLevel(
                        channel=channel_info,
                        operator=operator_info,
                        fit_result=fit_result
                    )
                    levels.append(level)
        
        if sort_by_energy:
            levels.sort(key=lambda x: x.sort_key())
        
        return levels
    
    def compute_energy_shifts(self, reference_masses: Dict[str, float]) -> None:
        """
        Compute energy shifts relative to non-interacting levels.
        
        Args:
            reference_masses: Dictionary of reference masses for different hadrons
        """
        logging.info("Computing energy shifts relative to non-interacting levels...")
        
        self.energy_shifts = {}
        self.shift_errors = {}
        
        for channel, channel_results in self.interacting_results.items():
            channel_str = str(channel)
            
            for operator, fit_result in channel_results.items():
                if not fit_result.success:
                    continue
                
                # Get the interacting energy
                interacting_energy = fit_result.energy_value
                interacting_error = fit_result.energy_error
                
                # Compute non-interacting energy from reference masses
                # This is a simplified implementation - in practice, this would
                # involve more complex calculations based on the specific channel
                non_interacting_energy = self._compute_non_interacting_energy(
                    channel, operator, reference_masses
                )
                
                if non_interacting_energy is not None:
                    # Compute energy shift
                    energy_shift = interacting_energy - non_interacting_energy
                    
                    # Store results
                    key = f"{channel_str}_{operator}"
                    self.energy_shifts[key] = energy_shift
                    self.shift_errors[key] = interacting_error  # Simplified error propagation
                    
                    logging.info(f"Energy shift for {key}: {energy_shift:.6f} ± {interacting_error:.6f}")
    
    def _compute_non_interacting_energy(self, channel: Any, operator: Any, 
                                      reference_masses: Dict[str, float]) -> Optional[float]:
        """
        Compute the non-interacting energy for a given channel and operator.
        
        Args:
            channel: Physics channel
            operator: Operator
            reference_masses: Reference masses for hadrons
            
        Returns:
            Non-interacting energy or None if cannot be computed
        """
        # This is a simplified implementation
        # In practice, this would involve:
        # 1. Identifying the hadrons in the channel
        # 2. Getting their momenta
        # 3. Computing the total energy from E = sqrt(m^2 + p^2)
        
        # For now, return a placeholder based on available reference masses
        if reference_masses:
            # Use the first available reference mass as a placeholder
            first_mass = list(reference_masses.values())[0]
            return 2.0 * first_mass  # Placeholder for two-hadron system
        
        return None
    
    def save_fit_results(self, output_dir: str) -> None:
        """
        Save fit results to files.
        
        Args:
            output_dir: Directory to save results
        """
        # Save single hadron results
        single_hadron_file = os.path.join(output_dir, 'single_hadron_fits.csv')
        self._save_single_hadron_results(single_hadron_file)
        
        # Save interacting results
        interacting_file = os.path.join(output_dir, 'interacting_fits.csv')
        self._save_interacting_results(interacting_file)
        
        logging.info(f"Fit results saved to {output_dir}")
    
    def _save_single_hadron_results(self, filename: str) -> None:
        """Save single hadron fit results to CSV."""
        data = []
        for hadron_name, hadron_results in self.single_hadron_results.items():
            for operator_name, fit_result in hadron_results.items():
                data.append({
                    'hadron': hadron_name,
                    'operator': operator_name,
                    'energy_value': fit_result.energy_value,
                    'energy_error': fit_result.energy_error,
                    'chisq_dof': fit_result.chisq_dof,
                    'quality': fit_result.quality,
                    'success': fit_result.success
                })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def _save_interacting_results(self, filename: str) -> None:
        """Save interacting fit results to CSV."""
        data = []
        for channel, channel_results in self.interacting_results.items():
            for operator, fit_result in channel_results.items():
                data.append({
                    'channel': str(channel),
                    'operator': str(operator),
                    'energy_value': fit_result.energy_value,
                    'energy_error': fit_result.energy_error,
                    'chisq_dof': fit_result.chisq_dof,
                    'quality': fit_result.quality,
                    'success': fit_result.success
                })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)


class SingleHadronFitter:
    """
    Specialized fitter for single hadron correlators.
    
    This class handles fitting of single hadron correlators with
    appropriate configurations and result storage.
    """
    
    def __init__(self, mcobs_handler, project_handler, proj_files_handler, params):
        """
        Initialize the single hadron fitter.
        
        Args:
            mcobs_handler: Monte Carlo observables handler
            project_handler: Project handler
            proj_files_handler: Project files handler
            params: Task parameters
        """
        self.mcobs_handler = mcobs_handler
        self.project_handler = project_handler
        self.proj_files_handler = proj_files_handler
        self.params = params
    
    def fit_single_hadron(self, channel, operator, fit_config, file_created, 
                         single_hadron, sh_index):
        """
        Fit a single hadron correlator.
        
        Args:
            channel: Physics channel
            operator: Operator to fit
            fit_config: Fit configuration
            file_created: Whether output file has been created
            single_hadron: Single hadron name
            sh_index: Single hadron index
            
        Returns:
            FitResult object with fit results
        """
        # This is a placeholder implementation
        # The actual implementation would use Sigmond fitting routines
        
        # Create a mock successful fit result
        fit_result = FitResult(
            success=True,
            energy_value=1.0,  # Placeholder energy
            energy_error=0.1,  # Placeholder error
            amplitude_value=1.0,
            amplitude_error=0.1,
            chisq_dof=1.0,
            quality=0.95,
            dof=10
        )
        
        # Add mock observables
        fit_result.energy_obs = None  # Would be actual Sigmond observable
        fit_result.amp_obs = None     # Would be actual Sigmond observable
        fit_result.ecm_estimate = 1.0  # Would be actual estimate
        
        return fit_result


class InteractingFitter:
    """
    Specialized fitter for interacting (multi-hadron) correlators.
    
    This class handles fitting of rotated correlators from GEVP
    analysis, including ratio fits and non-interacting level handling.
    """
    
    def __init__(self, mcobs_handler, project_handler, proj_files_handler, 
                 params, single_hadron_info):
        """
        Initialize the interacting fitter.
        
        Args:
            mcobs_handler: Monte Carlo observables handler
            project_handler: Project handler
            proj_files_handler: Project files handler
            params: Task parameters
            single_hadron_info: Information about single hadron fits
        """
        self.mcobs_handler = mcobs_handler
        self.project_handler = project_handler
        self.proj_files_handler = proj_files_handler
        self.params = params
        self.single_hadron_info = single_hadron_info
    
    def fit_interacting_correlator(self, channel, operator, fit_config, 
                                  ni_level, file_created):
        """
        Fit an interacting correlator.
        
        Args:
            channel: Physics channel
            operator: Operator to fit
            fit_config: Fit configuration
            ni_level: Non-interacting level for ratio fits
            file_created: Whether output file has been created
            
        Returns:
            FitResult object with fit results
        """
        # This is a placeholder implementation
        # The actual implementation would use Sigmond fitting routines
        
        # Create a mock successful fit result
        fit_result = FitResult(
            success=True,
            energy_value=2.0,  # Placeholder energy
            energy_error=0.2,  # Placeholder error
            amplitude_value=1.0,
            amplitude_error=0.1,
            chisq_dof=1.2,
            quality=0.90,
            dof=8
        )
        
        # Add mock observables
        fit_result.energy_obs = None  # Would be actual Sigmond observable
        fit_result.amp_obs = None     # Would be actual Sigmond observable
        fit_result.ecm_estimate = 2.0  # Would be actual estimate
        
        return fit_result


class RatioFitter:
    """
    Specialized fitter for ratio correlators.
    
    This class handles fitting of ratio correlators, which are used to
    reduce systematic uncertainties by dividing out single hadron contributions.
    """
    
    def __init__(self, spectrum_fitter: SpectrumFitter):
        """
        Initialize the ratio fitter.
        
        Args:
            spectrum_fitter: Main spectrum fitter instance
        """
        self.spectrum_fitter = spectrum_fitter
        self.mcobs_handler = spectrum_fitter.mcobs_handler
    
    def fit_ratio_correlators(self, channels: List[Any], 
                            non_interacting_levels: Dict[str, List[Tuple[str, int]]], 
                            fit_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit ratio correlators for specified channels.
        
        Args:
            channels: List of physics channels
            non_interacting_levels: Dictionary mapping channels to non-interacting levels
            fit_configs: Fitting configuration
            
        Returns:
            Dictionary of ratio fit results
        """
        ratio_results = {}
        
        for channel in channels:
            channel_str = str(channel)
            if channel_str in non_interacting_levels:
                ratio_results[channel] = self._fit_channel_ratios(
                    channel, non_interacting_levels[channel_str], fit_configs
                )
        
        return ratio_results
    
    def _fit_channel_ratios(self, channel: Any, ni_levels: List[Tuple[str, int]], 
                          fit_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit ratio correlators for a single channel.
        
        Args:
            channel: Physics channel
            ni_levels: List of (hadron_name, momentum) tuples for non-interacting levels
            fit_configs: Fitting configuration
            
        Returns:
            Dictionary of fit results for this channel
        """
        # Implementation for ratio fitting
        # This involves constructing ratio correlators and fitting them
        return {}


class SimultaneousFitter:
    """
    Handles simultaneous fits of multiple correlators.
    
    This class manages simultaneous fitting of correlators, which can
    provide better constraints and reduced uncertainties.
    """
    
    def __init__(self, spectrum_fitter: SpectrumFitter):
        """
        Initialize the simultaneous fitter.
        
        Args:
            spectrum_fitter: Main spectrum fitter instance
        """
        self.spectrum_fitter = spectrum_fitter
        self.mcobs_handler = spectrum_fitter.mcobs_handler
    
    def fit_simultaneous(self, correlator_groups: List[List[Any]], 
                        fit_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform simultaneous fits of correlator groups.
        
        Args:
            correlator_groups: List of correlator groups to fit simultaneously
            fit_configs: Fitting configuration
            
        Returns:
            Dictionary of simultaneous fit results
        """
        sim_results = {}
        
        for i, group in enumerate(correlator_groups):
            sim_results[f"group_{i}"] = self._fit_correlator_group(group, fit_configs)
        
        return sim_results
    
    def _fit_correlator_group(self, correlators: List[Any], 
                            fit_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit a group of correlators simultaneously.
        
        Args:
            correlators: List of correlators to fit together
            fit_configs: Fitting configuration
            
        Returns:
            Dictionary of fit results for this group
        """
        # Implementation for simultaneous fitting
        # This involves setting up a joint chi-squared and minimizing
        return {} 