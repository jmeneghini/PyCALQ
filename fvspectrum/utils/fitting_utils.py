"""
Fitting utilities for PyCALQ spectrum analysis.

This module contains utilities for correlator fitting, minimization,
and statistical analysis operations.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import scipy as sp

import sigmond


def get_pivot_info(log_list: List[str]) -> Dict[str, Any]:
    """
    Extract pivot information from log entries.
    
    Args:
        log_list: List of log entries
        
    Returns:
        Dictionary containing pivot information
    """
    pivot_info = {}
    for log_entry in log_list:
        if "pivot" in log_entry.lower():
            # Extract pivot information from log entry
            # This would parse specific pivot-related log messages
            pass
    return pivot_info


def get_pivot_type(pivot_file: str) -> int:
    """
    Determine pivot type from pivot file.
    
    Args:
        pivot_file: Path to pivot file
        
    Returns:
        Pivot type identifier (0 for single, 1 for rolling)
    """
    try:
        with open(pivot_file, 'r') as f:
            content = f.read()
            if "rolling" in content.lower():
                return 1
            else:
                return 0
    except (FileNotFoundError, IOError):
        logging.warning(f"Could not read pivot file {pivot_file}, defaulting to single pivot")
        return 0


def setup_pivoter(pivot_type: int, pivot_file: str, channel: Any, mcobs: Any) -> Any:
    """
    Set up pivoter based on pivot file and type.
    
    Args:
        pivot_type: Type of pivot (0=single, 1=rolling)
        pivot_file: Path to pivot configuration file
        channel: Physics channel
        mcobs: MCObs handler
        
    Returns:
        Configured pivoter object
    """
    import sigmond
    
    if pivot_type == 0:
        # Single pivot
        pivoter = sigmond.HermitianMatrixPivoter()
    else:
        # Rolling pivot
        pivoter = sigmond.RollingPivotOfHermitianMatrix()
    
    # Configure pivoter with file settings
    try:
        pivoter.readFromFile(pivot_file)
    except Exception as e:
        logging.warning(f"Failed to read pivot configuration from {pivot_file}: {e}")
    
    return pivoter


def betterchisqrdof(better_one: Dict[str, Any], worst_one: Dict[str, Any]) -> bool:
    """
    Compare two fits to determine which has better chi-squared per degree of freedom.
    
    Args:
        better_one: First fit result dictionary
        worst_one: Second fit result dictionary
        
    Returns:
        True if better_one is actually better than worst_one
    """
    if 'chisq_dof' not in better_one or 'chisq_dof' not in worst_one:
        return False
    
    return better_one['chisq_dof'] < worst_one['chisq_dof']


def correlated_chisquare(data: np.ndarray, cov_matrix: np.ndarray, 
                        modelpoints: np.ndarray, prior_sum: float = 0.0) -> float:
    """
    Calculate correlated chi-squared value.
    
    Args:
        data: Observed data points
        cov_matrix: Covariance matrix
        modelpoints: Model predictions
        prior_sum: Additional prior contribution
        
    Returns:
        Chi-squared value
    """
    residuals = data - modelpoints
    
    try:
        # Compute chi-squared using inverse covariance matrix
        inv_cov = np.linalg.inv(cov_matrix)
        chisq = np.dot(residuals, np.dot(inv_cov, residuals))
        return chisq + prior_sum
    except np.linalg.LinAlgError:
        logging.warning("Singular covariance matrix, using diagonal approximation")
        # Fallback to diagonal approximation
        diag_cov = np.diag(np.diag(cov_matrix))
        inv_diag_cov = np.linalg.inv(diag_cov)
        chisq = np.dot(residuals, np.dot(inv_diag_cov, residuals))
        return chisq + prior_sum


def minimize_corr_function(parameters: np.ndarray, cov_matrix: np.ndarray, 
                          model: str, trange: List[int], datapoints: np.ndarray,
                          fitop: Any, parameter_map: Dict[str, int], 
                          priors: Dict[str, Any] = None) -> float:
    """
    Objective function for correlator fitting minimization.
    
    Args:
        parameters: Current parameter values
        cov_matrix: Covariance matrix
        model: Fit model type
        trange: Time range for fitting
        datapoints: Observed data points
        fitop: Fit operator
        parameter_map: Mapping of parameter names to indices
        priors: Prior constraints
        
    Returns:
        Chi-squared value to minimize
    """
    if priors is None:
        priors = {}
    
    # Get model points from parameters and model
    try:
        modelpoints = get_model_points(parameters, model, trange, fitop, parameter_map)
        
        # Calculate prior contribution
        prior_sum = calculate_prior_sum(parameters, parameter_map, priors)
        
        # Calculate chi-squared
        chisq = correlated_chisquare(datapoints, cov_matrix, modelpoints, prior_sum)
        
        return chisq
    except Exception as e:
        logging.warning(f"Error in minimize_corr_function: {e}")
        return float('inf')


def get_model_points(parameters: np.ndarray, model: str, trange: List[int],
                    fitop: Any, parameter_map: Dict[str, int]) -> np.ndarray:
    """
    Calculate model points for given parameters.
    
    Args:
        parameters: Parameter values
        model: Model type
        trange: Time range
        fitop: Fit operator
        parameter_map: Parameter mapping
        
    Returns:
        Array of model predictions
    """
    modelpoints = np.zeros(len(trange))
    
    if model == "1-exp":
        # Single exponential: A * exp(-E * t)
        A = parameters[parameter_map.get('amplitude', 0)]
        E = parameters[parameter_map.get('energy', 1)]
        
        for i, t in enumerate(trange):
            modelpoints[i] = A * np.exp(-E * t)
            
    elif model == "2-exp":
        # Double exponential: A1 * exp(-E1 * t) + A2 * exp(-E2 * t)
        A1 = parameters[parameter_map.get('amplitude1', 0)]
        E1 = parameters[parameter_map.get('energy1', 1)]
        A2 = parameters[parameter_map.get('amplitude2', 2)]
        E2 = parameters[parameter_map.get('energy2', 3)]
        
        for i, t in enumerate(trange):
            modelpoints[i] = A1 * np.exp(-E1 * t) + A2 * np.exp(-E2 * t)
            
    elif model == "constant":
        # Constant model
        C = parameters[parameter_map.get('constant', 0)]
        modelpoints.fill(C)
        
    else:
        logging.warning(f"Unknown model type: {model}")
        modelpoints.fill(0.0)
    
    return modelpoints


def calculate_prior_sum(parameters: np.ndarray, parameter_map: Dict[str, int],
                       priors: Dict[str, Any]) -> float:
    """
    Calculate prior contribution to chi-squared.
    
    Args:
        parameters: Parameter values
        parameter_map: Parameter mapping
        priors: Prior constraints
        
    Returns:
        Prior contribution to chi-squared
    """
    prior_sum = 0.0
    
    for param_name, prior_info in priors.items():
        if param_name in parameter_map:
            param_idx = parameter_map[param_name]
            param_value = parameters[param_idx]
            
            if 'mean' in prior_info and 'sigma' in prior_info:
                # Gaussian prior
                mean = prior_info['mean']
                sigma = prior_info['sigma']
                prior_sum += ((param_value - mean) / sigma) ** 2
            elif 'min' in prior_info and 'max' in prior_info:
                # Uniform prior (penalty outside bounds)
                min_val = prior_info['min']
                max_val = prior_info['max']
                if param_value < min_val or param_value > max_val:
                    prior_sum += 1e6  # Large penalty
    
    return prior_sum


def get_possible_spectrum_ni_energies(unique_ni_dict: Dict[str, Any], 
                                    interacting_channels_list: List[Any],
                                    single_hadron_energy_dict: Dict[str, float],
                                    get_sh_operator_func: Any, lattice_extent: int,
                                    ref_ecm_energy: float) -> Dict[str, List[float]]:
    """
    Get possible non-interacting spectrum energies.
    
    Args:
        unique_ni_dict: Dictionary of unique non-interacting levels
        interacting_channels_list: List of interacting channels
        single_hadron_energy_dict: Dictionary of single hadron energies
        get_sh_operator_func: Function to get single hadron operators
        lattice_extent: Lattice spatial extent
        ref_ecm_energy: Reference center-of-mass energy
        
    Returns:
        Dictionary mapping channels to possible energy lists
    """
    possible_energies = {}
    
    for channel in interacting_channels_list:
        channel_energies = []
        
        # Get single hadron combinations for this channel
        if str(channel) in unique_ni_dict:
            ni_levels = unique_ni_dict[str(channel)]
            
            for level in ni_levels:
                # Calculate non-interacting energy
                total_energy = 0.0
                
                for hadron_info in level:
                    hadron_name = hadron_info.get('name', '')
                    momentum = hadron_info.get('momentum', 0)
                    
                    # Get single hadron energy
                    if hadron_name in single_hadron_energy_dict:
                        sh_energy = single_hadron_energy_dict[hadron_name]
                        
                        # Add momentum contribution
                        momentum_energy = calculate_momentum_energy(momentum, lattice_extent)
                        total_energy += np.sqrt(sh_energy**2 + momentum_energy**2)
                
                # Convert to center-of-mass frame if needed
                if ref_ecm_energy > 0:
                    total_energy = total_energy - ref_ecm_energy
                
                channel_energies.append(total_energy)
        
        possible_energies[str(channel)] = sorted(channel_energies)
    
    return possible_energies


def calculate_momentum_energy(momentum: int, lattice_extent: int) -> float:
    """
    Calculate momentum contribution to energy.
    
    Args:
        momentum: Momentum quantum number
        lattice_extent: Lattice spatial extent
        
    Returns:
        Momentum energy contribution
    """
    # Convert momentum to physical units
    momentum_physical = 2.0 * np.pi * momentum / lattice_extent
    return momentum_physical**2 / (2.0)  # Non-relativistic approximation


def construct_Z_matrix(zmags_in_channel: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Construct Z-factor matrix from operator overlaps.
    
    Args:
        zmags_in_channel: Dictionary of Z-factor magnitudes by operator
        
    Returns:
        Z-factor matrix
    """
    operators = list(zmags_in_channel.keys())
    n_ops = len(operators)
    n_levels = len(next(iter(zmags_in_channel.values())))
    
    z_matrix = np.zeros((n_ops, n_levels))
    
    for i, op in enumerate(operators):
        z_matrix[i, :] = zmags_in_channel[op]
    
    return z_matrix


def calculate_normalized_Z_matrix(z_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate normalized Z-factor matrix.
    
    Args:
        z_matrix: Raw Z-factor matrix
        
    Returns:
        Normalized Z-factor matrix
    """
    # Normalize each column (level) to unit norm
    normalized_z = z_matrix.copy()
    
    for level in range(z_matrix.shape[1]):
        column_norm = np.linalg.norm(z_matrix[:, level])
        if column_norm > 0:
            normalized_z[:, level] = z_matrix[:, level] / column_norm
    
    return normalized_z


def calculate_certainty_metrics(normalized_z: np.ndarray, assignments: List[int],
                               z_ops: List[str]) -> Dict[str, float]:
    """
    Calculate certainty metrics for operator-level assignments.
    
    Args:
        normalized_z: Normalized Z-factor matrix
        assignments: List of level assignments for each operator
        z_ops: List of operator names
        
    Returns:
        Dictionary of certainty metrics
    """
    metrics = {}
    
    for i, (op, level) in enumerate(zip(z_ops, assignments)):
        if level < normalized_z.shape[1]:
            # Calculate certainty as the Z-factor magnitude for this assignment
            certainty = abs(normalized_z[i, level])
            metrics[op] = certainty
        else:
            metrics[op] = 0.0
    
    return metrics


def optimal_per_operator_normalized_assignment(zmags_in_channel: Dict[str, np.ndarray],
                                             allowed_single_hadrons: List[str],
                                             get_single_hadrons: Any) -> Dict[str, Any]:
    """
    Find optimal assignment of operators to energy levels.
    
    Args:
        zmags_in_channel: Z-factor magnitudes by operator
        allowed_single_hadrons: List of allowed single hadron types
        get_single_hadrons: Function to get single hadrons from operator
        
    Returns:
        Dictionary containing optimal assignments and metrics
    """
    # Step 1: Construct Z matrix
    z_matrix = construct_Z_matrix(zmags_in_channel)
    
    # Step 2: Normalize Z matrix
    normalized_z = calculate_normalized_Z_matrix(z_matrix)
    
    # Step 3: Find optimal assignments
    operators = list(zmags_in_channel.keys())
    n_ops = len(operators)
    n_levels = normalized_z.shape[1]
    
    # Simple greedy assignment: assign each operator to its strongest level
    assignments = []
    used_levels = set()
    
    # Sort operators by their maximum Z-factor magnitude
    op_max_z = [(i, np.max(np.abs(normalized_z[i, :]))) for i in range(n_ops)]
    op_max_z.sort(key=lambda x: x[1], reverse=True)
    
    for op_idx, _ in op_max_z:
        # Find best available level for this operator
        best_level = -1
        best_z = 0.0
        
        for level in range(n_levels):
            if level not in used_levels:
                z_mag = abs(normalized_z[op_idx, level])
                if z_mag > best_z:
                    best_z = z_mag
                    best_level = level
        
        if best_level >= 0:
            assignments.append(best_level)
            used_levels.add(best_level)
        else:
            assignments.append(-1)  # No assignment possible
    
    # Step 4: Calculate certainty metrics
    certainty_metrics = calculate_certainty_metrics(normalized_z, assignments, operators)
    
    return {
        'assignments': dict(zip(operators, assignments)),
        'certainty_metrics': certainty_metrics,
        'z_matrix': z_matrix,
        'normalized_z': normalized_z
    } 