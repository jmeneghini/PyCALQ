"""
Backward compatibility module for sigmond_util.

This module provides backward compatibility for the original sigmond_util.py
by importing functions from the new modular utility modules. This allows
existing code to continue working while benefiting from the improved
modular structure.
"""

import logging
import warnings

# Import all functions from the new modular utilities
from fvspectrum.utils.project_utils import (
    ProjectInfo, CompactListDumper, check_raw_data_files,
    get_ensemble_info, setup_project, update_params
)

from fvspectrum.utils.data_utils import (
    get_data_handlers, get_mcobs_handlers, estimates_to_csv,
    estimates_to_df, bootstrap_error_by_array, channel_sort,
    update_process_index
)

from fvspectrum.utils.plot_utils import (
    set_latex_in_plots, write_channel_plots, get_selected_mom,
    filter_channels
)

from fvspectrum.utils.fitting_utils import (
    get_pivot_info, get_pivot_type, setup_pivoter, betterchisqrdof,
    correlated_chisquare, minimize_corr_function, get_model_points,
    calculate_prior_sum, get_possible_spectrum_ni_energies,
    calculate_momentum_energy, construct_Z_matrix, calculate_normalized_Z_matrix,
    calculate_certainty_metrics, optimal_per_operator_normalized_assignment
)

# Import remaining functions that might still be in the original file
try:
    from fvspectrum.sigmond_util import (
        sigmond_fit, sigmond_multi_exp_fit, scipy_fit, complete_one_fit,
        minimize_sample
    )
except ImportError:
    # These functions might not exist yet or might be in different modules
    logging.warning("Some fitting functions not available in modular form yet")


def __getattr__(name):
    """
    Provide backward compatibility for any missing functions.
    
    This function is called when an attribute is not found in the module.
    It provides a helpful error message and suggests using the new modular imports.
    """
    warnings.warn(
        f"Function '{name}' has been moved to a modular utility. "
        f"Please import from fvspectrum.utils instead of fvspectrum.sigmond_util. "
        f"See the documentation for the new module structure.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Try to find the function in the utils modules
    from fvspectrum import utils
    if hasattr(utils, name):
        return getattr(utils, name)
    
    raise AttributeError(f"Module 'sigmond_util' has no attribute '{name}'")


# Provide a deprecation warning when this module is imported
warnings.warn(
    "The sigmond_util module has been refactored into modular utilities. "
    "Please import from fvspectrum.utils instead. "
    "This compatibility module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)


# Export all the imported functions for backward compatibility
__all__ = [
    # Project utilities
    'ProjectInfo', 'CompactListDumper', 'check_raw_data_files',
    'get_ensemble_info', 'setup_project', 'update_params',
    
    # Data utilities
    'get_data_handlers', 'get_mcobs_handlers', 'estimates_to_csv',
    'estimates_to_df', 'bootstrap_error_by_array', 'channel_sort',
    'update_process_index',
    
    # Plot utilities
    'set_latex_in_plots', 'write_channel_plots', 'get_selected_mom',
    'filter_channels',
    
    # Fitting utilities
    'get_pivot_info', 'get_pivot_type', 'setup_pivoter', 'betterchisqrdof',
    'correlated_chisquare', 'minimize_corr_function', 'get_model_points',
    'calculate_prior_sum', 'get_possible_spectrum_ni_energies',
    'calculate_momentum_energy', 'construct_Z_matrix', 'calculate_normalized_Z_matrix',
    'calculate_certainty_metrics', 'optimal_per_operator_normalized_assignment',
] 