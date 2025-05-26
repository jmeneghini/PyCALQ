"""
Utilities module for PyCALQ spectrum analysis.

This module provides utility functions, helper classes, and common operations
used throughout the PyCALQ codebase. It includes file operations, data validation,
mathematical utilities, project setup, and other support functions.
"""

from .file_utils import *
from .project_utils import *
from .data_utils import *
from .plot_utils import *
from .fitting_utils import *

# Import specific utilities if they exist
try:
    from .validation_utils import *
except ImportError:
    pass

try:
    from .math_utils import *
except ImportError:
    pass

try:
    from .string_utils import *
except ImportError:
    pass

__all__ = [
    # File utilities
    'safe_file_operation', 'create_backup', 'ensure_directory_exists',
    'get_file_hash', 'load_json_file', 'save_json_file',
    'load_pickle_file', 'save_pickle_file', 'compress_file',
    'decompress_file', 'cleanup_temp_files', 'validate_file_permissions',
    'get_file_size', 'copy_file_with_metadata',
    
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