"""
Analysis constants for spectrum fitting and processing.

This module contains default configurations, thresholds, and parameters
used throughout the spectrum analysis pipeline.
"""

from typing import Dict, List, Any

# Default fit models and their typical use cases
DEFAULT_FIT_MODELS = {
    'single_hadron': '1-exp',
    'multi_hadron': '2-exp',
    'scattering': 'multi-exp',
    'effective_mass': 'constant'
}

# Default minimizer configuration
DEFAULT_MINIMIZER_CONFIG = {
    'minimizer': 'lmder',
    'parameter_rel_tol': 1.0e-06,
    'chisquare_rel_tol': 1.0e-04,
    'max_iterations': 2000,
    'verbosity': 'low'
}

# Default sampling configuration
DEFAULT_SAMPLING_CONFIG = {
    'mode': 'Jackknife',
    'number_resampling': 800,  # For Bootstrap
    'seed': 3103,              # For Bootstrap
    'boot_skip': 297           # For Bootstrap
}

# Quality thresholds for fit assessment
QUALITY_THRESHOLDS = {
    'excellent': 0.95,    # Q-value threshold for excellent fits
    'good': 0.50,         # Q-value threshold for good fits
    'acceptable': 0.05,   # Q-value threshold for acceptable fits
    'poor': 0.01,         # Q-value threshold for poor fits
    'max_chisq_dof': 5.0, # Maximum chi-squared per degree of freedom
    'min_dof': 3,         # Minimum degrees of freedom for reliable fits
    'max_relative_error': 0.5,  # Maximum relative error for energy estimates
}

# Time range limits and defaults
TIME_RANGE_LIMITS = {
    'min_tmin': 0,
    'max_tmax': 128,
    'default_tmin': 5,
    'default_tmax': 25,
    'min_fit_range': 3,    # Minimum number of time points for fitting
    'plateau_threshold': 0.1,  # Threshold for effective mass plateau identification
}

# GEVP (Generalized Eigenvalue Problem) defaults
GEVP_DEFAULTS = {
    'default_t0': 5,       # Default metric time
    'default_tN': 5,       # Default normalization time
    'default_tD': 10,      # Default diagonalization time
    'max_condition_number': 50.0,  # Maximum condition number for stability
    'pivot_types': {
        'single': 0,
        'rolling': 1
    },
    'eigenvalue_tolerance': 1.0e-12,  # Tolerance for eigenvalue calculations
}

# Channel filtering and selection
CHANNEL_FILTERS = {
    'momentum_squared_max': 25,    # Maximum momentum squared to consider
    'energy_cutoff_factor': 3.0,   # Factor times reference mass for energy cutoff
    'min_operators_per_channel': 1,  # Minimum operators required per channel
    'max_operators_per_channel': 20, # Maximum operators to consider per channel
}

# Data validation thresholds
DATA_VALIDATION = {
    'min_configurations': 100,     # Minimum gauge configurations
    'max_missing_fraction': 0.1,   # Maximum fraction of missing correlators
    'noise_threshold': 10.0,       # Maximum noise-to-signal ratio
    'correlation_threshold': 0.95,  # Minimum correlation for duplicate detection
    'outlier_sigma': 5.0,          # Sigma threshold for outlier detection
}

# File and directory naming conventions
NAMING_CONVENTIONS = {
    'correlator_extensions': ['.bin', '.dat', '.h5', '.hdf5'],
    'plot_extensions': ['.pdf', '.png', '.eps'],
    'data_extensions': ['.csv', '.txt', '.json'],
    'pickle_extension': '.pkl',
    'summary_prefix': 'summary_',
    'estimates_prefix': 'estimates_',
    'fits_prefix': 'fits_',
}

# Memory management settings
MEMORY_SETTINGS = {
    'max_correlators_in_memory': 1000,  # Maximum correlators to keep in memory
    'chunk_size': 100,                  # Chunk size for batch processing
    'garbage_collection_frequency': 50, # Frequency of garbage collection calls
    'multiprocessing_threshold': 10,    # Minimum channels for multiprocessing
}

# Numerical precision settings
NUMERICAL_PRECISION = {
    'float_precision': 1.0e-15,    # Machine precision threshold
    'energy_precision': 1.0e-10,   # Precision for energy comparisons
    'matrix_condition_limit': 1.0e12,  # Condition number limit for matrix operations
    'integration_tolerance': 1.0e-08,  # Tolerance for numerical integration
}

# Default task parameters
DEFAULT_TASK_PARAMS = {
    'universal': {
        'figheight': 6,
        'figwidth': 8,
        'info': False,
        'plot': True,
        'create_pdfs': True,
        'create_pickles': True,
        'create_summary': True,
        'generate_estimates': True,
    },
    'preview': {
        'separate_mom': True,
    },
    'average': {
        'average_by_bins': False,
        'average_hadron_irrep_info': True,
        'average_hadron_spatial_info': True,
        'erase_original_matrix_from_memory': False,
        'ignore_missing_correlators': True,
        'separate_mom': False,
        'tmax': 64,
        'tmin': 0,
    },
    'rotate': {
        'max_condition_number': 50,
        'pivot_type': 0,
        'precompute': True,
        'rotate_by_samplings': True,
        'used_averaged_bins': True,
    },
    'fit': {
        'compute_overlaps': True,
        'correlated': True,
        'do_interacting_fits': True,
        'non_interacting_energy_sums': False,
        'precompute': True,
        'use_rotated_samplings': True,
        'used_averaged_bins': True,
    },
    'compare': {
        'figheight': 8,
        'figwidth': 15,
        'plot_deltaE': True,
    }
} 