"""
Constants module for PyCALQ spectrum analysis.

This module centralizes all constants used throughout the PyCALQ codebase,
providing a single source of truth for configuration values, physical constants,
and default parameters.
"""

from .physics_constants import *
from .analysis_constants import *
from .plotting_constants import *

__all__ = [
    # Physics constants
    'HBAR_C', 'LATTICE_SPACING_UNITS', 'REFERENCE_MASSES',
    
    # Analysis constants  
    'DEFAULT_FIT_MODELS', 'DEFAULT_MINIMIZER_CONFIG', 'DEFAULT_SAMPLING_CONFIG',
    'QUALITY_THRESHOLDS', 'TIME_RANGE_LIMITS',
    
    # Plotting constants
    'DEFAULT_FIGURE_SIZE', 'COLOR_SCHEMES', 'PLOT_STYLES', 'FONT_SIZES'
] 