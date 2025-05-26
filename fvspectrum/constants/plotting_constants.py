"""
Plotting constants for consistent visualization.

This module contains default plotting configurations, color schemes,
and style settings used throughout the PyCALQ visualization components.
"""

from typing import Dict, List, Tuple

# Default figure dimensions
DEFAULT_FIGURE_SIZE = {
    'width': 8,
    'height': 6,
    'dpi': 300,
    'comparison_width': 15,
    'comparison_height': 8,
}

# Color schemes for different plot types
COLOR_SCHEMES = {
    'default': [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ],
    'energy_levels': [
        '#1f77b4',  # Ground state - Blue
        '#ff7f0e',  # First excited - Orange
        '#2ca02c',  # Second excited - Green
        '#d62728',  # Third excited - Red
        '#9467bd',  # Fourth excited - Purple
        '#8c564b',  # Fifth excited - Brown
    ],
    'momentum_squared': {
        0: '#1f77b4',   # PSQ=0 - Blue
        1: '#ff7f0e',   # PSQ=1 - Orange
        2: '#2ca02c',   # PSQ=2 - Green
        3: '#d62728',   # PSQ=3 - Red
        4: '#9467bd',   # PSQ=4 - Purple
        5: '#8c564b',   # PSQ=5 - Brown
    },
    'quality': {
        'excellent': '#2ca02c',  # Green
        'good': '#1f77b4',       # Blue
        'acceptable': '#ff7f0e', # Orange
        'poor': '#d62728',       # Red
        'failed': '#7f7f7f',     # Gray
    },
    'comparison': [
        '#1f77b4',  # Primary - Blue
        '#ff7f0e',  # Secondary - Orange
        '#2ca02c',  # Tertiary - Green
        '#d62728',  # Quaternary - Red
    ]
}

# Plot styles and markers
PLOT_STYLES = {
    'correlators': {
        'linestyle': '-',
        'linewidth': 1.5,
        'marker': 'o',
        'markersize': 4,
        'alpha': 0.8,
        'capsize': 3,
        'capthick': 1,
    },
    'effective_mass': {
        'linestyle': 'None',
        'marker': 's',
        'markersize': 5,
        'alpha': 0.7,
        'capsize': 3,
        'capthick': 1,
    },
    'fits': {
        'linestyle': '--',
        'linewidth': 2,
        'alpha': 0.9,
    },
    'energy_levels': {
        'linestyle': 'None',
        'marker': 'D',
        'markersize': 6,
        'alpha': 0.8,
        'capsize': 4,
        'capthick': 1.5,
    },
    'comparison': {
        'linestyle': '-',
        'linewidth': 2,
        'marker': 'o',
        'markersize': 5,
        'alpha': 0.8,
    }
}

# Font sizes for different plot elements
FONT_SIZES = {
    'title': 16,
    'axis_label': 14,
    'tick_label': 12,
    'legend': 12,
    'annotation': 10,
    'small_text': 8,
}

# Grid and axis settings
GRID_SETTINGS = {
    'show_grid': True,
    'grid_alpha': 0.3,
    'grid_linestyle': ':',
    'grid_linewidth': 0.5,
    'minor_grid': True,
    'minor_grid_alpha': 0.1,
}

# Legend settings
LEGEND_SETTINGS = {
    'location': 'best',
    'frameon': True,
    'fancybox': True,
    'shadow': True,
    'framealpha': 0.9,
    'facecolor': 'white',
    'edgecolor': 'black',
    'ncol': 1,
}

# Axis and tick settings
AXIS_SETTINGS = {
    'tick_direction': 'in',
    'tick_length': 4,
    'minor_tick_length': 2,
    'tick_width': 1,
    'spine_width': 1,
    'label_pad': 8,
}

# Error bar settings
ERROR_BAR_SETTINGS = {
    'capsize': 3,
    'capthick': 1,
    'elinewidth': 1,
    'alpha': 0.8,
    'fmt': 'o',
}

# Plot type specific settings
PLOT_TYPE_SETTINGS = {
    'correlator_plots': {
        'yscale': 'log',
        'xlabel': 'Time',
        'ylabel': 'Correlator',
        'title_template': 'Correlator: {channel}',
    },
    'effective_mass_plots': {
        'yscale': 'linear',
        'xlabel': 'Time',
        'ylabel': 'Effective Mass',
        'title_template': 'Effective Mass: {channel}',
    },
    'energy_spectrum_plots': {
        'yscale': 'linear',
        'xlabel': 'Channel',
        'ylabel': 'Energy',
        'title_template': 'Energy Spectrum',
    },
    'fit_quality_plots': {
        'yscale': 'linear',
        'xlabel': 'Fit Parameter',
        'ylabel': 'Q-value',
        'title_template': 'Fit Quality: {channel}',
    },
    'comparison_plots': {
        'yscale': 'linear',
        'xlabel': 'Analysis',
        'ylabel': 'Energy Difference',
        'title_template': 'Spectrum Comparison',
    }
}

# File output settings
OUTPUT_SETTINGS = {
    'pdf': {
        'format': 'pdf',
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
    },
    'png': {
        'format': 'png',
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'transparent': False,
    },
    'eps': {
        'format': 'eps',
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
    }
}

# LaTeX settings for publication-quality plots
LATEX_SETTINGS = {
    'use_latex': False,  # Set to True for LaTeX rendering
    'font_family': 'serif',
    'font_serif': ['Computer Modern Roman'],
    'font_sans_serif': ['Computer Modern Sans Serif'],
    'font_monospace': ['Computer Modern Typewriter'],
    'mathtext_fontset': 'cm',
    'axes_unicode_minus': False,
}

# Animation settings (for future use)
ANIMATION_SETTINGS = {
    'interval': 200,  # milliseconds between frames
    'repeat': True,
    'repeat_delay': 1000,  # milliseconds
    'blit': True,
}

# Interactive plot settings
INTERACTIVE_SETTINGS = {
    'toolbar': 'toolbar2',
    'navigation_toolbar': True,
    'key_press_handler': True,
    'zoom_on_axis': True,
} 