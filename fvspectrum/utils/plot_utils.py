"""
Plotting utilities for PyCALQ.

This module contains utilities for setting up matplotlib, managing plot styles,
and handling plot generation.
"""

import logging
import os
import shutil
from typing import List, Any, Dict, Optional
import matplotlib


def set_latex_in_plots(style_import) -> bool:
    """
    Check if latex is available on the system and configure matplotlib.
    
    Args:
        style_import: matplotlib.pyplot or similar module
        
    Returns:
        True if latex is available, False otherwise
    """
    try:
        style_file_path = os.path.join(
            os.path.dirname(__file__), "..", "spectrum_plotting_settings", "spectrum.mplstyle"
        )
        style_import.use(style_file_path)
    except Exception:
        logging.warning(
            'Spectrum style file has been moved or corrupted. '
            'Please consider reinstalling PyCALQ.'
        )
        
    if not shutil.which('latex'):
        matplotlib.rcParams['text.usetex'] = False
        logging.warning(
            "Latex not found on system, please install latex to get "
            "nice looking matplotlib plots."
        )
        return False
    return True


def write_channel_plots(operators: List[Any], plh: Any, create_pickles: bool, 
                       create_pdfs: bool, pdh: Any, data: Optional[Dict] = None) -> None:
    """
    Write channel plots for given operators.
    
    This function creates plots for correlator data associated with the given operators.
    
    Args:
        operators: List of operator objects
        plh: Plot handler object
        create_pickles: Whether to create pickle files
        create_pdfs: Whether to create PDF files
        pdh: Project data handler
        data: Optional pre-computed data dictionary
    """
    import sigmond
    
    # Clear any existing plots
    plh.clf()
    
    # Create plots for each operator pair
    for i, op1 in enumerate(operators):
        for j, op2 in enumerate(operators):
            if j <= i:  # Only plot upper triangle to avoid duplicates
                corr_info = sigmond.CorrelatorInfo(op1.operator_info, op2.operator_info)
                
                try:
                    if data and corr_info in data:
                        # Use pre-computed data
                        corr_data = data[corr_info]
                        plh.plot_correlator_data(corr_data, corr_info)
                    else:
                        # Load data from files
                        plh.plot_correlator(corr_info)
                    
                    # Save plot if requested
                    if create_pdfs:
                        plot_name = f"{op1.operator_info.getIDName()}_{op2.operator_info.getIDName()}"
                        plh.save_plot(plot_name)
                    
                    if create_pickles:
                        pickle_name = f"{op1.operator_info.getIDName()}_{op2.operator_info.getIDName()}.pkl"
                        plh.save_pickle(pickle_name)
                        
                except Exception as e:
                    logging.warning(f"Failed to plot correlator {corr_info}: {e}")
                    continue


def get_selected_mom(task_configs: Dict[str, Any]) -> List[int]:
    """
    Get selected momentum values from task configuration.
    
    Args:
        task_configs: Task configuration dictionary
        
    Returns:
        List of selected momentum squared values
    """
    if 'selected_mom' in task_configs:
        selected_mom = task_configs['selected_mom']
        if isinstance(selected_mom, int):
            return [selected_mom]
        elif isinstance(selected_mom, list):
            return selected_mom
        else:
            logging.warning("selected_mom must be int or list of ints")
            return []
    return []


def filter_channels(task_configs: Dict[str, Any], channel_list: List[Any]) -> List[Any]:
    """
    Filter channels based on task configuration.
    
    Args:
        task_configs: Task configuration dictionary
        channel_list: List of all available channels
        
    Returns:
        Filtered list of channels
    """
    filtered_channels = []
    
    # Filter by momentum if specified
    selected_mom = get_selected_mom(task_configs)
    if selected_mom:
        filtered_channels = [ch for ch in channel_list if ch.psq in selected_mom]
    else:
        filtered_channels = channel_list.copy()
    
    # Filter by channel type if specified
    if 'channel_types' in task_configs:
        channel_types = task_configs['channel_types']
        if not isinstance(channel_types, list):
            channel_types = [channel_types]
        
        type_filtered = []
        for ch in filtered_channels:
            if hasattr(ch, 'channel_type'):
                if ch.channel_type.value in channel_types:
                    type_filtered.append(ch)
            else:
                # Fallback for legacy channels without explicit type
                type_filtered.append(ch)
        filtered_channels = type_filtered
    
    # Filter by irrep if specified
    if 'irreps' in task_configs:
        irreps = task_configs['irreps']
        if not isinstance(irreps, list):
            irreps = [irreps]
        
        irrep_filtered = [ch for ch in filtered_channels if ch.irrep in irreps]
        filtered_channels = irrep_filtered
    
    return filtered_channels 