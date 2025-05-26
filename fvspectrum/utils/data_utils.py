"""
Data handling utilities for PyCALQ.

This module contains utilities for managing data handlers, MCObs handlers,
and data conversion operations.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Tuple, List, Any, Dict
import pandas as pd
import numpy as np

from sigmond_scripts import data_handler
from .project_utils import ProjectInfo


def get_data_handlers(project_info: ProjectInfo) -> Tuple[Any, Any, Any]:
    """
    Get all data handlers for the project.
    
    Args:
        project_info: ProjectInfo object containing project configuration
        
    Returns:
        Tuple of (data_handler, mcobs_handler, mcobs_get_handler)
    """
    this_data_handler = data_handler.DataHandler(project_info)
    mcobs_handler, mcobs_get_handler = get_mcobs_handlers(this_data_handler, project_info)
    return this_data_handler, mcobs_handler, mcobs_get_handler


def get_mcobs_handlers(this_data_handler: Any, project_info: ProjectInfo, 
                      additional_sampling_files: List[str] = None) -> Tuple[Any, Any]:
    """
    Get MCObs handler for managing bin and sample data.
    
    Must return mcobs_get_handler else mcobs_handler does not work.
    
    Args:
        this_data_handler: DataHandler instance
        project_info: ProjectInfo object
        additional_sampling_files: Additional sampling files to include
        
    Returns:
        Tuple of (mcobs_handler, mcobs_get_handler)
    """
    if additional_sampling_files is None:
        additional_sampling_files = []
        
    import sigmond
    
    mcobs_handler_init = ET.Element("MCObservables")
    
    # Add BL correlator data files
    bl_corr_files = (
        list(this_data_handler.raw_data_files.bl_corr_files) +
        list(this_data_handler.averaged_data_files.bl_corr_files) +
        list(this_data_handler.rotated_data_files.bl_corr_files)
    )
    if bl_corr_files:
        bl_corr_xml = ET.SubElement(mcobs_handler_init, "BLCorrelatorData")
        for filename in bl_corr_files:
            flist = filename.xml()
            bl_corr_xml.insert(1, flist)
    
    # Add BL VEV data files
    bl_vev_files = (
        list(this_data_handler.raw_data_files.bl_vev_files) +
        list(this_data_handler.averaged_data_files.bl_vev_files)
    )
    if bl_vev_files:
        bl_vev_files_xml = ET.SubElement(mcobs_handler_init, "BLVEVData")
        # Add file list info here if needed
    
    # Add bin data files
    bin_files = (
        list(this_data_handler.raw_data_files.bin_files) +
        list(this_data_handler.averaged_data_files.bin_files) +
        list(this_data_handler.rotated_data_files.bin_files)
    )
    if bin_files:
        bin_files_xml = ET.SubElement(mcobs_handler_init, "BinData")
        for filename in bin_files:
            flist = filename.xml()
            bin_files_xml.insert(1, flist)
    
    # Add sampling files
    sampling_files = list(this_data_handler.sampling_files) + additional_sampling_files
    if sampling_files:
        sampling_files_xml = ET.SubElement(mcobs_handler_init, "SamplingData")
        for filename in sampling_files:
            flist = filename.xml()
            sampling_files_xml.insert(1, flist)
    
    # Create MCObs handlers
    mcobs_xml_handler = sigmond.XMLHandler()
    mcobs_xml_handler.set_from_string(ET.tostring(mcobs_handler_init))
    mcobs_get_handler = sigmond.MCObsGetHandler(
        mcobs_xml_handler, project_info.bins_info, project_info.sampling_info
    )
    mcobs_handler = sigmond.MCObsHandler(mcobs_get_handler, False)
    
    return mcobs_handler, mcobs_get_handler


def estimates_to_csv(estimates: Dict[str, Any], data_file: str) -> None:
    """
    Write estimates to CSV file.
    
    Args:
        estimates: Dictionary of estimates to write
        data_file: Output CSV file path
    """
    df = estimates_to_df(estimates)
    df.to_csv(data_file, index=False)


def estimates_to_df(estimates: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert estimates dictionary to pandas DataFrame.
    
    Args:
        estimates: Dictionary of estimates
        
    Returns:
        DataFrame with estimates data
    """
    data = []
    for key, estimate in estimates.items():
        if hasattr(estimate, 'getFullEstimate'):
            # Sigmond estimate object
            full_est = estimate.getFullEstimate()
            data.append({
                'observable': str(key),
                'value': full_est.mean,
                'error': full_est.stddev
            })
        else:
            # Dictionary format
            data.append({
                'observable': str(key),
                'value': estimate.get('value', 0.0),
                'error': estimate.get('error', 0.0)
            })
    
    return pd.DataFrame(data)


def bootstrap_error_by_array(array: np.ndarray) -> float:
    """
    Calculate bootstrap error from array of values.
    
    Args:
        array: Array of bootstrap samples
        
    Returns:
        Bootstrap error estimate
    """
    return np.std(array, ddof=1)


def channel_sort(item: Any) -> str:
    """
    Sort key function for channels.
    
    Args:
        item: Channel object to sort
        
    Returns:
        String representation for sorting
    """
    return str(item)


def update_process_index(ip: int, nnodes: int) -> int:
    """
    Update process index for multiprocessing.
    
    Args:
        ip: Current process index
        nnodes: Total number of nodes
        
    Returns:
        Updated process index
    """
    return (ip + 1) % nnodes 