"""
Project setup and configuration utilities.

This module contains utilities for setting up PyCALQ projects, handling
ensemble information, and managing project configurations.
"""

import logging
import os
import xmltodict
import xml.etree.ElementTree as ET
from typing import NamedTuple, List, Dict, Any
import yaml

import sigmond


class ProjectInfo(NamedTuple):
    """
    Container for all important information for correlator analysis.
    
    This replaces the original ProjectInfo class with better documentation
    and type hints.
    """
    project_dir: str
    raw_data_dirs: List[str]
    ensembles_file: str
    echo_xml: bool
    bins_info: sigmond.MCBinsInfo
    sampling_info: sigmond.MCSamplingInfo
    data_files: Any  # data_files.DataFiles
    precompute: bool
    latex_compiler: str
    subtract_vev: bool


class CompactListDumper(yaml.Dumper):
    """YAML dumper that keeps lists in flow style for more compact output."""
    
    def represent_mapping(self, tag, mapping, flow_style=None):
        """Outer structures remain block style."""
        return super().represent_mapping(tag, mapping, flow_style=False)

    def represent_list(self, data):
        """Nested lists are always in flow style."""
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


def check_raw_data_files(raw_data_files: Any, project_dir: str) -> List[str]:
    """
    Validate that raw data files exist and are not within project directory.
    
    Args:
        raw_data_files: File paths or list of file paths
        project_dir: Project directory path
        
    Returns:
        List of validated file paths
        
    Raises:
        SystemExit: If validation fails
    """
    # Check that raw_data_files are provided
    if not raw_data_files:
        logging.critical("No directory to view. Add 'raw_data_files' to 'view_data' task parameters.")
    
    # Convert to list if single file/directory
    if not isinstance(raw_data_files, list):
        if os.path.isdir(raw_data_files) or os.path.isfile(raw_data_files):
            raw_data_files = [raw_data_files]
        else:
            logging.critical("Parameter 'raw_data_files' must be a real directory.")
    else:
        # Filter out non-existent files
        filtered_files = [f for f in raw_data_files if os.path.isdir(f) or os.path.isfile(f)]
        if filtered_files != raw_data_files:
            logging.critical("Item in 'raw_data_files' must be a real files.")
        raw_data_files = filtered_files

    # Check that raw_data_files are not in project directory
    parent_path = os.path.abspath(project_dir)
    for file in raw_data_files:
        child_path = os.path.abspath(file)
        if parent_path == os.path.commonpath([parent_path, child_path]):
            logging.critical(
                f"Data directory '{child_path}' cannot be a subdirectory "
                f"of project directory '{parent_path}'"
            )

    return raw_data_files


def get_ensemble_info(general_params: Dict[str, Any]) -> sigmond.MCEnsembleInfo:
    """
    Retrieve ensemble info from ensemble file based on ensemble_id.
    
    Args:
        general_params: Dictionary containing ensemble_id and other parameters
        
    Returns:
        MCEnsembleInfo object
        
    Raises:
        SystemExit: If ensemble file not found or ensemble_id not found
    """
    # Check for ensembles file
    ensemble_file_path = os.path.join(
        os.path.dirname(__file__), "..", "sigmond_utils", "ensembles.xml"
    )
    if not os.path.isfile(ensemble_file_path):
        logging.error("Ensembles file cannot be found.")

    # Parse ensemble file
    with open(ensemble_file_path, 'r') as f:
        ensembles = xmltodict.parse(f.read())
    
    # Check if ensemble_id exists
    ids = [item['Id'] for item in ensembles['KnownEnsembles']['Infos']['EnsembleInfo']]
    if general_params['ensemble_id'] not in ids:
        logging.critical(
            f"Ensemble Id not found, check your 'ensemble_id' parameter "
            f"or add your ensemble info to '{ensemble_file_path}'."
        )
    
    general_params["ensembles_file"] = ensemble_file_path
    
    # Get ensemble info
    ensemble_info = sigmond.MCEnsembleInfo(general_params['ensemble_id'], ensemble_file_path)
    return ensemble_info


def setup_project(general_params: Dict[str, Any], raw_data_files: List[str] = None) -> ProjectInfo:
    """
    Set up ProjectInfo class based on general parameters and raw data list.
    
    Args:
        general_params: Dictionary containing project configuration
        raw_data_files: List of raw data file paths
        
    Returns:
        ProjectInfo object with all project configuration
    """
    if raw_data_files is None:
        raw_data_files = []
        
    ensemble_info = get_ensemble_info(general_params)
    ensemble_file_path = general_params["ensembles_file"]
    
    # Set up bins info with optional tweaks
    if 'tweak_ensemble' in general_params:
        bins_info_config = general_params['tweak_ensemble']
        if 'keep_first' in bins_info_config:
            new_bins_info = ET.Element('MCBinsInfo')
            new_bins_info.append(ensemble_info.xml())
            tweaks = ET.SubElement(new_bins_info, 'TweakEnsemble')
            ET.SubElement(tweaks, 'KeepFirst').text = str(bins_info_config['keep_first'])
            ET.SubElement(tweaks, 'KeepLast').text = str(bins_info_config['keep_last'])
            bins_info = sigmond.MCBinsInfo(
                sigmond.XMLHandler().set_from_string(ET.tostring(new_bins_info))
            )
        else:
            bins_info = sigmond.MCBinsInfo(ensemble_info)
            bins_info.setRebin(bins_info_config.get('rebin', 1))
            bins_info.addOmissions(set(bins_info_config.get("omissions", [])))
    else:
        bins_info = sigmond.MCBinsInfo(ensemble_info)
        general_params['tweak_ensemble'] = {
            'rebin': 1,
            'omissions': []
        }

    # Set up sampling info
    if 'sampling_info' in general_params:
        sampling_info_config = general_params['sampling_info']
        try:
            sampling_mode = sigmond.SamplingMode.create(sampling_info_config['mode'].lower())
        except KeyError as err:
            logging.critical(f"Unknown sampling mode {err}")
        
        logging.info(str(sampling_mode).replace(".", ": "))
    else:
        sampling_mode = sigmond.SamplingMode.Jackknife
        general_params['sampling_info'] = {'mode': "Jackknife"}
        
    # Configure sampling based on mode
    if sampling_mode == sigmond.SamplingMode.Bootstrap:
        try:
            sampling_info_config = general_params['sampling_info']
            if 'seed' not in sampling_info_config:
                sampling_info_config['seed'] = 0
            if 'boot_skip' not in sampling_info_config:
                sampling_info_config['boot_skip'] = 0

            sampling_info = sigmond.MCSamplingInfo(
                sampling_info_config['number_resampling'], 
                sampling_info_config['seed'],
                sampling_info_config['boot_skip']
            )
        except KeyError as err:
            logging.critical(f"Missing required key {err}")
    else:
        sampling_info = sigmond.MCSamplingInfo()

    # Set up VEV subtraction
    subtract_vev = general_params.get('subtract_vev', False)

    # Import here to avoid circular imports
    from sigmond_scripts import data_files
    datafiles = data_files.DataFiles()

    return ProjectInfo(
        project_dir=general_params['project_dir'], 
        raw_data_dirs=raw_data_files, 
        ensembles_file=ensemble_file_path,
        echo_xml=False, 
        bins_info=bins_info, 
        sampling_info=sampling_info, 
        data_files=datafiles,
        precompute=True, 
        latex_compiler=None, 
        subtract_vev=subtract_vev
    )


def update_params(other_params: Dict[str, Any], task_params: Dict[str, Any]) -> None:
    """
    Update default parameters with user-specified task parameters.
    
    For each default param in other_params, the user set parameters
    in task_params will override the original value.
    
    Args:
        other_params: Dictionary of default parameters (modified in place)
        task_params: Dictionary of user-specified parameters (modified in place)
    """
    for param in other_params:
        if param in task_params:
            other_params[param] = task_params[param]
        else:
            task_params[param] = other_params[param] 