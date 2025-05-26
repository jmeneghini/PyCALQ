"""
File utility functions for PyCALQ.

This module provides utility functions for file and directory operations,
including path validation, file management, and safe I/O operations.
"""

import os
import shutil
import glob
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import json
import pickle


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
        
    Raises:
        OSError: If directory cannot be created
    """
    path = Path(directory_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        logging.error(f"Failed to create directory {path}: {e}")
        raise


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (including the dot)
    """
    return Path(file_path).suffix.lower()


def find_files_by_pattern(directory: Union[str, Path], pattern: str, 
                         recursive: bool = False) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return []
    
    if recursive:
        pattern = f"**/{pattern}"
        files = list(directory.glob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    return sorted(files)


def safe_file_write(file_path: Union[str, Path], content: str, 
                   backup: bool = True) -> bool:
    """
    Safely write content to a file with optional backup.
    
    Args:
        file_path: Path to the file
        content: Content to write
        backup: Whether to create a backup if file exists
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    try:
        # Create backup if requested and file exists
        if backup and file_path.exists():
            backup_file(file_path)
        
        # Ensure parent directory exists
        ensure_directory_exists(file_path.parent)
        
        # Write content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logging.debug(f"Successfully wrote file: {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to write file {file_path}: {e}")
        return False


def backup_file(file_path: Union[str, Path], backup_suffix: str = ".bak") -> Optional[Path]:
    """
    Create a backup copy of a file.
    
    Args:
        file_path: Path to the file to backup
        backup_suffix: Suffix for the backup file
        
    Returns:
        Path to the backup file if successful, None otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logging.warning(f"Cannot backup non-existent file: {file_path}")
        return None
    
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
    
    try:
        shutil.copy2(file_path, backup_path)
        logging.debug(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logging.error(f"Failed to create backup of {file_path}: {e}")
        return None


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0.0
    
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def safe_json_write(file_path: Union[str, Path], data: Dict[str, Any], 
                   indent: int = 2) -> bool:
    """
    Safely write data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to write
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        return safe_file_write(file_path, content)
    except Exception as e:
        logging.error(f"Failed to write JSON file {file_path}: {e}")
        return False


def safe_json_read(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Safely read data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data if successful, None otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logging.warning(f"JSON file does not exist: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to read JSON file {file_path}: {e}")
        return None


def safe_pickle_write(file_path: Union[str, Path], data: Any) -> bool:
    """
    Safely write data to a pickle file.
    
    Args:
        file_path: Path to the pickle file
        data: Data to pickle
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    try:
        ensure_directory_exists(file_path.parent)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.debug(f"Successfully wrote pickle file: {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to write pickle file {file_path}: {e}")
        return False


def safe_pickle_read(file_path: Union[str, Path]) -> Any:
    """
    Safely read data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded data if successful, None otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logging.warning(f"Pickle file does not exist: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to read pickle file {file_path}: {e}")
        return None


def clean_directory(directory: Union[str, Path], pattern: str = "*", 
                   keep_subdirs: bool = True) -> int:
    """
    Clean files from a directory matching a pattern.
    
    Args:
        directory: Directory to clean
        pattern: Pattern of files to remove
        keep_subdirs: Whether to keep subdirectories
        
    Returns:
        Number of files removed
    """
    directory = Path(directory)
    
    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return 0
    
    files_removed = 0
    
    try:
        for item in directory.glob(pattern):
            if item.is_file():
                item.unlink()
                files_removed += 1
                logging.debug(f"Removed file: {item}")
            elif item.is_dir() and not keep_subdirs:
                shutil.rmtree(item)
                files_removed += 1
                logging.debug(f"Removed directory: {item}")
        
        logging.info(f"Cleaned {files_removed} items from {directory}")
        return files_removed
        
    except Exception as e:
        logging.error(f"Failed to clean directory {directory}: {e}")
        return files_removed


def get_directory_size(directory: Union[str, Path]) -> float:
    """
    Get total size of a directory in megabytes.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in MB
    """
    directory = Path(directory)
    
    if not directory.exists():
        return 0.0
    
    total_size = 0
    
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        
        return total_size / (1024 * 1024)
        
    except Exception as e:
        logging.error(f"Failed to calculate directory size for {directory}: {e}")
        return 0.0


def copy_file_with_metadata(source: Union[str, Path], 
                          destination: Union[str, Path]) -> bool:
    """
    Copy a file preserving metadata.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        logging.error(f"Source file does not exist: {source}")
        return False
    
    try:
        ensure_directory_exists(destination.parent)
        shutil.copy2(source, destination)
        logging.debug(f"Copied {source} to {destination}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to copy {source} to {destination}: {e}")
        return False


def find_latest_file(directory: Union[str, Path], pattern: str = "*") -> Optional[Path]:
    """
    Find the most recently modified file matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        Path to the latest file, or None if no files found
    """
    files = find_files_by_pattern(directory, pattern)
    
    if not files:
        return None
    
    # Sort by modification time, most recent first
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] 