"""
Core data structures and enums for spectrum analysis.

This module defines the fundamental data structures, enums, and utility classes
used throughout the PyCALQ spectrum analysis pipeline. These structures provide
type safety, validation, and consistent interfaces across all components.
"""

from enum import Enum
from typing import NamedTuple, Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import re


class ObservableType(Enum):
    """Enumeration of observable types in spectrum analysis."""
    ENERGY_SHIFT_LAB = "dElab"          # Energy shift in lab frame
    ENERGY_LAB = "elab"                 # Energy in lab frame  
    ENERGY_CM = "ecm"                   # Energy in center of momentum frame
    ENERGY_CM_NORMALIZED = "ecm_ref"    # Energy in CM frame normalized by reference mass
    AMPLITUDE = "amp"                   # Correlator amplitude
    ENERGY_SHIFT_NORMALIZED = "dElab_ref"  # Energy shift normalized by reference mass


class FitModel(Enum):
    """Enumeration of correlator fitting models."""
    SINGLE_EXPONENTIAL = "1-exp"
    DOUBLE_EXPONENTIAL = "2-exp"
    MULTI_EXPONENTIAL = "multi-exp"
    CONSTANT = "constant"
    COSH = "cosh"
    SINH = "sinh"


class SamplingMode(Enum):
    """Enumeration of statistical resampling modes."""
    JACKKNIFE = "Jackknife"
    BOOTSTRAP = "Bootstrap"


class PivotType(Enum):
    """Enumeration of GEVP pivot types."""
    SINGLE_PIVOT = 0
    ROLLING_PIVOT = 1


class ChannelType(Enum):
    """Enumeration of physics channel types."""
    SINGLE_HADRON = "single_hadron"
    MULTI_HADRON = "multi_hadron"
    SCATTERING = "scattering"


@dataclass(frozen=True)
class FitConfiguration:
    """
    Immutable configuration for a correlator fit.
    
    This class encapsulates all parameters needed to perform a correlator fit,
    including the model, time range, and various fitting options.
    """
    model: str
    tmin: int
    tmax: int
    exclude_times: List[int] = field(default_factory=list)
    initial_params: Dict[str, float] = field(default_factory=dict)
    noise_cutoff: float = 0.0
    priors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ratio: bool = False
    tmin_plots: List[Dict[str, Any]] = field(default_factory=list)
    tmax_plots: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tmin >= self.tmax:
            raise ValueError(f"tmin ({self.tmin}) must be less than tmax ({self.tmax})")
        
        if self.noise_cutoff < 0:
            raise ValueError(f"noise_cutoff must be non-negative, got {self.noise_cutoff}")
        
        # Validate exclude_times are within range
        invalid_times = [t for t in self.exclude_times if not (self.tmin <= t <= self.tmax)]
        if invalid_times:
            raise ValueError(f"exclude_times {invalid_times} are outside tmin-tmax range")


@dataclass(frozen=True)
class PlotConfiguration:
    """Configuration for fit quality plots."""
    model: str
    tmin_min: Optional[int] = None
    tmin_max: Optional[int] = None
    tmax_min: Optional[int] = None
    tmax_max: Optional[int] = None
    
    def __post_init__(self):
        """Validate plot configuration."""
        if self.tmin_min is not None and self.tmin_max is not None:
            if self.tmin_min >= self.tmin_max:
                raise ValueError("tmin_min must be less than tmin_max")
        
        if self.tmax_min is not None and self.tmax_max is not None:
            if self.tmax_min >= self.tmax_max:
                raise ValueError("tmax_min must be less than tmax_max")


@dataclass(frozen=True)
class MinimizerConfiguration:
    """Configuration for the fitting minimizer."""
    minimizer: str = "lmder"
    parameter_rel_tol: float = 1.0e-06
    chisquare_rel_tol: float = 1.0e-04
    max_iterations: int = 2000
    verbosity: str = "low"
    
    def __post_init__(self):
        """Validate minimizer configuration."""
        valid_minimizers = ["lmder", "scipy"]
        if self.minimizer not in valid_minimizers:
            raise ValueError(f"minimizer must be one of {valid_minimizers}")
        
        valid_verbosity = ["low", "medium", "high"]
        if self.verbosity not in valid_verbosity:
            raise ValueError(f"verbosity must be one of {valid_verbosity}")


@dataclass(frozen=True)
class ChannelInfo:
    """
    Information about a physics channel.
    
    This class encapsulates the quantum numbers and properties
    that define a specific physics channel.
    """
    isospin: int
    strangeness: int
    momentum_squared: int  # More descriptive than 'psq'
    irrep: str
    channel_type: ChannelType = ChannelType.MULTI_HADRON
    
    def __str__(self) -> str:
        """String representation of channel."""
        return f"I={self.isospin}_S={self.strangeness}_PSQ={self.momentum_squared}_{self.irrep}"
    
    @property
    def psq(self) -> int:
        """Backward compatibility property for momentum_squared."""
        return self.momentum_squared
    
    def is_single_hadron(self) -> bool:
        """Check if this is a single hadron channel."""
        return self.channel_type == ChannelType.SINGLE_HADRON
    
    def is_scattering(self) -> bool:
        """Check if this is a scattering channel."""
        return self.channel_type == ChannelType.SCATTERING


@dataclass(frozen=True)
class OperatorInfo:
    """
    Information about a correlator operator.
    
    This class encapsulates the properties that define a specific
    operator used in correlator construction.
    """
    name: str
    channel: ChannelInfo
    level: int = 0
    momentum: int = 0
    operator_type: str = "standard"
    
    def __str__(self) -> str:
        """String representation of operator."""
        return f"{self.name}_{self.channel}_L{self.level}"
    
    def __post_init__(self):
        """Validate operator information."""
        if self.level < 0:
            raise ValueError(f"level must be non-negative, got {self.level}")


@dataclass(frozen=True)
class FitResult:
    """
    Results from a correlator fit.
    
    This class encapsulates all results and metadata from a correlator fit,
    including fit parameters, quality metrics, and observables.
    """
    success: bool
    energy_value: float
    energy_error: float
    amplitude_value: Optional[float] = None
    amplitude_error: Optional[float] = None
    chisq_dof: Optional[float] = None
    quality: Optional[float] = None
    dof: Optional[int] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    covariance_matrix: Optional[List[List[float]]] = None
    fit_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fit result."""
        if self.success:
            if self.energy_error < 0:
                raise ValueError("energy_error must be non-negative for successful fits")
            if self.amplitude_error is not None and self.amplitude_error < 0:
                raise ValueError("amplitude_error must be non-negative")
    
    def has_amplitude(self) -> bool:
        """Check if amplitude information is available."""
        return self.amplitude_value is not None and self.amplitude_error is not None
    
    def is_good_quality(self, quality_threshold: float = 0.05) -> bool:
        """Check if fit quality is acceptable."""
        return self.success and (self.quality is None or self.quality > quality_threshold)


@dataclass(frozen=True)
class SpectrumLevel:
    """
    A single energy level in the spectrum.
    
    This class represents one energy eigenvalue extracted from the
    correlator analysis, including all associated metadata.
    """
    channel: ChannelInfo
    operator: OperatorInfo
    fit_result: FitResult
    level_index: int = 0
    ecm_value: Optional[float] = None
    ecm_error: Optional[float] = None
    elab_value: Optional[float] = None
    elab_error: Optional[float] = None
    
    def sort_key(self) -> float:
        """Key for sorting spectrum levels by energy."""
        if not self.fit_result.success:
            return float('inf')  # Failed fits sort to end
        
        # Prefer CM frame energy if available
        if self.ecm_value is not None:
            return self.ecm_value
        return self.fit_result.energy_value
    
    def __lt__(self, other: 'SpectrumLevel') -> bool:
        """Enable sorting of spectrum levels."""
        return self.sort_key() < other.sort_key()


class HadronNameParser:
    """
    Utility class for parsing hadron names from operator strings.
    
    This class provides methods to extract hadron information from
    operator names and count hadrons in multi-hadron operators.
    """
    
    # Common hadron identifiers
    HADRON_NAMES = ['N', 'X', 'k', 'S', 'L', 'pi', 'P', 'K', 'Delta', 'Sigma', 'Lambda']
    HADRON_SEPARATORS = ['(', '-', '[', '_']
    
    # Regex patterns for different hadron naming conventions
    HADRON_PATTERNS = [
        r'([A-Za-z]+)\(([^)]+)\)',  # hadron(momentum)
        r'([A-Za-z]+)_([0-9]+)',    # hadron_momentum
        r'([A-Za-z]+)\[([^\]]+)\]', # hadron[momentum]
    ]
    
    @classmethod
    def count_hadrons(cls, operator_name: str) -> int:
        """
        Count the number of hadrons in an operator name.
        
        Args:
            operator_name: Name of the operator
            
        Returns:
            Number of hadrons found in the name
        """
        count = 0
        temp_name = operator_name.upper()  # Case-insensitive matching
        
        for hadron in cls.HADRON_NAMES:
            for separator in cls.HADRON_SEPARATORS:
                pattern = hadron.upper() + separator
                count += temp_name.count(pattern)
                temp_name = temp_name.replace(pattern, "")
        
        return max(count, 1)  # At least one hadron
    
    @classmethod
    def extract_hadrons(cls, operator_name: str) -> List[Dict[str, str]]:
        """
        Extract detailed hadron information from an operator name.
        
        Args:
            operator_name: Name of the operator
            
        Returns:
            List of dictionaries with hadron information
        """
        hadrons = []
        
        for pattern in cls.HADRON_PATTERNS:
            matches = re.findall(pattern, operator_name)
            for hadron_name, momentum_info in matches:
                hadrons.append({
                    'name': hadron_name,
                    'momentum_info': momentum_info,
                    'full_string': f"{hadron_name}({momentum_info})"
                })
        
        return hadrons
    
    @classmethod
    def is_single_hadron(cls, operator_name: str) -> bool:
        """Check if operator represents a single hadron."""
        return cls.count_hadrons(operator_name) == 1
    
    @classmethod
    def get_hadrons(cls, operator_name: str) -> List[str]:
        """
        Get a list of hadrons in an operator name.
        
        Used for assembling non-interacting levels for ratio fits.
        Based on the "name" piece of an operator (the part that is user defined).
        
        Args:
            operator_name: Name of the operator
            
        Returns:
            List of hadron strings in format "hadron(momentum)"
        """
        sh_list = []
        sh_psq_list = []
        for hadron in cls.HADRON_NAMES:
            for tag in cls.HADRON_SEPARATORS:
                if hadron + tag in operator_name:
                    parts = operator_name.split(hadron + tag)[1:]
                    sh_list += [hadron] * len(parts)
                    sh_psq_list += [part[0] for part in parts]
        return [f"{sh}({psq})" for sh, psq in zip(sh_list, sh_psq_list)]
    
    @classmethod
    def normalize_hadron_name(cls, hadron_name: str) -> str:
        """Normalize hadron name to standard form."""
        name_mapping = {
            'nucleon': 'N',
            'pion': 'pi',
            'kaon': 'K',
            'proton': 'N',
            'neutron': 'N'
        }
        return name_mapping.get(hadron_name.lower(), hadron_name)


@dataclass
class ValidationResult:
    """Result of data validation operations."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return bool(self.errors or self.warnings)


def create_channel_from_string(channel_str: str) -> ChannelInfo:
    """
    Create a ChannelInfo object from a string representation.
    
    Args:
        channel_str: String representation of channel
        
    Returns:
        ChannelInfo object
        
    Raises:
        ValueError: If string format is invalid
    """
    # Parse string like "I=1_S=0_PSQ=0_A1"
    pattern = r'I=(-?\d+)_S=(-?\d+)_PSQ=(\d+)_(.+)'
    match = re.match(pattern, channel_str)
    
    if not match:
        raise ValueError(f"Invalid channel string format: {channel_str}")
    
    isospin, strangeness, psq, irrep = match.groups()
    
    return ChannelInfo(
        isospin=int(isospin),
        strangeness=int(strangeness),
        momentum_squared=int(psq),
        irrep=irrep
    )


def energy_sort_key(item: Union[Dict[str, Any], SpectrumLevel]) -> float:
    """
    Universal sort key function for energy levels.
    
    Handles both modern SpectrumLevel objects and legacy dictionary formats
    for backward compatibility.
    
    Args:
        item: Dictionary or SpectrumLevel containing energy information
        
    Returns:
        Energy value for sorting
    """
    if isinstance(item, SpectrumLevel):
        return item.sort_key()
    
    # Legacy dictionary format compatibility
    return item.get("ecm value", item.get("ecm_value", item.get("energy_value", 0.0)))


# Legacy compatibility aliases
energy_sort = energy_sort_key  # Backward compatibility

# Global variable for table sorting (legacy compatibility)
_sorting_index = [3]

def table_sort_set(index: int):
    """
    Set the sorting index for table sorting (legacy compatibility).
    
    Args:
        index: Index to sort by
        
    Returns:
        table_sort function
    """
    _sorting_index[0] = index
    return table_sort

def table_sort(item: List[Any]) -> Any:
    """
    Sort a list of lists by the sorting_index element (legacy compatibility).
    
    Args:
        item: List item to extract sort key from
        
    Returns:
        Sort key value
    """
    return item[_sorting_index[0]]


def get_op_name(operator_info) -> str:
    """
    Get the operator name from operator info.
    
    Args:
        operator_info: Sigmond operator info object
        
    Returns:
        Operator name string
    """
    if operator_info.isBasicLapH():
        return operator_info.getBasicLapH().getIDName()
    else:
        return operator_info.getGenIrrep().getIDName()


def validate_fit_configuration(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate a fit configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        ValidationResult with validation status and messages
    """
    result = ValidationResult(is_valid=True)
    
    # Check required fields
    required_fields = ['model', 'tmin', 'tmax']
    for field in required_fields:
        if field not in config:
            result.add_error(f"Missing required field: {field}")
    
    # Validate time range
    if 'tmin' in config and 'tmax' in config:
        if config['tmin'] >= config['tmax']:
            result.add_error(f"tmin ({config['tmin']}) must be less than tmax ({config['tmax']})")
    
    # Validate model
    if 'model' in config:
        valid_models = [model.value for model in FitModel]
        if config['model'] not in valid_models:
            result.add_warning(f"Unknown model '{config['model']}'. Valid models: {valid_models}")
    
    return result


# Backward compatibility aliases
HadronNames = HadronNameParser  # Keep old class name for compatibility 