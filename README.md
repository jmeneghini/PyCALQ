# PyCALQ
**Correlator Analysis and Lüscher Quantization Condition**

Full analysis chain of the finite volume spectrum from two-point correlators to phase-shifts and other infinite-volume observables using the Lüscher formalism.

## 🚀 Recent Major Refactoring

PyCALQ has been **completely refactored** with a modern, maintainable architecture and **simplified task names** for better usability.

### ✨ What's New
- **Simplified Task Names**: Clean, intuitive task names (`preview`, `average`, `rotate`, `fit`, `compare`)
- **Clean Architecture**: Modular design with separation of concerns
- **Type Safety**: Full type hints throughout the codebase
- **Comprehensive Testing**: Unit and integration test suite
- **Better Error Handling**: Clear validation and error messages
- **Enhanced Documentation**: Self-documenting code with detailed docstrings
- **Improved Performance**: Better memory management and optimization

### 📁 New Code Structure
```
fvspectrum/
├── core/                    # Base classes and data structures
├── analysis/               # Correlator processing components
├── fitting/                # Spectrum fitting components
├── plotting/               # Visualization components
├── tasks/                  # Refactored task implementations
├── utils/                  # Modular utility functions
└── legacy_backup/          # Original files (preserved for reference)
```

### 🔄 Task Name Changes
The task names have been simplified for better usability:
- `preview_corrs` → `preview`
- `average_corrs` → `average`
- `rotate_corrs` → `rotate`
- `fit_spectrum` → `fit`
- `compare_spectrums` → `compare`

**All functionality remains identical** - only the YAML task names have changed for simplicity.

### 🎯 Final Simplification (Latest)
The latest update **removed all legacy wrapper complexity** for maximum simplicity:
- ✅ **Eliminated 7 wrapper files** (~2,000 lines of compatibility code)
- ✅ **Direct task class imports** (no wrapper layer overhead)
- ✅ **Simplified task names** (50% shorter, more intuitive)
- ✅ **Cleaner architecture** (single source of truth for each task)
- ✅ **Better maintainability** (fewer files, cleaner dependencies)

For detailed information about the refactoring, see [`README_REFACTORING.md`](README_REFACTORING.md).

---

## Prerequisites

[sigmond pybindings (pip branch)](https://github.com/andrewhanlon/sigmond/tree/pip)

## Setup
```bash
cd PyCALQ/
pip install -r requirements.txt

# For development and testing
pip install -r requirements-test.txt
```

## Sample Usage

```bash
# Basic usage (unchanged from before)
python run.py -g general_config.yml -t task_config.yml

# Get help
python run.py -h
```

```
usage: run.py [-h] [-g GENERAL] [-t TASKS [TASKS ...]]

options:
  -h, --help            show this help message and exit
  -g GENERAL, --general GENERAL
                        general configuration file
  -t TASKS [TASKS ...], --tasks TASKS [TASKS ...]
                        task(s) configuration file(s)
```

## 🧪 Testing

The refactored codebase includes a comprehensive test suite:

```bash
# Run all tests
python tests/run_tests.py all

# Run specific test types
python tests/run_tests.py unit          # Unit tests only
python tests/run_tests.py integration   # Integration tests only
python tests/run_tests.py lint          # Code linting

# With coverage reporting
python tests/run_tests.py all --coverage --verbose
```

---

## Configuration

### General Configuration

Any task requires a general configuration file. At minimum:
```yaml
general:
    project_dir: /path/to/directory/
    ensemble_id: cls21_s64_t128_D200
```

Complete general configuration options:
```yaml
general:
    project_dir: /path/to/directory/
    ensemble_id: cls21_s64_t128_D200
    sampling_info:                      # Optional
      mode: Bootstrap # or Jackknife     # Default: Jackknife
      number_resampling: 800            # Required for Bootstrap
      seed: 3103                        # Required for Bootstrap
      boot_skip: 297                    # Required for Bootstrap
    tweak_ensemble:                     # Optional
      omissions: [2000]                 # Default: []
      rebin: 10                         # Default: 1
    subtract_vev: false                 # Optional, default: false
```

**General Configuration Parameters:**
- `project_directory` - (str) New or existing directory for the project
- `ensemble_id` - (str) ID associated with ensemble infos in `fvspectrum/sigmond_utils/ensembles.xml`
- `sampling_info` - (dict) Monte Carlo resampling configuration
  - `mode` - (str) 'Bootstrap' or 'Jackknife' resampling method
  - `number_resampling` - (int) Number of bootstrap samples
  - `seed` - (int) Bootstrap random number seed
  - `boot_skip` - (int) Random numbers to skip before sampling
- `tweak_ensemble` - (dict) Sample set modifications
  - `omissions` - (list) Sample indices to omit from calculations
  - `rebin` - (int) Block size for averaging before resampling

### Task Configuration

Task configuration files follow this structure:
```yaml
tasks:
  - [task1]:
      [task1_configs]
  - [task2]:
      [task2_configs]
  # ...
```

**Available Tasks:**
- `preview` - Preview and analyze correlator data
- `average` - Average correlators over irreps and momenta
- `rotate` - Perform GEVP rotation to extract eigenvalues
- `fit` - Fit correlators to determine energy spectrum
- `compare` - Compare different spectrum analyses

Tasks are executed in the order listed above, regardless of YAML order. Multiple instances of the same task run in YAML order.

### Universal Task Parameters

All tasks support these common parameters:
```yaml
  figheight: 6                  # Optional, default: 6
  figwidth: 8                   # Optional, default: 8
  info: true                    # Optional, default: false
  plot: true                    # Optional, default: true
  create_pdfs: true             # Optional, default: true
  create_pickles: true          # Optional, default: true
  create_summary: true          # Optional, default: true
  generate_estimates: true      # Optional, default: true
```

**Universal Parameter Descriptions:**
- `figheight/figwidth` - (int) Matplotlib figure dimensions
- `info` - (bool) Print task configuration information
- `plot` - (bool) Generate plots (if false, no plots created)
- `create_pdfs` - (bool) Generate PDF plots
- `create_pickles` - (bool) Generate matplotlib pickle files
- `create_summary` - (bool) Generate LaTeX PDF summary document
- `generate_estimates` - (bool) Generate CSV files with bootstrap/jackknife estimates

### Common Task Parameters

Many tasks share these parameters:
```yaml
    raw_data_files:               # Required for data input tasks
    - /path/to/correlator/data.bin
    averaged_input_correlators_dir: /path/to/averaged/  # Optional
    reference_particle: pi        # Optional
    tmin: 0                       # Optional, default varies by task
    tmax: 64                      # Optional, default varies by task
    only:                         # Optional channel filter
    - psq=0
    - isosinglet S=0 E PSQ=3
    omit:                         # Optional channel exclusion
    - psq=0
    - isosinglet S=0 E PSQ=3
```

**Common Parameter Descriptions:**
- `raw_data_files` - (str/list) Raw correlator data files or directories
- `averaged_input_correlators_dir` - (str/list) Averaged data location
- `reference_particle` - (str) Reference particle for normalization
- `tmin/tmax` - (int) Time range for analysis
- `only` - (list) Include only specified channels (overrides `omit`)
- `omit` - (list) Exclude specified channels

---

## Tasks

### Preview Correlators
**Purpose:** Read and estimate/plot Lattice QCD temporal correlator data files.

```yaml
- preview:
    raw_data_files:               # Required 
    - /path/to/correlator/data.bin
    create_pdfs: true             # Optional, default: true
    create_pickles: true          # Optional, default: true
    create_summary: true          # Optional, default: true
    generate_estimates: true      # Optional, default: true
    # ... universal parameters
```

### Average Correlators
**Purpose:** Automatically average correlators within the same irrep row and total momentum.

```yaml
- average:
    raw_data_files:                       # Required
    - /path/to/correlator/data.bin
    average_by_bins: false                # Optional, default: false
    average_hadron_irrep_info: true       # Optional, default: true
    average_hadron_spatial_info: true     # Optional, default: true
    erase_original_matrix_from_memory: false # Optional, default: false
    ignore_missing_correlators: true      # Optional, default: true
    separate_mom: false                   # Optional, default: false
    run_tag: "unique"                     # Optional, default: ""
    tmax: 64                              # Optional, default: 64
    tmin: 0                               # Optional, default: 0
    # ... universal parameters
```

**Unique Parameters:**
- `average_by_bins` - (bool) Average bin-by-bin vs. by resampling
- `average_hadron_irrep_info` - (bool) Average over different irreps
- `average_hadron_spatial_info` - (bool) Average over spatial configurations
- `erase_original_matrix_from_memory` - (bool) Save memory by erasing originals
- `ignore_missing_correlators` - (bool) Handle missing correlators gracefully
- `separate_mom` - (bool) Separate output by momentum
- `run_tag` - (str) User-defined tag for output files

### Rotate Correlators
**Purpose:** Pivot correlator matrix and return time-dependent eigenvalues using GEVP.

```yaml
- rotate:
    tN: 5                                 # Required - normalize time
    t0: 5                                 # Required - metric time  
    tD: 10                                # Required - diagonalize time
    averaged_input_correlators_dir: null  # Optional, auto-detected
    max_condition_number: 50              # Optional, default: 50
    pivot_type: 0                         # Optional, default: 0 (single pivot)
    precompute: true                      # Optional, default: true
    rotate_by_samplings: true             # Optional, default: true
    run_tag: "unique"                     # Optional, default: ""
    used_averaged_bins: true              # Optional, default: true
    omit_operators: []                    # Optional, default: []
    # ... universal and common parameters
```

**Unique Parameters:**
- `t0` - (int) Metric time for pivot setup
- `tN` - (int) Normalize time for pivot
- `tD` - (int) Diagonalize time for pivot
- `pivot_type` - (int) 0=single pivot, 1=rolling pivot
- `max_condition_number` - (float) Maximum eigenvalue ratio for stability
- `precompute` - (bool) Precompute bootstrap samples in sigmond
- `rotate_by_samplings` - (bool) Rotate by samples vs. bins
- `used_averaged_bins` - (bool) Use bin files vs. sampling files
- `omit_operators` - (list) Operators to exclude from pivot

### Fit Spectrum
**Purpose:** Fit single hadron and/or rotated correlators to determine energy spectrum.

```yaml
- fit:
    default_corr_fit:                     # Required (unless both interacting/noninteracting specified)
        model: 1-exp                        # Required
        tmin: 15                            # Required
        tmax: 25                            # Required
        exclude_times: []                   # Optional, default: []
        initial_params: {}                  # Optional, default: {}
        noise_cutoff: 0.0                   # Optional, default: 0.0
        priors: {}                          # Optional, default: {}
        ratio: false                        # Optional, default: false
        sim_fit: false                      # Optional, default: false
        tmin_plots: []                      # Optional, default: []
        tmax_plots: []                      # Optional, default: []
    
    # Alternative: separate configs for interacting vs non-interacting
    default_noninteracting_corr_fit: null # Optional
    default_interacting_corr_fit: null    # Optional
    
    # Operator-specific overrides
    correlator_fits: {}                   # Optional, default: {}
    
    # Single hadron configuration (required for ratio fits)
    single_hadrons:                       # Required for ratio fits
        pi:                                 # Hadron name
        - operator_name                     # Operators ordered by momentum
    
    # Ratio fit configuration
    single_hadrons_ratio: {}              # Optional, default: {}
    non_interacting_levels: {}            # Optional, required for ratio fits
    
    # Analysis options
    compute_overlaps: true                # Optional, default: true
    correlated: true                      # Optional, default: true
    do_interacting_fits: true             # Optional, default: true
    non_interacting_energy_sums: false    # Optional, default: false
    
    # File locations (auto-detected if not specified)
    averaged_input_correlators_dir: null  # Optional
    rotated_input_correlators_dir: null   # Optional
    pivot_file: null                      # Optional
    
    # Pivot parameters (auto-detected if not specified)
    tN: null                              # Optional
    t0: null                              # Optional
    tD: null                              # Optional
    pivot_type: null                      # Optional
    
    # Run configuration
    run_tag: ""                           # Optional, default: ""
    rotate_run_tag: ""                    # Optional, default: ""
    precompute: true                      # Optional, default: true
    use_rotated_samplings: true           # Optional, default: true
    used_averaged_bins: true              # Optional, default: true
    
    # Minimizer settings
    minimizer_info:                       # Optional
        chisquare_rel_tol: 0.0001           # Default: 0.0001
        max_iterations: 2000                # Default: 2000
        minimizer: lmder                    # Default: lmder
        parameter_rel_tol: 1.0e-06          # Default: 1.0e-06
        verbosity: low                      # Default: low
    
    # Threshold analysis
    thresholds: []                        # Optional, default: []
    
    # ... universal and common parameters
```

**Fit Configuration Parameters:**
- `model` - (str) Fit model (see [sigmond fit models](https://github.com/andrewhanlon/sigmond_scripts/blob/pip/src/sigmond_scripts/fit_info.py))
- `tmin/tmax` - (int) Time range for fitting
- `exclude_times` - (list) Time values to exclude from fit
- `initial_params` - (dict) Initial parameter values `{param_name: value}`
- `noise_cutoff` - (float) Exclude data where error/value > cutoff
- `priors` - (dict) Prior constraints `{param_name: {"Mean": value, "Error": width}}`
- `ratio` - (bool) Use ratio correlators with non-interacting denominators
- `sim_fit` - (bool) Perform simultaneous fits with single hadrons

**Analysis Parameters:**
- `compute_overlaps` - (bool) Calculate and plot operator overlaps
- `correlated` - (bool) Use correlated vs uncorrelated fits
- `do_interacting_fits` - (bool) Fit interacting (rotated) correlators
- `single_hadrons` - (dict) Map hadron names to operator lists
- `non_interacting_levels` - (dict) Define non-interacting levels for ratio fits

### Compare Spectrums
**Purpose:** Compare spectrum results from different analyses side-by-side.

```yaml
- compare:
    compare_plots:                # Required - list of comparison types
    - compare_gevp:               # Compare different pivot configurations
        gevp_values:              # Required - list of pivot configs
        - t0: 8
          tD: 16
          tN: 5
          pivot_type: 0           # Optional, default: 0
        - t0: 8
          tD: 18
          tN: 5
        rebin: 1                  # Optional
        run_tag: ''               # Optional, default: ''
        sampling_mode: J          # Optional
    
    - compare_files: []           # Compare explicitly named files
    
    - compare_rebin:              # Compare different rebinning schemes
        rebin_values: [1, 2, 4]   # Required
        run_tag: ''               # Optional, default: ''
        sampling_mode: J          # Optional
        pivot_type: 0             # Optional, default: 0
        t0: 8                     # Required
        tN: 5                     # Required
        tD: 18                    # Required
    
    - compare_tags:               # Compare different run tags
        filetags: []              # Required
        sampling_mode: J          # Optional
        pivot_type: 0             # Optional, default: 0
        t0: 8                     # Required
        tN: 5                     # Required
        tD: 18                    # Required
        rebin: 1                  # Optional
    
    figheight: 8                  # Optional, default: 8
    figwidth: 15                  # Optional, default: 15
    plot: true                    # Required
    plot_deltaE: true             # Optional, default: true
    reference_particle: P         # Optional, default: null
```

**Comparison Types:**
- `compare_gevp` - Compare different GEVP pivot configurations
- `compare_files` - Compare explicitly specified data files
- `compare_rebin` - Compare different rebinning factors
- `compare_tags` - Compare different analysis run tags
- `plot_deltaE` - Include energy shift plots from non-interacting levels

---

## 🔧 Development

### Adding New Tasks

The refactored architecture makes it easy to add new tasks. Create a new task by inheriting from the base classes:

```python
from fvspectrum.core.base_task import CorrelatorAnalysisTask

class MyNewTask(CorrelatorAnalysisTask):
    @property
    def info(self) -> str:
        return "Documentation for my new task"
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        defaults = super()._get_default_parameters()
        defaults.update({
            'my_param': 'default_value'
        })
        return defaults
    
    def run(self) -> None:
        # Implement task logic
        pass
    
    def plot(self) -> None:
        # Implement plotting logic
        pass
```

### Using Individual Components

The new modular architecture allows using components independently:

```python
from fvspectrum.analysis.correlator_processor import CorrelatorProcessor
from fvspectrum.fitting.spectrum_fitter import SpectrumFitter
from fvspectrum.plotting.spectrum_plotter import SpectrumPlotter

# Use components directly
processor = CorrelatorProcessor(data_handler, project_handler)
fitter = SpectrumFitter(mcobs_handler, project_handler)
plotter = SpectrumPlotter(proj_files_handler, plot_config)
```

### Task Integration

To integrate a new task into PyCALQ:

1. **Add to task manager** (`general/task_manager.py`):
```python
class Task(Enum):
    # ... existing tasks
    my_new_task = 6
```

2. **Update PyCALQ configuration** (`pycalq.py`):
```python
# Add import
import fvspectrum.my_new_task_new

# Add to task mapping
TASK_MAP = {
    # ... existing mappings
    tm.Task.my_new_task: fvspectrum.my_new_task_new.MyNewTask,
}

# Add documentation
TASK_DOC = {
    # ... existing docs
    tm.Task.my_new_task: fvspectrum.my_new_task_new.doc,
}
```

3. **Create compatibility wrapper** (`fvspectrum/my_new_task_new.py`):
```python
from fvspectrum.tasks.my_new_task import MyNewTask, TASK_DOCUMENTATION

doc = TASK_DOCUMENTATION
MyNewTaskClass = MyNewTask
```

---

## 📚 Additional Documentation

- **Refactoring Details**: [`README_REFACTORING.md`](README_REFACTORING.md) - Complete refactoring documentation
- **Refactoring Summary**: [`REFACTORING_SUMMARY.md`](REFACTORING_SUMMARY.md) - Technical summary of changes
- **Test Documentation**: [`tests/README.md`](tests/README.md) - Testing framework documentation

---

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're using correct import paths for new components
2. **Configuration Errors**: Check YAML syntax and parameter names
3. **File Not Found**: Verify data file paths are correct
4. **Memory Issues**: Use appropriate chunking for large datasets

### Debugging

Enable detailed logging for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your analysis - will show detailed debug information
```

### Legacy Code Access

Original implementations are preserved in `fvspectrum/legacy_backup/` for reference.

---

## 📋 To Do

### Current Issues
- **Spectrum task**: Separate estimates for interacting and non-interacting levels with same channel names
- **Spectrum task**: Skip nonzero momentum single hadron fits when operator overlaps are disabled

### Desired Updates
- **Scheduler Integration**: Internal setup for SLURM or other scheduler systems
- **Enhanced Validation**: More comprehensive YAML configuration validation
- **Performance Monitoring**: Built-in profiling and benchmarking tools
- **Web Interface**: Optional web-based analysis interface

---

## 📄 License

See [`LICENSE`](LICENSE) for license information.

## 🤝 Contributing

1. **Write tests** for any new functionality
2. **Run the test suite** before submitting changes: `python tests/run_tests.py all`
3. **Follow established patterns** in the refactored codebase
4. **Update documentation** for user-facing changes

The refactored architecture makes PyCALQ more maintainable and extensible while preserving all existing functionality. Happy analyzing! 🎉
