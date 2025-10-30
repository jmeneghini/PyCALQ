#!/usr/bin/env python3
"""
PyCALQ Command Line Interface

This module provides the main CLI interface for PyCALQ, including:
- Main PyCALQ analysis functionality (-g/-t or -f options)
- Estimates to fit config conversion (--estimates-to-fit-cfg)
"""
import argparse
import sys
import csv
import yaml
import os
import multiprocessing as mp

# Import the main PyCALQ functionality
# Defer heavy imports (e.g., sigmond) until needed by the 'run' command

# Constants for csv2fitsyaml functionality
ISO_MAP = {
    "singlet": "isosinglet",
    "doublet": "isodoublet",
    "triplet": "isotriplet",
}

def _add_estimates_arguments(parser):
    parser.add_argument('csv_file', help='Input CSV file with estimates')
    parser.add_argument('-o', '--output', help='Output YAML file (default: print to stdout)')

def row_key(r):
    """Convert CSV row to YAML key format"""
    iso = ISO_MAP.get(r["isospin"].strip(), r["isospin"].strip())
    S = r["strangeness"].strip()
    irrep = r["irrep"].strip()
    psq_or_p = r["momentum"].strip()
    rot = r["rotate level"].strip()
    if psq_or_p == "0":
        return f"{iso} S={S} P=(0,0,0) {irrep} ROT {rot}"
    else:
        return f"{iso} S={S} PSQ={psq_or_p} {irrep} ROT {rot}"

def estimates_to_fit_cfg(csv_path, output_path=None):
    """Convert estimates CSV to fit configuration YAML"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Generate output
    output_lines = []
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            key = row_key(r)
            tmin = r["tmin"].strip()
            tmax = r.get("tmax", "").strip()
            output_lines.append(f"{key}:")
            output_lines.append(f"  tmin: {int(float(tmin))}")
            if tmax != "":
                output_lines.append(f"  tmax: {int(float(tmax))}")

    output_content = "\n".join(output_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_content)
        print(f"Fit configuration written to: {output_path}")
    else:
        print(output_content)

def parse_full_input_file(full_input_path):
    """Parse a full_input_X.yml file and extract general and task configurations"""
    if not os.path.exists(full_input_path):
        raise FileNotFoundError(f"Full input file not found: {full_input_path}")

    with open(full_input_path, 'r') as f:
        full_config = yaml.safe_load(f)

    if not full_config:
        raise ValueError(f"Empty or invalid YAML file: {full_input_path}")

    # Extract general config
    if 'general' not in full_config:
        raise ValueError(f"No 'general' section found in {full_input_path}")

    general_config = {'general': full_config['general']}

    # Extract task configs - everything except 'general'
    task_configs = {'tasks': []}
    for key, value in full_config.items():
        if key != 'general':
            task_configs['tasks'].append({key: value})

    if not task_configs['tasks']:
        raise ValueError(f"No task configurations found in {full_input_path}")

    return general_config, task_configs

def setup_parser():
    """Set up the command line argument parser"""
    parser = argparse.ArgumentParser(
        prog='pycalq',
        description='Python Calculational Analysis for Lattice QCD (PyCALQ)',
        epilog='For more information, see the PyCALQ documentation.'
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Main analysis command (default)
    main_parser = subparsers.add_parser('run', help='Run PyCALQ analysis')

    # Add the full input option
    main_parser.add_argument('-f', "--full-input",
                            help="full input configuration file (contains both general and task configs)")

    # Traditional separate files
    main_parser.add_argument('-g', "--general", help="general configuration file")
    main_parser.add_argument('-t', "--tasks", nargs='+', help="task(s) configuration file(s)")

    # Estimates to fit config converter
    convert_parser = subparsers.add_parser('estimates-to-fit-cfg',
                                          help='Convert estimates CSV to fit configuration YAML')
    _add_estimates_arguments(convert_parser)

    # If no subcommand provided, assume 'run'
    return parser

def set_multiprocessing_start_method():
    """Set the multiprocessing start method to 'fork' to get around Pybind11 pickling."""
    required_method = 'fork'
    if required_method in mp.get_all_start_methods():
        mp.set_start_method(required_method, force=True)
    else:
        raise RuntimeError(f"Required multiprocessing start method '{required_method}' is not available on this platform.")

def main():
    """Main CLI entry point"""
    parser = setup_parser()
    args = parser.parse_args()

    try:
        if args.command == 'estimates-to-fit-cfg':
            # Handle CSV to fit config conversion
            estimates_to_fit_cfg(args.csv_file, args.output)

        elif args.command == 'run' or args.command is None:
            # Handle main PyCALQ analysis
            full_input = getattr(args, 'full_input', None)
            general_arg = getattr(args, 'general', None)
            tasks_arg = getattr(args, 'tasks', None)

            # Validate arguments - either -f OR both -g and -t must be provided
            if full_input:
                if general_arg or tasks_arg:
                    parser.error("Cannot use -f/--full-input together with -g/--general or -t/--tasks")
                gen_configs, task_configs = parse_full_input_file(full_input)
            else:
                # Using separate files - both must be provided
                if not general_arg or not tasks_arg:
                    parser.error("Must provide either -f/--full-input OR both -g/--general and -t/--tasks")
                gen_configs, task_configs = general_arg, tasks_arg

            set_multiprocessing_start_method() # Ensure 'fork' start method for multiprocessing before importing pycalq
            try:
                import pycalq  # lazy import to avoid requiring heavy deps for other subcommands
            except Exception as e:
                raise RuntimeError(
                    "Failed to import PyCALQ core. Ensure dependencies (e.g., sigmond) are installed and use the spectrum environment."
                ) from e

            # Run PyCALQ
            if task_configs:
                pycalq_instance = pycalq.PyCALQ(gen_configs, task_configs)
            else:
                pycalq_instance = pycalq.PyCALQ(gen_configs)
            pycalq_instance.run()

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
