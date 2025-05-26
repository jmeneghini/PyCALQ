#!/usr/bin/env python3
"""
Test runner for PyCALQ tests.

This script provides a convenient way to run different types of tests
with proper reporting and coverage analysis.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_unit_tests(verbose=False, coverage=False, module=None):
    """
    Run unit tests.
    
    Args:
        verbose: Enable verbose output
        coverage: Generate coverage report
        module: Specific module to test (e.g., 'core', 'analysis')
    """
    if module:
        cmd = ["python", "-m", "pytest", f"tests/unit/{module}/"]
        description = f"Unit Tests - {module.title()} Module"
    else:
        cmd = ["python", "-m", "pytest", "tests/unit/"]
        description = "Unit Tests - All Modules"
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=fvspectrum", "--cov-report=html", "--cov-report=term"])
    
    cmd.extend(["-m", "unit"])
    
    return run_command(cmd, description)


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "integration"])
    
    return run_command(cmd, "Integration Tests")


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=fvspectrum", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "All Tests")


def run_linting():
    """Run code linting."""
    success = True
    
    # Run flake8
    cmd = ["python", "-m", "flake8", "fvspectrum/", "--max-line-length=100", "--ignore=E203,W503"]
    if not run_command(cmd, "Flake8 Linting"):
        success = False
    
    # Run mypy (if available)
    try:
        cmd = ["python", "-m", "mypy", "fvspectrum/", "--ignore-missing-imports"]
        if not run_command(cmd, "MyPy Type Checking"):
            success = False
    except FileNotFoundError:
        print("MyPy not available, skipping type checking")
    
    return success


def check_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-cov",
        "flake8"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run PyCALQ tests")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "all", "lint"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if test dependencies are installed"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        if check_dependencies():
            print("All test dependencies are installed.")
            return 0
        else:
            return 1
    
    # Change to project root directory
    os.chdir(project_root)
    
    print(f"Running tests from: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    
    success = True
    
    if args.test_type == "unit":
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.test_type == "integration":
        success = run_integration_tests(args.verbose)
    elif args.test_type == "all":
        success = run_all_tests(args.verbose, args.coverage)
    elif args.test_type == "lint":
        success = run_linting()
    
    if success:
        print(f"\n✅ {args.test_type.title()} tests completed successfully!")
        return 0
    else:
        print(f"\n❌ {args.test_type.title()} tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 