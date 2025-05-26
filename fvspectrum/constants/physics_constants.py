"""
Physical constants for lattice QCD analysis.

This module contains physical constants and reference values used in
lattice QCD spectrum analysis and finite volume calculations.
"""

from typing import Dict

# Fundamental constants
HBAR_C = 197.3269804  # MeV·fm (2018 CODATA value)

# Lattice spacing units and conversions
LATTICE_SPACING_UNITS = {
    'fm': 1.0,
    'GeV_inv': 1.0 / HBAR_C,  # Convert fm to GeV^-1
    'MeV_inv': 1000.0 / HBAR_C,  # Convert fm to MeV^-1
}

# Reference particle masses (in MeV)
REFERENCE_MASSES = {
    'pi': 139.57039,      # Charged pion mass
    'pi0': 134.9768,      # Neutral pion mass
    'K': 493.677,         # Charged kaon mass
    'K0': 497.611,        # Neutral kaon mass
    'N': 938.272088,      # Nucleon (proton) mass
    'Lambda': 1115.683,   # Lambda baryon mass
    'Sigma': 1189.37,     # Sigma baryon mass
    'Delta': 1232.0,      # Delta baryon mass
    'rho': 775.26,        # Rho meson mass
    'omega': 782.65,      # Omega meson mass
    'phi': 1019.461,      # Phi meson mass
}

# Typical lattice QCD ensemble parameters
ENSEMBLE_PARAMETERS = {
    'typical_volumes': [16, 24, 32, 48, 64, 96],  # Typical lattice sizes
    'typical_beta_values': [5.8, 6.0, 6.2, 6.4, 6.6],  # Typical gauge couplings
    'typical_masses': {  # Typical quark masses in lattice units
        'light': [0.001, 0.002, 0.004, 0.008],
        'strange': [0.02, 0.03, 0.04, 0.05],
        'charm': [0.2, 0.3, 0.4, 0.5]
    }
}

# Finite volume corrections and thresholds
FINITE_VOLUME = {
    'min_mL_threshold': 3.0,  # Minimum mL for reliable finite volume analysis
    'max_mL_threshold': 10.0,  # Maximum mL before finite volume effects negligible
    'luescher_validity_threshold': 0.1,  # Maximum momentum for Lüscher formula validity
}

# Physical scales and conversion factors
PHYSICAL_SCALES = {
    'r0': 0.5,  # Sommer scale in fm (approximate)
    'r1': 0.31,  # Alternative Sommer scale in fm
    'w0': 0.1755,  # Wilson flow scale in fm
    't0': 0.15,  # Wilson flow time scale in fm^2
}

# Quantum numbers and symmetries
QUANTUM_NUMBERS = {
    'isospin_values': [0, 0.5, 1, 1.5, 2],
    'strangeness_values': [-3, -2, -1, 0, 1, 2, 3],
    'momentum_squared_max': 25,  # Typical maximum momentum squared
    'irrep_names': {
        'cubic': ['A1', 'A2', 'E', 'T1', 'T2'],
        'little_group': ['A1', 'A2', 'B1', 'B2', 'E']
    }
} 