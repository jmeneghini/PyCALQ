"""
Unit tests for core data structures.

This module tests the data classes, enums, and utility functions
defined in fvspectrum.core.data_structures.
"""

import pytest
import numpy as np
from dataclasses import FrozenInstanceError

from fvspectrum.core.data_structures import (
    FitConfiguration, FitResult, SpectrumLevel, MinimizerConfiguration,
    ObservableType, HadronNames
)


class TestFitConfiguration:
    """Test the FitConfiguration data class."""
    
    def test_basic_creation(self):
        """Test basic creation of FitConfiguration."""
        config = FitConfiguration(
            model="1-exp",
            tmin=10,
            tmax=20
        )
        
        assert config.model == "1-exp"
        assert config.tmin == 10
        assert config.tmax == 20
        assert config.exclude_times == []
        assert config.initial_params == {}
        assert config.noise_cutoff == 0.0
        assert config.priors == {}
        assert config.ratio is False

        assert config.tmin_plots == []
        assert config.tmax_plots == []
    
    def test_with_optional_parameters(self):
        """Test creation with optional parameters."""
        config = FitConfiguration(
            model="2-exp",
            tmin=15,
            tmax=25,
            exclude_times=[18, 19],
            initial_params={"A0": 1.0, "E0": 0.5},
            noise_cutoff=0.01,
            priors={"E0": (0.4, 0.6)},
            ratio=True,

            tmin_plots=[12, 13, 14],
            tmax_plots=[23, 24, 25]
        )
        
        assert config.model == "2-exp"
        assert config.exclude_times == [18, 19]
        assert config.initial_params == {"A0": 1.0, "E0": 0.5}
        assert config.noise_cutoff == 0.01
        assert config.priors == {"E0": (0.4, 0.6)}
        assert config.ratio is True

        assert config.tmin_plots == [12, 13, 14]
        assert config.tmax_plots == [23, 24, 25]
    
    def test_immutability(self):
        """Test that FitConfiguration is immutable."""
        config = FitConfiguration(model="1-exp", tmin=10, tmax=20)
        
        with pytest.raises(FrozenInstanceError):
            config.model = "2-exp"
        
        with pytest.raises(FrozenInstanceError):
            config.tmin = 15


class TestFitResult:
    """Test the FitResult data class."""
    
    def test_successful_fit(self):
        """Test creation of successful fit result."""
        result = FitResult(
            success=True,
            energy_value=1.5,
            energy_error=0.05,
            amplitude_value=2.0,
            amplitude_error=0.1,
            chisq_dof=1.2,
            quality=0.95,
            dof=10,
            parameters={"A0": 2.0, "E0": 1.5}
        )
        
        assert result.success is True
        assert result.energy_value == 1.5
        assert result.energy_error == 0.05
        assert result.amplitude_value == 2.0
        assert result.amplitude_error == 0.1
        assert result.chisq_dof == 1.2
        assert result.quality == 0.95
        assert result.dof == 10
        assert result.parameters == {"A0": 2.0, "E0": 1.5}
    
    def test_failed_fit(self):
        """Test creation of failed fit result."""
        result = FitResult(
            success=False,
            energy_value=0.0,
            energy_error=0.0
        )
        
        assert result.success is False
        assert result.energy_value == 0.0
        assert result.energy_error == 0.0
        assert result.amplitude_value is None
        assert result.amplitude_error is None
        assert result.chisq_dof is None
        assert result.quality is None
        assert result.dof is None
        assert result.parameters is None
    
    def test_immutability(self):
        """Test that FitResult is immutable."""
        result = FitResult(success=True, energy_value=1.0, energy_error=0.1)
        
        with pytest.raises(FrozenInstanceError):
            result.success = False
        
        with pytest.raises(FrozenInstanceError):
            result.energy_value = 2.0


class TestSpectrumLevel:
    """Test the SpectrumLevel data class."""
    
    def test_creation(self):
        """Test creation of SpectrumLevel."""
        fit_result = FitResult(success=True, energy_value=1.5, energy_error=0.05)
        
        level = SpectrumLevel(
            channel="test_channel",
            operator="test_operator",
            fit_result=fit_result
        )
        
        assert level.channel == "test_channel"
        assert level.operator == "test_operator"
        assert level.fit_result == fit_result
    
    def test_sort_key(self):
        """Test the sort_key method."""
        fit_result1 = FitResult(success=True, energy_value=1.0, energy_error=0.05)
        fit_result2 = FitResult(success=True, energy_value=2.0, energy_error=0.05)
        fit_result3 = FitResult(success=False, energy_value=0.0, energy_error=0.0)
        
        level1 = SpectrumLevel("ch1", "op1", fit_result1)
        level2 = SpectrumLevel("ch2", "op2", fit_result2)
        level3 = SpectrumLevel("ch3", "op3", fit_result3)
        
        # Successful fits should sort by energy
        assert level1.sort_key() < level2.sort_key()
        
        # Failed fits should sort to the end
        assert level3.sort_key() > level1.sort_key()
        assert level3.sort_key() > level2.sort_key()


class TestMinimizerConfiguration:
    """Test the MinimizerConfiguration data class."""
    
    def test_default_creation(self):
        """Test creation with default values."""
        config = MinimizerConfiguration()
        
        assert config.minimizer == "lmder"
        assert config.parameter_rel_tol == 1e-6
        assert config.chisquare_rel_tol == 1e-4
        assert config.max_iterations == 2000
        assert config.verbosity == "low"
    
    def test_custom_creation(self):
        """Test creation with custom values."""
        config = MinimizerConfiguration(
            minimizer="scipy",
            parameter_rel_tol=1e-8,
            chisquare_rel_tol=1e-6,
            max_iterations=5000,
            verbosity="high"
        )
        
        assert config.minimizer == "scipy"
        assert config.parameter_rel_tol == 1e-8
        assert config.chisquare_rel_tol == 1e-6
        assert config.max_iterations == 5000
        assert config.verbosity == "high"


class TestObservableType:
    """Test the ObservableType enum."""
    
    def test_enum_values(self):
        """Test that enum has expected values."""
        assert ObservableType.DELAB.value == "dElab"
        assert ObservableType.ELAB.value == "elab"
        assert ObservableType.ECM.value == "ecm"
        assert ObservableType.ECM_REF.value == "ecm_ref"
        assert ObservableType.AMP.value == "amp"
        assert ObservableType.DELAB_REF.value == "dElab_ref"
    
    def test_enum_membership(self):
        """Test enum membership."""
        assert "dElab" in [obs.value for obs in ObservableType]
        assert "invalid" not in [obs.value for obs in ObservableType]


class TestHadronNames:
    """Test the HadronNames utility class."""
    
    def test_parse_simple_hadron(self):
        """Test parsing simple hadron names."""
        result = HadronNames.parse_hadron_name("pi")
        assert result == ["pi"]
        
        result = HadronNames.parse_hadron_name("N")
        assert result == ["N"]
    
    def test_parse_multi_hadron(self):
        """Test parsing multi-hadron names."""
        result = HadronNames.parse_hadron_name("pi_pi")
        assert result == ["pi", "pi"]
        
        result = HadronNames.parse_hadron_name("N_pi")
        assert result == ["N", "pi"]
    
    def test_parse_complex_hadron(self):
        """Test parsing complex hadron names with momentum."""
        result = HadronNames.parse_hadron_name("pi(p=1)")
        assert "pi" in result[0]
        
        result = HadronNames.parse_hadron_name("N(p=0)_pi(p=1)")
        assert len(result) == 2
    
    def test_get_hadron_count(self):
        """Test counting hadrons in names."""
        assert HadronNames.get_hadron_count("pi") == 1
        assert HadronNames.get_hadron_count("pi_pi") == 2
        assert HadronNames.get_hadron_count("N_pi_K") == 3
    
    def test_is_single_hadron(self):
        """Test single hadron detection."""
        assert HadronNames.is_single_hadron("pi") is True
        assert HadronNames.is_single_hadron("N") is True
        assert HadronNames.is_single_hadron("pi_pi") is False
        assert HadronNames.is_single_hadron("N_pi") is False
    
    def test_normalize_name(self):
        """Test name normalization."""
        assert HadronNames.normalize_name("PI") == "pi"
        assert HadronNames.normalize_name("nucleon") == "N"
        assert HadronNames.normalize_name("pion") == "pi"


if __name__ == "__main__":
    pytest.main([__file__]) 