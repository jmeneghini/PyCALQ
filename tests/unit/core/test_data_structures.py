"""
Unit tests for core data structures.

This module contains comprehensive tests for the core data structures
used throughout PyCALQ, including validation, serialization, and
backward compatibility.
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import FrozenInstanceError

from fvspectrum.core.data_structures import (
    ObservableType, FitModel, SamplingMode, PivotType, ChannelType,
    FitConfiguration, PlotConfiguration, MinimizerConfiguration,
    ChannelInfo, OperatorInfo, FitResult, SpectrumLevel,
    HadronNameParser, ValidationResult, create_channel_from_string,
    energy_sort_key, validate_fit_configuration
)


class TestEnumerations:
    """Test enumeration classes."""
    
    def test_observable_type_values(self):
        """Test ObservableType enum values."""
        assert ObservableType.ENERGY_SHIFT_LAB.value == "dElab"
        assert ObservableType.ENERGY_LAB.value == "elab"
        assert ObservableType.ENERGY_CM.value == "ecm"
        assert ObservableType.ENERGY_CM_NORMALIZED.value == "ecm_ref"
        assert ObservableType.AMPLITUDE.value == "amp"
        assert ObservableType.ENERGY_SHIFT_NORMALIZED.value == "dElab_ref"
    
    def test_fit_model_values(self):
        """Test FitModel enum values."""
        assert FitModel.SINGLE_EXPONENTIAL.value == "1-exp"
        assert FitModel.DOUBLE_EXPONENTIAL.value == "2-exp"
        assert FitModel.MULTI_EXPONENTIAL.value == "multi-exp"
        assert FitModel.CONSTANT.value == "constant"
        assert FitModel.COSH.value == "cosh"
        assert FitModel.SINH.value == "sinh"
    
    def test_sampling_mode_values(self):
        """Test SamplingMode enum values."""
        assert SamplingMode.JACKKNIFE.value == "Jackknife"
        assert SamplingMode.BOOTSTRAP.value == "Bootstrap"
    
    def test_pivot_type_values(self):
        """Test PivotType enum values."""
        assert PivotType.SINGLE_PIVOT.value == 0
        assert PivotType.ROLLING_PIVOT.value == 1
    
    def test_channel_type_values(self):
        """Test ChannelType enum values."""
        assert ChannelType.SINGLE_HADRON.value == "single_hadron"
        assert ChannelType.MULTI_HADRON.value == "multi_hadron"
        assert ChannelType.SCATTERING.value == "scattering"


class TestFitConfiguration:
    """Test FitConfiguration dataclass."""
    
    def test_valid_configuration(self):
        """Test creating a valid fit configuration."""
        config = FitConfiguration(
            model="1-exp",
            tmin=5,
            tmax=25,
            exclude_times=[10, 15],
            noise_cutoff=0.1
        )
        
        assert config.model == "1-exp"
        assert config.tmin == 5
        assert config.tmax == 25
        assert config.exclude_times == [10, 15]
        assert config.noise_cutoff == 0.1
        assert config.ratio is False

    
    def test_invalid_time_range(self):
        """Test validation of invalid time range."""
        with pytest.raises(ValueError, match="tmin .* must be less than tmax"):
            FitConfiguration(model="1-exp", tmin=25, tmax=5)
    
    def test_negative_noise_cutoff(self):
        """Test validation of negative noise cutoff."""
        with pytest.raises(ValueError, match="noise_cutoff must be non-negative"):
            FitConfiguration(model="1-exp", tmin=5, tmax=25, noise_cutoff=-0.1)
    
    def test_exclude_times_validation(self):
        """Test validation of exclude_times outside range."""
        with pytest.raises(ValueError, match="exclude_times .* are outside tmin-tmax range"):
            FitConfiguration(
                model="1-exp", 
                tmin=5, 
                tmax=25, 
                exclude_times=[1, 30]
            )
    
    def test_immutability(self):
        """Test that FitConfiguration is immutable."""
        config = FitConfiguration(model="1-exp", tmin=5, tmax=25)
        
        with pytest.raises(FrozenInstanceError):
            config.model = "2-exp"


class TestChannelInfo:
    """Test ChannelInfo dataclass."""
    
    def test_channel_creation(self):
        """Test creating a channel."""
        channel = ChannelInfo(
            isospin=1,
            strangeness=0,
            momentum_squared=0,
            irrep="A1",
            channel_type=ChannelType.SINGLE_HADRON
        )
        
        assert channel.isospin == 1
        assert channel.strangeness == 0
        assert channel.momentum_squared == 0
        assert channel.irrep == "A1"
        assert channel.channel_type == ChannelType.SINGLE_HADRON
    
    def test_backward_compatibility_psq(self):
        """Test backward compatibility psq property."""
        channel = ChannelInfo(
            isospin=1,
            strangeness=0,
            momentum_squared=4,
            irrep="A1"
        )
        
        assert channel.psq == 4
        assert channel.psq == channel.momentum_squared
    
    def test_string_representation(self):
        """Test string representation of channel."""
        channel = ChannelInfo(
            isospin=1,
            strangeness=-1,
            momentum_squared=3,
            irrep="T1"
        )
        
        expected = "I=1_S=-1_PSQ=3_T1"
        assert str(channel) == expected
    
    def test_channel_type_methods(self):
        """Test channel type checking methods."""
        single_hadron = ChannelInfo(
            isospin=0, strangeness=0, momentum_squared=0, irrep="A1",
            channel_type=ChannelType.SINGLE_HADRON
        )
        
        scattering = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1",
            channel_type=ChannelType.SCATTERING
        )
        
        assert single_hadron.is_single_hadron()
        assert not single_hadron.is_scattering()
        assert not scattering.is_single_hadron()
        assert scattering.is_scattering()


class TestOperatorInfo:
    """Test OperatorInfo dataclass."""
    
    def test_operator_creation(self):
        """Test creating an operator."""
        channel = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1"
        )
        
        operator = OperatorInfo(
            name="pi_p000",
            channel=channel,
            level=1,
            momentum=0,
            operator_type="interpolating"
        )
        
        assert operator.name == "pi_p000"
        assert operator.channel == channel
        assert operator.level == 1
        assert operator.momentum == 0
        assert operator.operator_type == "interpolating"
    
    def test_negative_level_validation(self):
        """Test validation of negative level."""
        channel = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1"
        )
        
        with pytest.raises(ValueError, match="level must be non-negative"):
            OperatorInfo(name="test", channel=channel, level=-1)
    
    def test_string_representation(self):
        """Test string representation of operator."""
        channel = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1"
        )
        
        operator = OperatorInfo(name="pi_p000", channel=channel, level=2)
        expected = f"pi_p000_{channel}_L2"
        assert str(operator) == expected


class TestFitResult:
    """Test FitResult dataclass."""
    
    def test_successful_fit_result(self):
        """Test creating a successful fit result."""
        result = FitResult(
            success=True,
            energy_value=0.5,
            energy_error=0.01,
            amplitude_value=1.0,
            amplitude_error=0.05,
            chisq_dof=1.2,
            quality=0.8,
            dof=10
        )
        
        assert result.success
        assert result.energy_value == 0.5
        assert result.energy_error == 0.01
        assert result.has_amplitude()
        assert result.is_good_quality()
    
    def test_failed_fit_result(self):
        """Test creating a failed fit result."""
        result = FitResult(
            success=False,
            energy_value=0.0,
            energy_error=0.0
        )
        
        assert not result.success
        assert not result.has_amplitude()
        assert not result.is_good_quality()
    
    def test_negative_error_validation(self):
        """Test validation of negative errors."""
        with pytest.raises(ValueError, match="energy_error must be non-negative"):
            FitResult(success=True, energy_value=0.5, energy_error=-0.01)
        
        with pytest.raises(ValueError, match="amplitude_error must be non-negative"):
            FitResult(
                success=True, 
                energy_value=0.5, 
                energy_error=0.01,
                amplitude_value=1.0,
                amplitude_error=-0.05
            )
    
    def test_quality_threshold(self):
        """Test quality threshold checking."""
        good_result = FitResult(
            success=True, energy_value=0.5, energy_error=0.01, quality=0.8
        )
        
        poor_result = FitResult(
            success=True, energy_value=0.5, energy_error=0.01, quality=0.01
        )
        
        assert good_result.is_good_quality(quality_threshold=0.05)
        assert not poor_result.is_good_quality(quality_threshold=0.05)


class TestSpectrumLevel:
    """Test SpectrumLevel dataclass."""
    
    def test_spectrum_level_creation(self):
        """Test creating a spectrum level."""
        channel = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1"
        )
        
        operator = OperatorInfo(name="pi_p000", channel=channel)
        
        fit_result = FitResult(
            success=True, energy_value=0.5, energy_error=0.01
        )
        
        level = SpectrumLevel(
            channel=channel,
            operator=operator,
            fit_result=fit_result,
            level_index=0,
            ecm_value=0.48,
            ecm_error=0.009
        )
        
        assert level.channel == channel
        assert level.operator == operator
        assert level.fit_result == fit_result
        assert level.level_index == 0
        assert level.ecm_value == 0.48
        assert level.ecm_error == 0.009
    
    def test_sort_key_successful_fit(self):
        """Test sort key for successful fit."""
        channel = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1"
        )
        operator = OperatorInfo(name="pi_p000", channel=channel)
        
        fit_result = FitResult(
            success=True, energy_value=0.5, energy_error=0.01
        )
        
        level = SpectrumLevel(
            channel=channel, operator=operator, fit_result=fit_result,
            ecm_value=0.48
        )
        
        assert level.sort_key() == 0.48  # Should use ecm_value
    
    def test_sort_key_failed_fit(self):
        """Test sort key for failed fit."""
        channel = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1"
        )
        operator = OperatorInfo(name="pi_p000", channel=channel)
        
        fit_result = FitResult(
            success=False, energy_value=0.0, energy_error=0.0
        )
        
        level = SpectrumLevel(
            channel=channel, operator=operator, fit_result=fit_result
        )
        
        assert level.sort_key() == float('inf')  # Failed fits sort to end
    
    def test_spectrum_level_sorting(self):
        """Test sorting of spectrum levels."""
        channel = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1"
        )
        operator = OperatorInfo(name="pi_p000", channel=channel)
        
        level1 = SpectrumLevel(
            channel=channel, operator=operator,
            fit_result=FitResult(success=True, energy_value=0.5, energy_error=0.01),
            ecm_value=0.5
        )
        
        level2 = SpectrumLevel(
            channel=channel, operator=operator,
            fit_result=FitResult(success=True, energy_value=0.3, energy_error=0.01),
            ecm_value=0.3
        )
        
        levels = [level1, level2]
        levels.sort()
        
        assert levels[0] == level2  # Lower energy first
        assert levels[1] == level1


class TestHadronNameParser:
    """Test HadronNameParser utility class."""
    
    def test_count_hadrons_single(self):
        """Test counting hadrons in single hadron operator."""
        assert HadronNameParser.count_hadrons("pi(0)") == 1
        assert HadronNameParser.count_hadrons("N_0") == 1
        assert HadronNameParser.count_hadrons("K[000]") == 1
    
    def test_count_hadrons_multiple(self):
        """Test counting hadrons in multi-hadron operator."""
        assert HadronNameParser.count_hadrons("pi(0)pi(0)") == 2
        assert HadronNameParser.count_hadrons("N(0)pi(0)") == 2
        assert HadronNameParser.count_hadrons("pi(0)pi(1)K(0)") == 3
    
    def test_extract_hadrons(self):
        """Test extracting hadron information."""
        hadrons = HadronNameParser.extract_hadrons("pi(000)N(100)")
        
        assert len(hadrons) == 2
        assert hadrons[0]['name'] == 'pi'
        assert hadrons[0]['momentum_info'] == '000'
        assert hadrons[1]['name'] == 'N'
        assert hadrons[1]['momentum_info'] == '100'
    
    def test_is_single_hadron(self):
        """Test single hadron detection."""
        assert HadronNameParser.is_single_hadron("pi(0)")
        assert not HadronNameParser.is_single_hadron("pi(0)pi(0)")
    
    def test_normalize_hadron_name(self):
        """Test hadron name normalization."""
        assert HadronNameParser.normalize_hadron_name("pion") == "pi"
        assert HadronNameParser.normalize_hadron_name("nucleon") == "N"
        assert HadronNameParser.normalize_hadron_name("kaon") == "K"
        assert HadronNameParser.normalize_hadron_name("unknown") == "unknown"


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.info) == 0
        assert not result.has_issues()
    
    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")
        
        assert not result.is_valid
        assert "Test error" in result.errors
        assert result.has_issues()
    
    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")
        
        assert result.is_valid  # Warnings don't invalidate
        assert "Test warning" in result.warnings
        assert result.has_issues()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_channel_from_string_valid(self):
        """Test creating channel from valid string."""
        channel = create_channel_from_string("I=1_S=0_PSQ=3_T1")
        
        assert channel.isospin == 1
        assert channel.strangeness == 0
        assert channel.momentum_squared == 3
        assert channel.irrep == "T1"
    
    def test_create_channel_from_string_invalid(self):
        """Test creating channel from invalid string."""
        with pytest.raises(ValueError, match="Invalid channel string format"):
            create_channel_from_string("invalid_format")
    
    def test_energy_sort_key_spectrum_level(self):
        """Test energy sort key with SpectrumLevel."""
        channel = ChannelInfo(
            isospin=1, strangeness=0, momentum_squared=0, irrep="A1"
        )
        operator = OperatorInfo(name="pi_p000", channel=channel)
        fit_result = FitResult(success=True, energy_value=0.5, energy_error=0.01)
        
        level = SpectrumLevel(
            channel=channel, operator=operator, fit_result=fit_result,
            ecm_value=0.48
        )
        
        assert energy_sort_key(level) == 0.48
    
    def test_energy_sort_key_dictionary(self):
        """Test energy sort key with dictionary."""
        data = {"ecm_value": 0.5, "energy_value": 0.6}
        assert energy_sort_key(data) == 0.5
        
        data = {"energy_value": 0.6}
        assert energy_sort_key(data) == 0.6
    
    def test_validate_fit_configuration_valid(self):
        """Test validation of valid fit configuration."""
        config = {
            "model": "1-exp",
            "tmin": 5,
            "tmax": 25
        }
        
        result = validate_fit_configuration(config)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_fit_configuration_missing_fields(self):
        """Test validation with missing required fields."""
        config = {"model": "1-exp"}  # Missing tmin, tmax
        
        result = validate_fit_configuration(config)
        assert not result.is_valid
        assert len(result.errors) == 2
    
    def test_validate_fit_configuration_invalid_time_range(self):
        """Test validation with invalid time range."""
        config = {
            "model": "1-exp",
            "tmin": 25,
            "tmax": 5
        }
        
        result = validate_fit_configuration(config)
        assert not result.is_valid
        assert any("tmin" in error and "tmax" in error for error in result.errors) 