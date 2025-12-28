"""
Tests for dual mode functionality.

Tests configuration, mode selection, and model requirements for:
- rgb mode: RGB stream only (uses model0)
- thermal mode: Thermal stream only (uses model1)
- dual mode: Both streams simultaneously
"""

import pytest
from pathlib import Path

from jetson_bench.config import BenchConfig, ModelConfig


class TestModeConfig:
    """Test mode configuration in BenchConfig."""

    def test_default_mode_is_rgb(self):
        """Default mode should be 'rgb'."""
        config = BenchConfig()
        assert config.mode == "rgb"

    def test_valid_modes(self):
        """All three modes should be accepted."""
        for mode in ["rgb", "thermal", "dual"]:
            config = BenchConfig(mode=mode)
            assert config.mode == mode

    def test_dual_mode_source_fields(self):
        """Dual mode should accept rgb_source and thermal_source."""
        config = BenchConfig(
            mode="dual",
            rgb_source="/path/to/rgb.mp4",
            thermal_source="/path/to/thermal.mp4"
        )
        assert config.rgb_source == "/path/to/rgb.mp4"
        assert config.thermal_source == "/path/to/thermal.mp4"

    def test_single_mode_source_fields(self):
        """Single modes should accept source_type and source_path."""
        config = BenchConfig(
            mode="rgb",
            source_type="video",
            source_path="/path/to/video.mp4"
        )
        assert config.source_type == "video"
        assert config.source_path == "/path/to/video.mp4"


class TestModelSlots:
    """Test model slot configuration."""

    def test_exactly_two_models(self):
        """Config should have exactly 2 model slots."""
        config = BenchConfig()
        assert len(config.models) == 2

    def test_model_names(self):
        """Model names should be yolo11n_rgb and yolo11n_thermal."""
        config = BenchConfig()
        assert config.models[0].name == "yolo11n_rgb"
        assert config.models[1].name == "yolo11n_thermal"

    def test_cannot_have_one_model(self):
        """Should raise error if only one model is provided."""
        with pytest.raises(ValueError, match="exactly 2 models"):
            BenchConfig(models=[ModelConfig(name="only_one")])

    def test_cannot_have_three_models(self):
        """Should raise error if three models are provided."""
        with pytest.raises(ValueError, match="exactly 2 models"):
            BenchConfig(models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
                ModelConfig(name="model3"),
            ])


class TestModelRequirements:
    """Test model requirements per mode."""

    def test_rgb_mode_uses_model0(self):
        """RGB mode should use models[0] (yolo11n_rgb)."""
        config = BenchConfig(mode="rgb")
        assert config.models[0].name == "yolo11n_rgb"

    def test_thermal_mode_uses_model1(self):
        """Thermal mode should use models[1] (yolo11n_thermal)."""
        config = BenchConfig(mode="thermal")
        assert config.models[1].name == "yolo11n_thermal"

    def test_dual_mode_uses_both_models(self):
        """Dual mode should use both models."""
        config = BenchConfig(mode="dual")
        assert len(config.models) == 2
        assert config.models[0].name == "yolo11n_rgb"
        assert config.models[1].name == "yolo11n_thermal"


class TestModelValidation:
    """Test model validation behavior."""

    def test_model_without_weights_is_skipped(self):
        """Model without weights should be skipped after validation."""
        config = BenchConfig(mode="rgb")
        config.validate_models()

        # Both models have no weights, so both should be skipped
        assert config.models[0].skip_reason is not None
        assert config.models[1].skip_reason is not None

    def test_model_with_valid_weights(self):
        """Model with valid .pt file should not be skipped."""
        # Create a dummy .pt file path (using this test file as base)
        test_file = Path(__file__)
        dummy_pt = test_file.with_suffix(".pt")

        # Create temporary file for testing
        dummy_pt.touch()
        try:
            config = BenchConfig(mode="rgb")
            config.set_model_weights(0, str(dummy_pt))
            config.validate_models()

            # Model 0 should be valid (not skipped)
            assert config.models[0].skip_reason is None
            # Model 1 still has no weights
            assert config.models[1].skip_reason is not None
        finally:
            # Cleanup
            dummy_pt.unlink()

    def test_model_with_invalid_suffix(self):
        """Model with unsupported suffix should be skipped."""
        config = BenchConfig(mode="rgb")
        config.set_model_weights(0, str(Path(__file__)))  # .py file
        config.validate_models()

        assert config.models[0].skip_reason is not None
        assert "Unsupported weight format" in config.models[0].skip_reason


class TestActiveSkippedModels:
    """Test get_active_models and get_skipped_models."""

    def test_all_skipped_without_weights(self):
        """All models should be skipped if no weights provided."""
        config = BenchConfig()
        config.validate_models()

        assert len(config.get_active_models()) == 0
        assert len(config.get_skipped_models()) == 2

    def test_partial_active(self):
        """Only models with valid weights should be active."""
        # Create dummy .pt file
        test_file = Path(__file__)
        dummy_pt = test_file.with_suffix(".pt")
        dummy_pt.touch()

        try:
            config = BenchConfig()
            config.set_model_weights(0, str(dummy_pt))
            config.validate_models()

            assert len(config.get_active_models()) == 1
            assert len(config.get_skipped_models()) == 1
            assert config.get_active_models()[0].name == "yolo11n_rgb"
        finally:
            dummy_pt.unlink()


class TestSetModelWeights:
    """Test set_model_weights method."""

    def test_set_model0_weights(self):
        """Should be able to set model0 weights."""
        config = BenchConfig()
        config.set_model_weights(0, "/path/to/model.pt")
        assert config.models[0].weight_path == "/path/to/model.pt"

    def test_set_model1_weights(self):
        """Should be able to set model1 weights."""
        config = BenchConfig()
        config.set_model_weights(1, "/path/to/thermal_model.pt")
        assert config.models[1].weight_path == "/path/to/thermal_model.pt"

    def test_invalid_index_2(self):
        """Index 2 should raise ValueError (only 0-1 valid)."""
        config = BenchConfig()
        with pytest.raises(ValueError):
            config.set_model_weights(2, "/path/to/model.pt")

    def test_invalid_index_negative(self):
        """Negative index should raise ValueError."""
        config = BenchConfig()
        with pytest.raises(ValueError):
            config.set_model_weights(-1, "/path/to/model.pt")


class TestGPUMemoryLimit:
    """Test GPU memory limit configuration."""

    def test_default_limit_is_8gb(self):
        """Default GPU memory limit should be 8GB."""
        config = BenchConfig()
        assert config.gpu_mem_limit_gb == 8.0

    def test_limit_clamped_to_8gb(self):
        """GPU memory limit should be clamped to 8GB max."""
        config = BenchConfig(gpu_mem_limit_gb=16.0)
        assert config.gpu_mem_limit_gb == 8.0

    def test_can_set_lower_limit(self):
        """Should be able to set limit below 8GB."""
        config = BenchConfig(gpu_mem_limit_gb=4.0)
        assert config.gpu_mem_limit_gb == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
