"""
Tests for configuration module.
"""

import pytest
from pathlib import Path
from jetson_bench.config import BenchConfig, ModelConfig


def test_model_config_validation():
    """Test model configuration validation."""
    # Valid model (assuming this file exists)
    model = ModelConfig(name="test_model", weight_path=__file__)
    assert model.validate() is True
    assert model.skip_reason is None

    # Invalid - no path
    model = ModelConfig(name="test_model", weight_path=None)
    assert model.validate() is False
    assert "No weight path" in model.skip_reason

    # Invalid - file doesn't exist
    model = ModelConfig(name="test_model", weight_path="/nonexistent/model.pt")
    assert model.validate() is False
    assert "not found" in model.skip_reason

    # Invalid - wrong format
    model = ModelConfig(name="test_model", weight_path=__file__)
    # Change suffix to invalid
    model.weight_path = str(Path(__file__).with_suffix(".txt"))
    assert model.validate() is False
    assert "Unsupported weight format" in model.skip_reason


def test_bench_config_must_have_2_models():
    """Test that BenchConfig enforces exactly 2 models."""
    # Should work with default (2 models)
    config = BenchConfig()
    assert len(config.models) == 2

    # Should fail if we try to create with wrong number
    with pytest.raises(ValueError, match="exactly 2 models"):
        config = BenchConfig(models=[ModelConfig(name="only_one")])


def test_set_model_weights():
    """Test setting model weights."""
    config = BenchConfig()

    # Valid index
    config.set_model_weights(0, "/path/to/model.pt")
    assert config.models[0].weight_path == "/path/to/model.pt"

    config.set_model_weights(1, "/path/to/thermal_model.pt")
    assert config.models[1].weight_path == "/path/to/thermal_model.pt"

    # Invalid index (2 is now invalid since we only have 2 models)
    with pytest.raises(ValueError):
        config.set_model_weights(2, "/path/to/model.pt")

    with pytest.raises(ValueError):
        config.set_model_weights(-1, "/path/to/model.pt")


def test_get_active_and_skipped_models():
    """Test filtering active and skipped models."""
    config = BenchConfig()

    # Initially all have no weights, so all should be skipped after validation
    config.validate_models()
    assert len(config.get_active_models()) == 0
    assert len(config.get_skipped_models()) == 2

    # Set one model weight to a valid file
    config.set_model_weights(0, __file__)
    config.validate_models()
    assert len(config.get_active_models()) == 1
    assert len(config.get_skipped_models()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
