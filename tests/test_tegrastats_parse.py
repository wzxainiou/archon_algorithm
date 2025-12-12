"""
Tests for tegrastats parser.
"""

import pytest
from jetson_bench.metrics.tegrastats import TegrastatsMonitor


def test_tegrastats_parse():
    """Test parsing tegrastats output lines."""
    monitor = TegrastatsMonitor()

    # Sample tegrastats line from Jetson Orin Nano
    sample_line = (
        "RAM 2505/7775MB (lfb 1429x4MB) SWAP 0/3887MB (cached 0MB) "
        "CPU [2%@2035,0%@2035,0%@2035,0%@2035,0%@2034,0%@2035] "
        "GR3D_FREQ 45% cpu@38.25C soc2@36.25C soc0@38.5C tj@38.25C "
        "soc1@37.25C gpu@36.5C POM_5V_IN 1234mW"
    )

    metrics = monitor._parse_line(sample_line)

    assert metrics is not None
    assert metrics.ram_used_mb == 2505
    assert metrics.ram_total_mb == 7775
    assert metrics.swap_used_mb == 0
    assert metrics.swap_total_mb == 3887
    assert metrics.gpu_util_percent == 45.0
    assert metrics.cpu_util_percent == pytest.approx(2.0 / 6)  # Average of CPU cores
    assert metrics.temp_gpu == 36.5
    assert metrics.power_mw == 1234


def test_tegrastats_parse_minimal():
    """Test parsing with minimal information."""
    monitor = TegrastatsMonitor()

    # Minimal line with just RAM
    sample_line = "RAM 1000/8000MB"

    metrics = monitor._parse_line(sample_line)

    assert metrics is not None
    assert metrics.ram_used_mb == 1000
    assert metrics.ram_total_mb == 8000
    assert metrics.gpu_util_percent is None
    assert metrics.cpu_util_percent is None


def test_tegrastats_parse_invalid():
    """Test parsing invalid/malformed lines."""
    monitor = TegrastatsMonitor()

    # Empty line
    metrics = monitor._parse_line("")
    assert metrics is not None  # Should return empty metrics, not fail

    # Garbage line
    metrics = monitor._parse_line("complete garbage with no valid data")
    assert metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
