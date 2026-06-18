"""Shared configuration and validation helpers for the QServer-backed MCP service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert common nested Python values into JSON-friendly objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def parse_range(value: str | None) -> tuple[float, float] | None:
    """Parse a comma-separated numeric range such as ``0,100``."""
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Ranges must contain exactly two comma-separated values.")
    return (float(parts[0]), float(parts[1]))


def validate_position_in_range(
    center: float | None,
    allowable_range: tuple[float, float] | None,
    axis_label: str,
    unit: str = "um",
) -> None:
    """Validate that a position lies within an allowable range."""
    if allowable_range is None:
        return
    if len(allowable_range) != 2:
        raise ValueError(
            f"The allowable range for the {axis_label} direction must contain exactly two values."
        )
    lower, upper = allowable_range
    if lower > upper:
        raise ValueError(
            f"The allowable range for the {axis_label} direction "
            f"({allowable_range}) has the lower bound greater than the upper bound."
        )
    if center is None:
        raise ValueError(
            f"The scan center position in the {axis_label} direction must be provided "
            "when an allowable range is set."
        )
    if not lower <= center <= upper:
        raise ValueError(
            f"The scan center position in the {axis_label} direction {center} {unit} is out "
            f"of the allowable range {allowable_range} {unit}."
        )


@dataclass(frozen=True)
class APSTwoIDDConfig:
    """Configuration for the APS 2-ID-D QueueServer-backed MCP service."""

    sample_name: str = "smp1"
    dwell_imaging: float = 0.05
    dwell_line_scan: float = 0.2
    xrf_on: bool = True
    preamp1_on: bool = False
    using_xrf_maps: bool = False
    xrf_elms: tuple[str, ...] = ("Cr",)
    xrf_roi_num: int = 16
    allowable_x_range: tuple[float, float] | None = None
    allowable_y_range: tuple[float, float] | None = None
    allowable_z_range: tuple[float, float] | None = None
    allowable_energy_range: tuple[float, float] | None = None
    plot_image_in_log_scale: bool = False
    show_colorbar_in_image: bool = False
    line_scan_return_gaussian_fit: bool = False
    scan_samy: bool = True
