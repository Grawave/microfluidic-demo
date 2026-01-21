"""
Configuration loader for simulation parameters.

Provides centralized access to simulation defaults and ensures
consistency across all scripts and UI components.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
_DEFAULTS_FILE = _CONFIG_DIR / "simulation_defaults.yaml"

# Cache the config to avoid repeated file reads
_cached_config: Dict[str, Any] = None


def load_defaults() -> Dict[str, Any]:
    """
    Load simulation defaults from YAML config.

    Returns:
        Dict with all default parameters
    """
    global _cached_config

    if _cached_config is None:
        with open(_DEFAULTS_FILE) as f:
            _cached_config = yaml.safe_load(f)

    return _cached_config


def get_grid_resolution(preset: str = None) -> Tuple[float, float]:
    """
    Get grid resolution in cells/mm.

    Args:
        preset: Optional preset name ("low", "medium", "high", "ultra")
                If None, uses default from config

    Returns:
        Tuple of (x_resolution, y_resolution) in cells/mm
    """
    config = load_defaults()

    if preset:
        res = config["grid"]["presets"][preset]
    else:
        res = config["grid"]["resolution_cells_per_mm"]

    return tuple(res)


def get_fluid_segments(preset: str = None) -> int:
    """
    Get number of segments per channel for fluid temperature tracking.

    Args:
        preset: Optional preset name ("coarse", "medium", "fine", "ultra", "extreme")
                If None, uses default from config

    Returns:
        Number of segments per channel
    """
    config = load_defaults()

    if preset:
        return config["fluid_visualization"]["presets"][preset]
    else:
        return config["fluid_visualization"]["segments_per_channel"]


def get_grid_size(chip_size_mm: Tuple[float, float], preset: str = None) -> Tuple[int, int]:
    """
    Calculate grid size in cells for given chip size and resolution.

    Args:
        chip_size_mm: Chip dimensions (length, width) in mm
        preset: Optional resolution preset

    Returns:
        Grid size (nx, ny) in cells
    """
    res_x, res_y = get_grid_resolution(preset)
    nx = int(chip_size_mm[0] * res_x)
    ny = int(chip_size_mm[1] * res_y)
    return (nx, ny)


def get_grid_resolution_text(chip_size_mm: Tuple[float, float] = (45, 45), preset: str = None) -> str:
    """
    Get formatted grid resolution text for UI display.

    Args:
        chip_size_mm: Chip dimensions in mm
        preset: Optional resolution preset

    Returns:
        Formatted string like "360×360 cells"
    """
    config = load_defaults()
    nx, ny = get_grid_size(chip_size_mm, preset)
    return config["ui_text"]["grid_resolution_format"].format(nx=nx, ny=ny)


def get_fluid_segments_text(preset: str = None) -> str:
    """
    Get formatted fluid segments text for UI display.

    Args:
        preset: Optional segments preset

    Returns:
        Formatted string like "200-Segment Fluid Temperature Tracking"
    """
    config = load_defaults()
    n = get_fluid_segments(preset)
    return config["ui_text"]["fluid_segments_format"].format(n=n)


def get_technical_subtitle(chip_size_mm: Tuple[float, float] = (45, 45)) -> str:
    """
    Get technical subtitle for UI with current grid and segment settings.

    Args:
        chip_size_mm: Chip dimensions in mm

    Returns:
        Formatted technical subtitle
    """
    config = load_defaults()
    grid_res = get_grid_resolution_text(chip_size_mm)
    segments = get_fluid_segments_text()
    return config["ui_text"]["technical_subtitle"].format(
        grid_res=grid_res,
        segments=segments
    )


def get_slideshow_config() -> Dict[str, Any]:
    """
    Get slideshow generation configuration.

    Returns:
        Dict with slideshow parameters
    """
    config = load_defaults()
    return config["slideshow"]


def get_solver_config() -> Dict[str, Any]:
    """
    Get solver configuration.

    Returns:
        Dict with solver parameters
    """
    config = load_defaults()
    return config["solver"]


if __name__ == "__main__":
    # Test config loading
    print("=== Simulation Configuration ===")
    print(f"Grid resolution: {get_grid_resolution()} cells/mm")
    print(f"Grid size (45×45mm chip): {get_grid_size((45, 45))}")
    print(f"Grid resolution text: {get_grid_resolution_text()}")
    print(f"Fluid segments: {get_fluid_segments()}")
    print(f"Fluid segments text: {get_fluid_segments_text()}")
    print(f"Technical subtitle: {get_technical_subtitle()}")
    print(f"\nSlideshow config: {get_slideshow_config()}")
