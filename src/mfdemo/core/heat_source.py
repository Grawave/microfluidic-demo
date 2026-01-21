"""
Heat source implementation with multiple distribution types.

Supports dimension-agnostic heat source generation for thermal simulations.
"""

from typing import List, Optional
import numpy as np
from ..models.config import HeatSourceConfig, DistributionType
from .grid import StructuredGrid


class HeatSource:
    """
    Represents a heat source with spatial distribution.

    Supports multiple distribution types (Gaussian, uniform, point source)
    and works in arbitrary dimensions.
    """

    def __init__(self, config: HeatSourceConfig):
        """
        Initialize heat source from configuration.

        Args:
            config: HeatSourceConfig instance
        """
        self.config = config
        self.id = config.id
        self.position = np.array(config.position_mm, dtype=np.float64)
        self.intensity = config.intensity_w_per_mm2
        self.distribution_type = config.distribution_type
        self.spread = config.spread_mm
        self.extent = (
            np.array(config.extent_mm, dtype=np.float64)
            if config.extent_mm is not None
            else None
        )
        self.time_profile = config.time_profile

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.position)

    def apply_to_grid(self, grid: StructuredGrid,
                     time: Optional[float] = None) -> np.ndarray:
        """
        Generate heat source field on a grid.

        Args:
            grid: StructuredGrid instance
            time: Time value for time-dependent sources (optional)

        Returns:
            Heat source field (W/mm² for 2D, W/mm³ for 3D)
                with shape matching grid.shape

        Raises:
            ValueError: If grid dimensions don't match source
        """
        if grid.ndim != self.ndim:
            raise ValueError(
                f"Grid is {grid.ndim}D but heat source is {self.ndim}D"
            )

        # Get spatial distribution
        if self.distribution_type == DistributionType.GAUSSIAN:
            heat_field = self._gaussian_distribution(grid)
        elif self.distribution_type == DistributionType.UNIFORM_RECT:
            heat_field = self._uniform_rect_distribution(grid)
        elif self.distribution_type == DistributionType.POINT:
            heat_field = self._point_source_distribution(grid)
        else:
            raise NotImplementedError(
                f"Distribution type {self.distribution_type} not implemented"
            )

        # Apply temporal modulation if time is provided
        if time is not None:
            temporal_factor = self._get_temporal_factor(time)
            heat_field *= temporal_factor

        return heat_field

    def _gaussian_distribution(self, grid: StructuredGrid) -> np.ndarray:
        """
        Generate Gaussian heat distribution.

        Q(r) = Q₀ * exp(-r² / (2σ²))
        where r² = sum((x_i - x₀_i)² for all dimensions)

        The intensity parameter is interpreted as the peak value at the center.
        Total integrated power will be approximately: P = intensity × π × spread² (for 2D)

        Args:
            grid: StructuredGrid instance

        Returns:
            Heat distribution field (W/mm² or W/mm³)
        """
        coords = grid.cell_centers()

        # Compute squared distance in all dimensions
        r_squared = np.zeros(grid.shape, dtype=np.float64)
        for i in range(grid.ndim):
            r_squared += (coords[i] - self.position[i]) ** 2

        # Gaussian distribution with sigma = spread/2.5 for better concentration
        # This ensures 95% of power is within the spread radius
        sigma = self.spread / 2.5
        sigma_squared = sigma ** 2
        heat_field = self.intensity * np.exp(-r_squared / (2 * sigma_squared))

        # Truncate Gaussian tails below 0.1% of peak to prevent spreading heat everywhere
        # This dramatically reduces the number of non-zero cells
        threshold = 0.001 * self.intensity
        heat_field[heat_field < threshold] = 0.0

        return heat_field

    def _uniform_rect_distribution(self, grid: StructuredGrid) -> np.ndarray:
        """
        Generate uniform rectangular (or cuboid) heat distribution.

        Heat is uniformly distributed within a rectangular region.

        Args:
            grid: StructuredGrid instance

        Returns:
            Heat distribution field

        Raises:
            ValueError: If extent is not specified
        """
        if self.extent is None:
            raise ValueError("extent_mm must be specified for uniform_rect distribution")

        if len(self.extent) != grid.ndim:
            raise ValueError(
                f"Extent has {len(self.extent)} dimensions, expected {grid.ndim}"
            )

        coords = grid.cell_centers()
        heat_field = np.zeros(grid.shape, dtype=np.float64)

        # Create mask for rectangular region
        mask = np.ones(grid.shape, dtype=bool)
        for i in range(grid.ndim):
            half_extent = self.extent[i] / 2
            lower = self.position[i] - half_extent
            upper = self.position[i] + half_extent
            mask &= (coords[i] >= lower) & (coords[i] <= upper)

        # Uniform intensity within region
        heat_field[mask] = self.intensity

        return heat_field

    def _point_source_distribution(self, grid: StructuredGrid) -> np.ndarray:
        """
        Generate point source (concentrated in single cell).

        Args:
            grid: StructuredGrid instance

        Returns:
            Heat distribution field
        """
        heat_field = np.zeros(grid.shape, dtype=np.float64)

        # Find nearest grid cell
        try:
            idx = grid.point_to_index(self.position)
            # Convert intensity from W/mm² to W/mm³ by dividing by cell volume
            # (for point source, all power in one cell)
            heat_field[idx] = self.intensity / grid.cell_volume()
        except ValueError:
            # Point is outside grid - ignore this source
            pass

        return heat_field

    def _get_temporal_factor(self, time: float) -> float:
        """
        Get temporal modulation factor.

        Args:
            time: Time value (s)

        Returns:
            Multiplier for heat intensity (dimensionless)
        """
        if self.time_profile == "constant":
            return 1.0
        elif self.time_profile == "pulsed":
            # Example: sinusoidal pulsing with 1 Hz frequency
            return 0.5 * (1 + np.sin(2 * np.pi * time))
        elif self.time_profile == "ramped":
            # Example: linear ramp from 0 to 1 over 10 seconds
            return min(time / 10.0, 1.0)
        else:
            return 1.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HeatSource(id='{self.id}', pos={self.position}, "
            f"intensity={self.intensity} W/mm^{self.ndim}, "
            f"type={self.distribution_type.value})"
        )


class HeatSourceCollection:
    """
    Collection of multiple heat sources.

    Manages multiple heat sources and combines their contributions.
    """

    def __init__(self, sources: Optional[List[HeatSource]] = None):
        """
        Initialize collection.

        Args:
            sources: Optional list of HeatSource instances
        """
        self.sources: List[HeatSource] = sources if sources is not None else []

    def add_source(self, source: HeatSource):
        """Add a heat source to the collection."""
        self.sources.append(source)

    def add_from_config(self, config: HeatSourceConfig):
        """Add heat source from configuration."""
        self.add_source(HeatSource(config))

    def get_combined_field(self, grid: StructuredGrid,
                          time: Optional[float] = None) -> np.ndarray:
        """
        Get combined heat source field from all sources.

        Args:
            grid: StructuredGrid instance
            time: Time value for transient sources (optional)

        Returns:
            Combined heat source field (sum of all sources)
        """
        combined = np.zeros(grid.shape, dtype=np.float64)

        for source in self.sources:
            try:
                source_field = source.apply_to_grid(grid, time)
                combined += source_field
            except ValueError as e:
                # Skip sources that don't match grid dimensions
                print(f"Warning: Skipping source {source.id}: {e}")
                continue

        return combined

    def get_total_power(self, grid: StructuredGrid) -> float:
        """
        Compute total power from all sources.

        Args:
            grid: StructuredGrid instance

        Returns:
            Total power (W)
        """
        combined = self.get_combined_field(grid)
        return grid.integrate_field(combined)

    def get_source_by_id(self, source_id: str) -> Optional[HeatSource]:
        """
        Get a source by its ID.

        Args:
            source_id: Source identifier

        Returns:
            HeatSource instance or None if not found
        """
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def __len__(self) -> int:
        """Number of sources in collection."""
        return len(self.sources)

    def __iter__(self):
        """Iterate over sources."""
        return iter(self.sources)

    def __repr__(self) -> str:
        """String representation."""
        return f"HeatSourceCollection({len(self.sources)} sources)"


def create_heat_sources_from_chip_config(chip_config) -> HeatSourceCollection:
    """
    Create heat source collection from chip configuration.

    Args:
        chip_config: ChipConfig instance

    Returns:
        HeatSourceCollection with all sources from config
    """
    collection = HeatSourceCollection()

    for source_config in chip_config.thermal.heat_sources:
        collection.add_from_config(source_config)

    return collection
