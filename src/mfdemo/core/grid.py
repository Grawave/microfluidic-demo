"""
Dimension-agnostic structured grid implementation.

This module provides the foundational Grid class that works seamlessly
for 2D and 3D simulations without code changes.
"""

from typing import Tuple, Dict, Optional, List
import numpy as np
from ..models.config import GridConfig


class StructuredGrid:
    """
    N-dimensional structured (Cartesian) grid for simulation.

    This class provides a dimension-agnostic interface for spatial grids.
    Works identically for 2D and 3D domains based on the ndim parameter.

    Attributes:
        ndim: Number of spatial dimensions (2 or 3)
        origin: Origin coordinates (ndim-length array)
        size: Domain size in each dimension (mm)
        resolution: Grid resolution (cells per mm) in each dimension
        shape: Grid shape (number of cells in each dimension)
        spacing: Cell spacing in each dimension (mm)
        n_cells: Total number of grid cells
        fields: Dictionary storing scalar/vector fields
    """

    def __init__(self, config: GridConfig):
        """
        Initialize grid from configuration.

        Args:
            config: GridConfig instance with validated parameters
        """
        self.ndim = config.ndim
        self.origin = np.array(config.origin_mm, dtype=np.float64)
        self.size = np.array(config.size_mm, dtype=np.float64)
        self.resolution = np.array(config.resolution_cells_per_mm, dtype=np.float64)

        # Derived properties
        self.shape = config.shape
        self.spacing = config.spacing
        self.n_cells = config.n_cells

        # Storage for fields (temperature, pressure, velocity components, etc.)
        self.fields: Dict[str, np.ndarray] = {}

        # Cache for coordinate arrays (lazy evaluation)
        self._coords_cache: Optional[Tuple[np.ndarray, ...]] = None

    def add_field(self, name: str, initial_value: float = 0.0, dtype=np.float64):
        """
        Add a scalar field to the grid.

        Args:
            name: Field name (e.g., 'temperature', 'pressure')
            initial_value: Initial value for all cells
            dtype: Data type for the field
        """
        self.fields[name] = np.full(self.shape, initial_value, dtype=dtype)

    def add_vector_field(self, name: str, initial_value: float = 0.0):
        """
        Add a vector field (one component per dimension).

        Args:
            name: Base field name (components will be name_x, name_y, name_z)
            initial_value: Initial value for all components
        """
        component_names = ['x', 'y', 'z'][:self.ndim]
        for comp in component_names:
            field_name = f"{name}_{comp}"
            self.add_field(field_name, initial_value)

    def get_field(self, name: str) -> np.ndarray:
        """
        Get a field by name.

        Args:
            name: Field name

        Returns:
            Field array (shape matches grid.shape)

        Raises:
            KeyError: If field doesn't exist
        """
        if name not in self.fields:
            raise KeyError(f"Field '{name}' not found in grid")
        return self.fields[name]

    def set_field(self, name: str, values: np.ndarray):
        """
        Set field values.

        Args:
            name: Field name
            values: New values (must match grid.shape)

        Raises:
            ValueError: If shape mismatch
        """
        if values.shape != self.shape:
            raise ValueError(
                f"Shape mismatch: field is {self.shape}, got {values.shape}"
            )
        self.fields[name] = values.astype(np.float64)

    def cell_centers(self) -> Tuple[np.ndarray, ...]:
        """
        Get coordinate arrays for cell centers (meshgrid style).

        Returns coordinate arrays suitable for broadcasting:
        - For 2D: (X, Y) where X.shape = (nx, 1), Y.shape = (1, ny)
        - For 3D: (X, Y, Z) where shapes are (nx, 1, 1), (1, ny, 1), (1, 1, nz)

        Returns:
            Tuple of ndim coordinate arrays
        """
        if self._coords_cache is not None:
            return self._coords_cache

        coords_1d = []
        for i in range(self.ndim):
            # 1D coordinate array for dimension i
            c = self.origin[i] + np.arange(self.shape[i]) * self.spacing[i]
            coords_1d.append(c)

        # Create broadcastable arrays using meshgrid with indexing='ij'
        coords = np.meshgrid(*coords_1d, indexing='ij')

        self._coords_cache = tuple(coords)
        return self._coords_cache

    def cell_volume(self) -> float:
        """
        Volume (or area in 2D) of a single cell.

        Returns:
            Cell volume in mm² (2D) or mm³ (3D)
        """
        return float(np.prod(self.spacing))

    def total_volume(self) -> float:
        """
        Total domain volume.

        Returns:
            Domain volume in mm² (2D) or mm³ (3D)
        """
        return float(np.prod(self.size))

    def get_bounds(self) -> np.ndarray:
        """
        Get domain bounds.

        Returns:
            Array of shape (ndim, 2) with [min, max] for each dimension
        """
        bounds = np.zeros((self.ndim, 2))
        bounds[:, 0] = self.origin
        bounds[:, 1] = self.origin + self.size
        return bounds

    def is_inside(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the domain.

        Args:
            point: Coordinates (length ndim)

        Returns:
            True if point is within domain bounds
        """
        point = np.asarray(point)
        if len(point) != self.ndim:
            raise ValueError(f"Point must have {self.ndim} dimensions")

        bounds = self.get_bounds()
        return np.all((point >= bounds[:, 0]) & (point <= bounds[:, 1]))

    def point_to_index(self, point: np.ndarray) -> Tuple[int, ...]:
        """
        Convert physical coordinates to grid indices.

        Args:
            point: Physical coordinates (length ndim)

        Returns:
            Grid indices (tuple of ints)

        Raises:
            ValueError: If point is outside domain
        """
        point = np.asarray(point)
        if not self.is_inside(point):
            raise ValueError(f"Point {point} is outside domain")

        # Convert to fractional indices
        frac_idx = (point - self.origin) / self.spacing

        # Round to nearest integer
        indices = tuple(int(np.clip(np.round(idx), 0, s - 1))
                       for idx, s in zip(frac_idx, self.shape))

        return indices

    def index_to_point(self, indices: Tuple[int, ...]) -> np.ndarray:
        """
        Convert grid indices to physical coordinates (cell center).

        Args:
            indices: Grid indices (tuple of ints)

        Returns:
            Physical coordinates (ndim array)
        """
        if len(indices) != self.ndim:
            raise ValueError(f"Indices must have {self.ndim} elements")

        indices_array = np.array(indices)
        return self.origin + indices_array * self.spacing

    def get_neighbor_indices(self, direction: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get index arrays for accessing neighbor cells in a direction.

        Useful for finite difference operations like gradients and Laplacians.

        Args:
            direction: Dimension index (0=x, 1=y, 2=z)

        Returns:
            (source_indices, neighbor_indices) - arrays of multi-indices

        Raises:
            ValueError: If direction is invalid
        """
        if direction < 0 or direction >= self.ndim:
            raise ValueError(
                f"Invalid direction {direction} for {self.ndim}D grid"
            )

        # Build slicing tuples
        # Source: all cells except last in direction
        # Neighbor: all cells except first in direction
        src_slices = [slice(None)] * self.ndim
        nb_slices = [slice(None)] * self.ndim

        src_slices[direction] = slice(None, -1)
        nb_slices[direction] = slice(1, None)

        return (tuple(src_slices), tuple(nb_slices))

    def compute_gradient(self, field: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute gradient of a field in specified direction using central differences.

        Args:
            field: Scalar field (shape must match grid.shape)
            direction: Direction index (0=x, 1=y, 2=z)

        Returns:
            Gradient field (same shape as input, boundaries set to 0)

        Raises:
            ValueError: If field shape doesn't match grid
        """
        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid {self.shape}")

        dx = self.spacing[direction]
        grad = np.zeros_like(field)

        # Build slicing for central difference: (f[i+1] - f[i-1]) / (2*dx)
        slices_center = [slice(1, -1)] * self.ndim
        slices_forward = [slice(1, -1)] * self.ndim
        slices_backward = [slice(1, -1)] * self.ndim

        slices_forward[direction] = slice(2, None)
        slices_backward[direction] = slice(None, -2)

        grad[tuple(slices_center)] = (
            (field[tuple(slices_forward)] - field[tuple(slices_backward)])
            / (2 * dx)
        )

        # Boundaries remain 0 (could use forward/backward differences instead)
        return grad

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian ∇²f using finite differences.

        Args:
            field: Scalar field (shape must match grid.shape)

        Returns:
            Laplacian field (same shape, boundaries set to 0)
        """
        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid {self.shape}")

        laplacian = np.zeros_like(field)

        # Sum second derivatives in all directions
        for d in range(self.ndim):
            dx2 = self.spacing[d] ** 2

            # Build slicing for second derivative: (f[i+1] - 2*f[i] + f[i-1]) / dx²
            slices_center = [slice(1, -1)] * self.ndim
            slices_forward = [slice(1, -1)] * self.ndim
            slices_backward = [slice(1, -1)] * self.ndim

            slices_forward[d] = slice(2, None)
            slices_backward[d] = slice(None, -2)

            laplacian[tuple(slices_center)] += (
                (field[tuple(slices_forward)]
                 - 2 * field[tuple(slices_center)]
                 + field[tuple(slices_backward)])
                / dx2
            )

        return laplacian

    def integrate_field(self, field: np.ndarray) -> float:
        """
        Integrate a field over the entire domain.

        Args:
            field: Scalar field to integrate

        Returns:
            Integral value (sum of field × cell_volume)
        """
        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid {self.shape}")

        return float(np.sum(field) * self.cell_volume())

    def get_statistics(self, field_name: str) -> Dict[str, float]:
        """
        Compute statistics for a field.

        Args:
            field_name: Name of field

        Returns:
            Dictionary with min, max, mean, std, integral
        """
        field = self.get_field(field_name)

        return {
            'min': float(np.min(field)),
            'max': float(np.max(field)),
            'mean': float(np.mean(field)),
            'std': float(np.std(field)),
            'integral': self.integrate_field(field)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StructuredGrid(ndim={self.ndim}, shape={self.shape}, "
            f"size={self.size} mm, spacing={self.spacing} mm)"
        )


def create_grid_from_chip_config(chip_config) -> StructuredGrid:
    """
    Convenience function to create grid from chip configuration.

    Args:
        chip_config: ChipConfig instance

    Returns:
        StructuredGrid instance
    """
    return StructuredGrid(chip_config.geometry)
