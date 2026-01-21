"""
Dimension-agnostic configuration models using Pydantic.

These models validate and load YAML configuration files for chips and channels.
Key design: All models work for ndim=2 or ndim=3 without code changes.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import numpy as np


# =============================================================================
# Enumerations
# =============================================================================

class DistributionType(str, Enum):
    """Heat source distribution types."""
    GAUSSIAN = "gaussian"
    UNIFORM_RECT = "uniform_rect"
    POINT = "point"


class BoundaryType(str, Enum):
    """Boundary condition types."""
    FIXED_TEMP = "fixed_temp"
    ADIABATIC = "adiabatic"
    CONVECTIVE = "convective"


class PatternType(str, Enum):
    """Channel pattern generation types."""
    STRAIGHT = "straight"
    BRANCHED_TREE = "branched_tree"
    FRACTAL = "fractal"
    SERPENTINE = "serpentine"
    VARIABLE_WIDTH = "variable_width"
    HIGH_DENSITY = "high_density"
    CUSTOM_HYBRID = "custom_hybrid"


class FlowRegime(str, Enum):
    """Flow regime assumptions."""
    LAMINAR = "laminar"
    TURBULENT = "turbulent"


class FluidType(str, Enum):
    """Coolant fluid types."""
    WATER = "water"
    DIELECTRIC = "dielectric"


# =============================================================================
# Grid Configuration
# =============================================================================

class GridConfig(BaseModel):
    """Dimension-agnostic grid configuration."""

    ndim: int = Field(2, ge=1, le=3, description="Number of spatial dimensions (2 or 3)")
    size_mm: List[float] = Field(..., description="Domain size [x, y] or [x, y, z] in mm")
    resolution_cells_per_mm: List[int] = Field(
        ...,
        description="Grid resolution in each dimension (cells per mm)"
    )
    origin_mm: Optional[List[float]] = Field(
        None,
        description="Origin coordinates (defaults to zeros)"
    )

    @field_validator('size_mm', 'resolution_cells_per_mm')
    @classmethod
    def check_dimension_match(cls, v, info):
        """Ensure list length matches ndim."""
        # Note: ndim may not be in values during validation, so check length is 2 or 3
        if len(v) not in [2, 3]:
            raise ValueError(f"Must be 2D or 3D, got length {len(v)}")
        return v

    @model_validator(mode='after')
    def validate_dimensions(self):
        """Validate all dimensional quantities match ndim."""
        if len(self.size_mm) != self.ndim:
            raise ValueError(
                f"size_mm length ({len(self.size_mm)}) must match ndim ({self.ndim})"
            )
        if len(self.resolution_cells_per_mm) != self.ndim:
            raise ValueError(
                f"resolution length ({len(self.resolution_cells_per_mm)}) must match ndim ({self.ndim})"
            )
        if self.origin_mm is not None and len(self.origin_mm) != self.ndim:
            raise ValueError(
                f"origin_mm length ({len(self.origin_mm)}) must match ndim ({self.ndim})"
            )

        # Set default origin if not provided
        if self.origin_mm is None:
            self.origin_mm = [0.0] * self.ndim

        return self

    @property
    def shape(self) -> tuple:
        """Grid shape (number of cells in each dimension)."""
        return tuple(
            int(s * r) for s, r in zip(self.size_mm, self.resolution_cells_per_mm)
        )

    @property
    def spacing(self) -> np.ndarray:
        """Cell spacing in each dimension."""
        return np.array(self.size_mm) / (np.array(self.shape) - 1)

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return int(np.prod(self.shape))


# =============================================================================
# Material and Thermal Properties
# =============================================================================

class MaterialProperties(BaseModel):
    """Material thermal properties."""

    name: str = Field(..., description="Material name (e.g., 'silicon')")
    thermal_conductivity_w_per_m_k: float = Field(
        ...,
        gt=0,
        description="Thermal conductivity (W/m·K)"
    )
    specific_heat_j_per_kg_k: float = Field(
        ...,
        gt=0,
        description="Specific heat capacity (J/kg·K)"
    )
    density_kg_per_m3: float = Field(..., gt=0, description="Density (kg/m³)")

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity α = k / (ρ * c) in m²/s."""
        return self.thermal_conductivity_w_per_m_k / (
            self.density_kg_per_m3 * self.specific_heat_j_per_kg_k
        )


class HeatSourceConfig(BaseModel):
    """Configuration for a single heat source (dimension-agnostic)."""

    id: str = Field(..., description="Unique identifier for this heat source")
    position_mm: List[float] = Field(
        ...,
        description="Center position [x, y] for 2D or [x, y, z] for 3D"
    )
    intensity_w_per_mm2: float = Field(
        ...,
        gt=0,
        description="Heat intensity (W/mm² for 2D, W/mm³ for 3D)"
    )
    distribution_type: DistributionType = Field(
        default=DistributionType.GAUSSIAN,
        description="Spatial distribution of heat"
    )
    spread_mm: float = Field(
        default=1.0,
        gt=0,
        description="Spread/width of distribution (mm)"
    )
    extent_mm: Optional[List[float]] = Field(
        None,
        description="Extent for uniform_rect distribution [width, height] or [width, height, depth]"
    )
    time_profile: Literal["constant", "pulsed", "ramped"] = Field(
        default="constant",
        description="Temporal variation (for transient sims)"
    )

    @field_validator('position_mm')
    @classmethod
    def check_position_dims(cls, v):
        """Validate position is 2D or 3D."""
        if len(v) not in [2, 3]:
            raise ValueError(f"Position must be 2D or 3D, got {len(v)} dimensions")
        return v


class BoundaryConditionConfig(BaseModel):
    """Boundary conditions (dimension-agnostic)."""

    type: BoundaryType = Field(
        default=BoundaryType.CONVECTIVE,
        description="Type of boundary condition"
    )
    ambient_temp_c: float = Field(
        default=25.0,
        description="Ambient temperature (°C) for convective BC"
    )
    heat_transfer_coefficient_w_per_m2_k: float = Field(
        default=10.0,
        gt=0,
        description="Heat transfer coefficient (W/m²·K) for convective BC"
    )
    fixed_temp_c: Optional[float] = Field(
        None,
        description="Fixed temperature (°C) for fixed_temp BC"
    )


class ThermalConfig(BaseModel):
    """Thermal configuration for a chip."""

    tdp_watts: float = Field(..., gt=0, description="Thermal Design Power (W)")
    base_temp_c: float = Field(
        default=85.0,
        description="Base junction temperature under load (°C)"
    )
    material: MaterialProperties
    heat_sources: List[HeatSourceConfig] = Field(
        default_factory=list,
        description="List of heat source regions"
    )
    boundaries: BoundaryConditionConfig = Field(
        default_factory=BoundaryConditionConfig
    )


# =============================================================================
# Chip Model Configuration
# =============================================================================

class ChipMetadata(BaseModel):
    """Metadata for a chip model."""

    name: str
    description: str
    manufacturer: Optional[str] = None
    category: Optional[str] = None  # e.g., "AI_GPU", "Server_CPU", "ARM"


class ChipConfig(BaseModel):
    """Complete chip model configuration (dimension-agnostic)."""

    metadata: ChipMetadata
    geometry: GridConfig
    thermal: ThermalConfig

    @model_validator(mode='after')
    def validate_heat_sources(self):
        """Ensure all heat sources match chip geometry dimensions."""
        for hs in self.thermal.heat_sources:
            if len(hs.position_mm) != self.geometry.ndim:
                raise ValueError(
                    f"Heat source '{hs.id}' position has {len(hs.position_mm)} dimensions, "
                    f"but chip geometry is {self.geometry.ndim}D"
                )
        return self

    # Convenience properties for easier access
    @property
    def name(self) -> str:
        """Chip name from metadata."""
        return self.metadata.name

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.geometry.ndim

    @property
    def size_mm(self) -> list:
        """Chip size in mm."""
        return self.geometry.size_mm

    @property
    def tdp_w(self) -> float:
        """Thermal Design Power in watts."""
        return self.thermal.tdp_watts

    @property
    def heat_sources(self) -> list:
        """List of heat sources."""
        return self.thermal.heat_sources


# =============================================================================
# Fluid and Flow Properties
# =============================================================================

class FluidProperties(BaseModel):
    """Coolant fluid properties."""

    density_kg_per_m3: float = Field(..., gt=0, description="Density (kg/m³)")
    viscosity_pa_s: float = Field(..., gt=0, description="Dynamic viscosity (Pa·s)")
    specific_heat_j_per_kg_k: float = Field(
        ...,
        gt=0,
        description="Specific heat (J/kg·K)"
    )
    thermal_conductivity_w_per_m_k: float = Field(
        ...,
        gt=0,
        description="Thermal conductivity (W/m·K)"
    )
    prandtl_number: float = Field(
        ...,
        gt=0,
        description="Prandtl number (dimensionless)"
    )

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity ν = μ/ρ in m²/s."""
        return self.viscosity_pa_s / self.density_kg_per_m3


class FlowOperatingConditions(BaseModel):
    """Operating conditions for flow."""

    inlet_temp_c: float = Field(default=20.0, description="Inlet temperature (°C)")
    flow_rate_ml_per_min: float = Field(
        ...,
        gt=0,
        description="Volumetric flow rate (ml/min)"
    )
    inlet_pressure_pa: float = Field(
        default=101325.0,
        gt=0,
        description="Inlet pressure (Pa, default=1 atm)"
    )


class FlowConfig(BaseModel):
    """Complete flow configuration."""

    fluid_type: FluidType = Field(default=FluidType.WATER)
    fluid: FluidProperties
    operating_conditions: FlowOperatingConditions
    regime: FlowRegime = Field(default=FlowRegime.LAMINAR)
    model: Literal["poiseuille_1d", "navier_stokes_2d", "navier_stokes_3d"] = Field(
        default="poiseuille_1d",
        description="Flow model complexity"
    )

    # Convenience properties for backward compatibility
    @property
    def inlet_pressure_pa(self) -> float:
        """Inlet pressure in Pa."""
        return self.operating_conditions.inlet_pressure_pa

    @property
    def outlet_pressure_pa(self) -> float:
        """Outlet pressure in Pa (atmospheric by default)."""
        return getattr(self.operating_conditions, 'outlet_pressure_pa', 101325.0)


# =============================================================================
# Channel Pattern and Geometry
# =============================================================================

class PatternConfig(BaseModel):
    """Pattern generation configuration (dimension-agnostic)."""

    ndim: int = Field(2, ge=2, le=3, description="Spatial dimensions")
    type: PatternType
    algorithm: str = Field(
        ...,
        description="Algorithm name for pattern generation (e.g., 'recursive_dichotomy')"
    )

    # Generation parameters
    levels: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Branching depth (for branched/fractal)"
    )
    branching_factor: Optional[int] = Field(
        None,
        ge=2,
        le=4,
        description="Children per parent node"
    )
    branching_angle_deg: Optional[float] = Field(
        None,
        ge=0,
        le=90,
        description="Angle between branches (degrees)"
    )
    length_decay: Optional[float] = Field(
        None,
        gt=0,
        lt=1,
        description="Child length = parent × decay"
    )
    width_decay: Optional[float] = Field(
        None,
        gt=0,
        lt=1,
        description="Child width = parent × decay"
    )

    # Inlet/outlet positions
    inlet_position_mm: List[float] = Field(
        ...,
        description="Inlet position [x, y] or [x, y, z]"
    )
    outlet_positions_mm: List[List[float]] = Field(
        ...,
        description="List of outlet positions (can have multiple outlets)"
    )

    # Optimization
    target_coverage_ratio: float = Field(
        default=0.35,
        ge=0,
        le=1,
        description="Target fraction of chip area/volume covered"
    )
    prefer_hotspots: bool = Field(
        default=True,
        description="Bias generation toward heat source locations"
    )

    @model_validator(mode='after')
    def validate_positions(self):
        """Validate inlet/outlet positions match ndim."""
        if len(self.inlet_position_mm) != self.ndim:
            raise ValueError(
                f"Inlet position has {len(self.inlet_position_mm)} dims, expected {self.ndim}"
            )
        for i, outlet in enumerate(self.outlet_positions_mm):
            if len(outlet) != self.ndim:
                raise ValueError(
                    f"Outlet {i} has {len(outlet)} dims, expected {self.ndim}"
                )
        return self


class ChannelGeometry(BaseModel):
    """Channel cross-section geometry."""

    width_um: float = Field(..., gt=0, description="Main channel width (μm)")
    depth_um: float = Field(..., gt=0, description="Channel depth (μm)")
    min_width_um: Optional[float] = Field(
        None,
        gt=0,
        description="Minimum width for tapered/branched channels (μm)"
    )
    wall_thickness_um: float = Field(
        default=10.0,
        gt=0,
        description="Silicon thickness between channels (μm)"
    )
    roughness_um: float = Field(
        default=0.5,
        ge=0,
        description="Surface roughness (μm, affects pressure drop)"
    )
    coating: Optional[str] = Field(
        None,
        description="Surface coating (e.g., 'hydrophobic')"
    )

    @model_validator(mode='after')
    def validate_widths(self):
        """Ensure min_width < width."""
        if self.min_width_um is not None and self.min_width_um >= self.width_um:
            raise ValueError("min_width_um must be less than width_um")
        return self


# =============================================================================
# Channel Variant Configuration
# =============================================================================

class ChannelMetadata(BaseModel):
    """Metadata for a channel variant."""

    name: str
    description: str
    inspiration: Optional[str] = None  # e.g., "Arterial blood vessel network"


class ExpectedPerformance(BaseModel):
    """Expected performance metrics (for comparison)."""

    max_pressure_drop_pa: Optional[float] = Field(None, gt=0)
    reynolds_number: Optional[float] = Field(None, gt=0)
    heat_removal_rate_w: Optional[float] = Field(None, gt=0)


class ChannelConfig(BaseModel):
    """Complete channel variant configuration (dimension-agnostic)."""

    metadata: ChannelMetadata
    pattern: PatternConfig
    geometry: ChannelGeometry
    flow: FlowConfig
    expected_performance: Optional[ExpectedPerformance] = None

    @model_validator(mode='after')
    def validate_consistency(self):
        """Cross-validate pattern and geometry."""
        # Pattern min width should match geometry min width if both specified
        if self.geometry.min_width_um is not None:
            if self.pattern.width_decay is not None:
                expected_min = self.geometry.width_um * (
                    self.pattern.width_decay ** (self.pattern.levels or 1)
                )
                if abs(expected_min - self.geometry.min_width_um) > 1.0:  # 1 μm tolerance
                    # This is just a warning, not an error
                    pass
        return self

    # Convenience properties for easier access
    @property
    def name(self) -> str:
        """Channel config name from metadata."""
        return self.metadata.name

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.pattern.ndim

    @property
    def parameters(self) -> dict:
        """Pattern parameters as dict (for backward compatibility)."""
        params = {
            'inlet_position_mm': self.pattern.inlet_position_mm,
            'outlet_positions_mm': self.pattern.outlet_positions_mm,
        }
        # Also provide singular outlet_position_mm for backward compatibility (first outlet)
        if self.pattern.outlet_positions_mm:
            params['outlet_position_mm'] = self.pattern.outlet_positions_mm[0]
        return params


# =============================================================================
# Utility Functions
# =============================================================================

def load_chip_config(filepath: str) -> ChipConfig:
    """Load and validate chip configuration from YAML file."""
    import yaml

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    return ChipConfig(**data)


def load_channel_config(filepath: str) -> ChannelConfig:
    """Load and validate channel configuration from YAML file."""
    import yaml

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    return ChannelConfig(**data)


# =============================================================================
# Preset Fluid Properties
# =============================================================================

FLUID_PRESETS = {
    "water": FluidProperties(
        density_kg_per_m3=1000.0,
        viscosity_pa_s=0.001,
        specific_heat_j_per_kg_k=4186.0,
        thermal_conductivity_w_per_m_k=0.6,
        prandtl_number=7.0
    ),
    "dielectric": FluidProperties(
        density_kg_per_m3=1200.0,
        viscosity_pa_s=0.002,
        specific_heat_j_per_kg_k=2000.0,
        thermal_conductivity_w_per_m_k=0.1,
        prandtl_number=20.0
    )
}

MATERIAL_PRESETS = {
    "silicon": MaterialProperties(
        name="silicon",
        thermal_conductivity_w_per_m_k=150.0,
        specific_heat_j_per_kg_k=700.0,
        density_kg_per_m3=2330.0
    )
}


# =============================================================================
# Temperature-Dependent Property Functions
# =============================================================================

def water_viscosity(T_celsius: float) -> float:
    """
    Temperature-dependent dynamic viscosity of water.

    Uses NIST correlation (valid for 0-100°C):
    μ(T) = A * 10^(B / (T + C))

    where T is in Celsius, A = 2.414e-5 Pa·s, B = 247.8, C = 140

    Reference: NIST Chemistry WebBook

    Args:
        T_celsius: Temperature in Celsius

    Returns:
        Dynamic viscosity in Pa·s
    """
    T = max(0.0, min(100.0, T_celsius))  # Clamp to valid range
    return 2.414e-5 * 10**(247.8 / (T + 140.0))


def water_thermal_conductivity(T_celsius: float) -> float:
    """
    Temperature-dependent thermal conductivity of water.

    Polynomial fit valid for 0-100°C:
    k(T) = 0.569 + 0.00179*T - 0.00000657*T²

    Reference: Engineering Toolbox

    Args:
        T_celsius: Temperature in Celsius

    Returns:
        Thermal conductivity in W/m·K
    """
    T = max(0.0, min(100.0, T_celsius))  # Clamp to valid range
    return 0.569 + 0.00179 * T - 0.00000657 * T**2


def silicon_thermal_conductivity(T_celsius: float) -> float:
    """
    Temperature-dependent thermal conductivity of silicon.

    Uses power law correlation:
    k(T) = k_ref * (T_ref / T)^n

    where k_ref = 150 W/m·K at T_ref = 300K, n = 1.3

    This captures the phonon scattering mechanism in silicon.

    Reference: Glassbrenner & Slack (1964), Phys. Rev.

    Args:
        T_celsius: Temperature in Celsius

    Returns:
        Thermal conductivity in W/m·K
    """
    T_kelvin = T_celsius + 273.15
    T_kelvin = max(200.0, min(500.0, T_kelvin))  # Clamp to valid range
    T_ref = 300.0  # Reference temperature (K)
    k_ref = 150.0  # Reference conductivity (W/m·K)
    n = 1.3  # Exponent for silicon
    return k_ref * (T_ref / T_kelvin) ** n


def nusselt_rectangular_channel(aspect_ratio: float) -> float:
    """
    Nusselt number for fully developed laminar flow in rectangular channel
    with uniform wall temperature boundary condition.

    Uses Shah & London (1978) correlation with polynomial fit:
    Nu = 7.541 * (1 - 2.610*α + 4.970*α² - 5.119*α³ + 2.702*α⁴ - 0.548*α⁵)

    where α = min(H,W) / max(H,W) is the aspect ratio (0 < α ≤ 1)

    Reference: Shah & London, "Laminar Flow Forced Convection in Ducts" (1978)

    Args:
        aspect_ratio: H/W ratio where H ≤ W (0 < aspect_ratio ≤ 1)

    Returns:
        Nusselt number (dimensionless)
    """
    alpha = max(0.01, min(1.0, aspect_ratio))  # Clamp to valid range

    # Polynomial coefficients for uniform wall temperature
    Nu = 7.541 * (1.0 - 2.610*alpha + 4.970*alpha**2 - 5.119*alpha**3
                  + 2.702*alpha**4 - 0.548*alpha**5)

    return Nu
