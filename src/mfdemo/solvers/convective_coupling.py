"""
Convective coupling between thermal and flow solvers.

Calculates heat transfer coefficients and fluid temperatures
for coupled thermal-fluidic simulation.
"""

from typing import Dict, Tuple
import numpy as np

from ..core.channel_network import ChannelNetwork
from ..core.grid import StructuredGrid
from ..models.config import nusselt_rectangular_channel, water_thermal_conductivity


class ConvectiveCoupling:
    """
    Handles convective heat transfer between chip and coolant.

    Calculates:
    - Heat transfer coefficients (h) from flow conditions
    - Fluid outlet temperatures from heat absorbed
    - Heat flux distribution along channels
    """

    def __init__(self, channel_network: ChannelNetwork,
                 fluid_properties: Dict[str, float]):
        """
        Initialize convective coupling calculator.

        Args:
            channel_network: ChannelNetwork instance with geometry
            fluid_properties: Dict with keys:
                - density_kg_per_m3: Fluid density
                - specific_heat_j_per_kg_k: Specific heat capacity
                - thermal_conductivity_w_per_m_k: Thermal conductivity
                - viscosity_pa_s: Dynamic viscosity
        """
        self.network = channel_network
        self.fluid = fluid_properties

        # Fluid properties
        self.rho = fluid_properties['density_kg_per_m3']
        self.cp = fluid_properties['specific_heat_j_per_kg_k']
        self.k_fluid = fluid_properties['thermal_conductivity_w_per_m_k']
        self.mu = fluid_properties['viscosity_pa_s']

    def update_fluid_thermal_conductivity(self, T_avg_celsius: float) -> None:
        """
        Update fluid thermal conductivity for temperature-dependent calculations.

        Args:
            T_avg_celsius: Average fluid temperature (°C)
        """
        self.k_fluid = water_thermal_conductivity(T_avg_celsius)

    def calculate_heat_transfer_coefficients(self,
                                            edge_flows: Dict[int, float]) -> Dict[int, float]:
        """
        Calculate convective heat transfer coefficient h for each channel.

        Uses Nusselt number correlation for laminar flow in rectangular channels.
        For fully developed laminar flow with uniform wall temperature: Nu ≈ 3.66 (constant)

        h = Nu * k_fluid / D_h

        Args:
            edge_flows: Dict[edge_id → flow_rate (m³/s)]

        Returns:
            Dict[edge_id → h (W/m²·K)]
        """
        h_coefficients = {}

        for edge_id, flow_rate in edge_flows.items():
            edge = self.network.edges[edge_id]

            # Hydraulic diameter (m)
            D_h = edge.hydraulic_diameter_m

            # Nusselt number for laminar flow in rectangular channels
            # Using Shah & London (1978) correlation with aspect ratio dependence
            # For uniform wall temperature boundary condition
            width = edge.width_um
            depth = edge.depth_um
            aspect_ratio = min(width, depth) / max(width, depth)
            Nu = nusselt_rectangular_channel(aspect_ratio)

            # Heat transfer coefficient (W/m²·K)
            h = Nu * self.k_fluid / D_h if D_h > 0 else 0

            h_coefficients[edge_id] = h

        return h_coefficients

    def calculate_fluid_temperatures(self,
                                     edge_flows: Dict[int, float],
                                     heat_flux_per_channel: Dict[int, float],
                                     inlet_temp_c: float = 25.0) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Calculate fluid outlet temperature for each channel.

        Energy balance:
        Q_removed = ṁ * cp * (T_out - T_in)
        T_out = T_in + Q_removed / (ṁ * cp)

        where ṁ = ρ * Q (mass flow rate)

        Args:
            edge_flows: Dict[edge_id → flow_rate (m³/s)]
            heat_flux_per_channel: Dict[edge_id → heat_removed (W)]
            inlet_temp_c: Inlet temperature (°C)

        Returns:
            Tuple of:
            - Dict[edge_id → outlet_temp (°C)]
            - Dict[edge_id → delta_temp (°C)]
        """
        outlet_temps = {}
        delta_temps = {}

        for edge_id, flow_rate in edge_flows.items():
            # Mass flow rate (kg/s)
            mass_flow = self.rho * flow_rate

            # Heat removed by this channel (W)
            Q_removed = heat_flux_per_channel.get(edge_id, 0.0)

            # Temperature rise (°C)
            if mass_flow > 1e-12:  # Avoid division by zero
                delta_T = Q_removed / (mass_flow * self.cp)
            else:
                delta_T = 0.0

            # Outlet temperature (°C)
            T_out = inlet_temp_c + delta_T

            outlet_temps[edge_id] = T_out
            delta_temps[edge_id] = delta_T

        return outlet_temps, delta_temps

    def estimate_heat_flux_per_channel(self,
                                       temperature_field: np.ndarray,
                                       grid: StructuredGrid,
                                       h_coefficients: Dict[int, float],
                                       inlet_temp_c: float = 25.0) -> Dict[int, float]:
        """
        Estimate heat removed by each channel from temperature field.

        This is a simplified estimation that integrates heat flux
        along the channel centerline.

        Q = ∫ h * (T_chip - T_fluid) * dA

        For simplicity, we approximate:
        - T_fluid ≈ T_inlet (actual varies along channel)
        - dA = perimeter * channel_length

        Args:
            temperature_field: 2D array of chip temperatures (°C)
            grid: Grid object with spatial information
            h_coefficients: Dict[edge_id → h (W/m²·K)]
            inlet_temp_c: Inlet temperature (°C)

        Returns:
            Dict[edge_id → heat_removed (W)]
        """
        heat_flux_per_channel = {}

        for edge_id, h in h_coefficients.items():
            edge = self.network.edges[edge_id]

            # Get channel geometry
            node_start = self.network.nodes[edge.node_start]
            node_end = self.network.nodes[edge.node_end]

            # Channel centerline points
            start_pos = node_start.position_mm
            end_pos = node_end.position_mm

            # Sample temperature along centerline
            # For 2D grid, interpolate at channel location
            n_samples = 10
            temperatures = []

            for i in range(n_samples):
                t = i / (n_samples - 1)
                pos = start_pos * (1 - t) + end_pos * t

                # Convert mm to grid indices
                # Assuming grid origin at (0,0)
                if grid.ndim == 2:
                    x_idx = int(pos[0] / grid.spacing[0])
                    y_idx = int(pos[1] / grid.spacing[1])

                    # Clamp to grid bounds
                    x_idx = np.clip(x_idx, 0, grid.shape[0] - 1)
                    y_idx = np.clip(y_idx, 0, grid.shape[1] - 1)

                    T_local = temperature_field[y_idx, x_idx]
                    temperatures.append(T_local)

            # Average temperature along channel
            T_avg = np.mean(temperatures)

            # Temperature difference (°C)
            delta_T = T_avg - inlet_temp_c

            # Heat transfer area (m²)
            # Perimeter * length
            # For rectangular channel: P = 2(w + h)
            w = edge.width_um * 1e-6  # μm → m
            h_channel = edge.depth_um * 1e-6  # μm → m
            perimeter = 2 * (w + h_channel)  # m

            length = edge.length_mm * 1e-3  # mm → m
            area = perimeter * length  # m²

            # Heat flux (W)
            Q = h * delta_T * area

            heat_flux_per_channel[edge_id] = max(0, Q)  # Non-negative

        return heat_flux_per_channel

    def calculate_fluid_temperatures_per_segment(self,
                                                edge_flows: Dict[int, float],
                                                temperature_field: np.ndarray,
                                                grid: StructuredGrid,
                                                h_coefficients: Dict[int, float],
                                                inlet_temp_c: float = 25.0,
                                                n_segments: int = 10) -> Dict[int, list]:
        """
        Calculate fluid temperature evolution along each channel as segments.

        This provides a temperature profile along the channel length, not just inlet→outlet.
        Each channel is divided into n_segments, and fluid temperature is tracked as it
        progresses through each segment, accounting for local heat transfer.

        Args:
            edge_flows: Dict[edge_id → flow_rate (m³/s)]
            temperature_field: 2D array of chip temperatures (°C)
            grid: Grid object with spatial information
            h_coefficients: Dict[edge_id → h (W/m²·K)]
            inlet_temp_c: Inlet temperature (°C)
            n_segments: Number of segments to divide each channel into

        Returns:
            Dict[edge_id → list of (position, temperature)] tuples
            where position is normalized [0, 1] along channel
        """
        fluid_profiles = {}

        for edge_id, flow_rate in edge_flows.items():
            edge = self.network.edges[edge_id]
            h = h_coefficients.get(edge_id, 0)

            if flow_rate < 1e-12 or h == 0:
                # No flow or heat transfer, return constant temp
                fluid_profiles[edge_id] = [(i / (n_segments - 1), inlet_temp_c)
                                           for i in range(n_segments)]
                continue

            # Get channel geometry
            node_start = self.network.nodes[edge.node_start]
            node_end = self.network.nodes[edge.node_end]
            start_pos = node_start.position_mm
            end_pos = node_end.position_mm

            # Channel dimensions
            w = edge.width_um * 1e-6  # μm → m
            h_channel = edge.depth_um * 1e-6  # μm → m
            perimeter = 2 * (w + h_channel)  # m

            length = edge.length_mm * 1e-3  # mm → m
            segment_length = length / n_segments  # m
            segment_area = perimeter * segment_length  # m²

            # Mass flow rate (kg/s)
            mass_flow = self.rho * flow_rate

            # Track fluid temperature along the channel
            T_fluid = inlet_temp_c
            profile = []

            for i in range(n_segments):
                # Position along channel [0, 1]
                t = i / (n_segments - 1) if n_segments > 1 else 0
                pos = start_pos * (1 - t) + end_pos * t

                # Sample chip temperature at this location
                if grid.ndim == 2:
                    x_idx = int(pos[0] / grid.spacing[0])
                    y_idx = int(pos[1] / grid.spacing[1])
                    x_idx = np.clip(x_idx, 0, grid.shape[0] - 1)
                    y_idx = np.clip(y_idx, 0, grid.shape[1] - 1)
                    T_chip = temperature_field[y_idx, x_idx]
                else:
                    T_chip = inlet_temp_c  # Fallback

                # Store current fluid temperature
                profile.append((t, T_fluid))

                # Calculate heat transfer for this segment
                delta_T = T_chip - T_fluid
                Q_segment = h * delta_T * segment_area

                # Calculate temperature rise for next segment
                if mass_flow > 0:
                    dT = Q_segment / (mass_flow * self.cp)
                    T_fluid += dT

            fluid_profiles[edge_id] = profile

        return fluid_profiles

    def get_fluid_temperature_statistics(self,
                                        outlet_temps: Dict[int, float],
                                        inlet_temp_c: float = 25.0) -> Dict[str, float]:
        """
        Calculate statistics about fluid temperatures.

        Args:
            outlet_temps: Dict[edge_id → outlet_temp (°C)]
            inlet_temp_c: Inlet temperature (°C)

        Returns:
            Dict with statistics:
            - inlet_temp_c: Inlet temperature
            - avg_outlet_temp_c: Average outlet temperature
            - max_outlet_temp_c: Maximum outlet temperature
            - min_outlet_temp_c: Minimum outlet temperature
            - avg_temp_rise_c: Average temperature rise
            - max_temp_rise_c: Maximum temperature rise
            - boiling_margin_c: Margin below boiling (100°C)
        """
        if not outlet_temps:
            return {
                'inlet_temp_c': inlet_temp_c,
                'avg_outlet_temp_c': inlet_temp_c,
                'max_outlet_temp_c': inlet_temp_c,
                'min_outlet_temp_c': inlet_temp_c,
                'avg_temp_rise_c': 0.0,
                'max_temp_rise_c': 0.0,
                'boiling_margin_c': 100.0 - inlet_temp_c
            }

        temps = list(outlet_temps.values())
        avg_outlet = np.mean(temps)
        max_outlet = np.max(temps)
        min_outlet = np.min(temps)

        avg_rise = avg_outlet - inlet_temp_c
        max_rise = max_outlet - inlet_temp_c

        boiling_point = 100.0  # °C at atmospheric pressure
        boiling_margin = boiling_point - max_outlet

        return {
            'inlet_temp_c': inlet_temp_c,
            'avg_outlet_temp_c': avg_outlet,
            'max_outlet_temp_c': max_outlet,
            'min_outlet_temp_c': min_outlet,
            'avg_temp_rise_c': avg_rise,
            'max_temp_rise_c': max_rise,
            'boiling_margin_c': boiling_margin
        }
