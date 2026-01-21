#!/usr/bin/env python3
"""
Find channel count and pressure drop for target outlet temperature.
Target: ~60°C max outlet temp (±5°C)
"""

import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mfdemo.core.grid import create_grid_from_chip_config
from mfdemo.core.heat_source import create_heat_sources_from_chip_config
from mfdemo.core.channel_network import ChannelNetwork
from mfdemo.solvers.thermal_solver import ThermalSolver
from mfdemo.solvers.microfluidic_solver import MicrofluidicSolver
from mfdemo.solvers.convective_coupling import ConvectiveCoupling
from mfdemo.models.config import ChipConfig, ChannelConfig


def create_channel_network(chip_size, n_channels, channel_width_um, channel_depth_um):
    """Create parallel channel network."""
    network = ChannelNetwork(ndim=2)
    chip_length, chip_width = chip_size

    spacing = chip_width / (n_channels + 1)
    y_start = 1.0
    y_end = chip_length - 1.0

    for i in range(n_channels):
        x = (i + 1) * spacing
        inlet_id = network.add_node(
            position=np.array([x, y_start]),
            node_type='inlet'
        )
        outlet_id = network.add_node(
            position=np.array([x, y_end]),
            node_type='outlet'
        )
        network.add_edge(
            node_start=inlet_id,
            node_end=outlet_id,
            width_um=channel_width_um,
            depth_um=channel_depth_um
        )

    return network


def run_simulation(n_channels, pressure_drop_kpa, channel_size_um=300):
    """Run simulation and return max outlet temperature."""
    # Load configs
    chip_path = Path(__file__).parent.parent / "configs" / "chips" / "intel_xeon_cpu.yaml"
    with open(chip_path) as f:
        chip_config = ChipConfig(**yaml.safe_load(f))

    channel_path = Path(__file__).parent.parent / "configs" / "channels" / "straight_parallel.yaml"
    with open(channel_path) as f:
        channel_config = ChannelConfig(**yaml.safe_load(f))

    chip_size = chip_config.geometry.size_mm

    # Create channel network
    network = create_channel_network(
        chip_size=chip_size,
        n_channels=n_channels,
        channel_width_um=channel_size_um,
        channel_depth_um=channel_size_um
    )

    # Create grid and heat sources
    grid = create_grid_from_chip_config(chip_config)
    heat_sources = create_heat_sources_from_chip_config(chip_config)

    # Create solvers
    thermal_solver = ThermalSolver(grid, chip_config)
    flow_solver = MicrofluidicSolver(network, channel_config)

    fluid_properties = {
        'density_kg_per_m3': channel_config.flow.fluid.density_kg_per_m3,
        'specific_heat_j_per_kg_k': channel_config.flow.fluid.specific_heat_j_per_kg_k,
        'thermal_conductivity_w_per_m_k': channel_config.flow.fluid.thermal_conductivity_w_per_m_k,
        'viscosity_pa_s': channel_config.flow.fluid.viscosity_pa_s,
    }
    convective_coupling = ConvectiveCoupling(network, fluid_properties)

    # Solve flow
    inlet_pressure = 101325.0 + pressure_drop_kpa * 1000
    outlet_pressure = 101325.0
    node_pressures, edge_flows = flow_solver.solve_flow(
        inlet_pressure_pa=inlet_pressure,
        outlet_pressure_pa=outlet_pressure
    )

    inlet_temp = 20.0

    # Iterative coupling
    temperature_field = thermal_solver.solve_steady_state(heat_sources, solver_method='direct')
    T_max_prev = temperature_field.max()

    for iteration in range(15):
        h_coefficients = convective_coupling.calculate_heat_transfer_coefficients(edge_flows)
        heat_flux_per_channel = convective_coupling.estimate_heat_flux_per_channel(
            temperature_field, grid, h_coefficients, inlet_temp
        )
        fluid_outlet_temps, _ = convective_coupling.calculate_fluid_temperatures(
            edge_flows, heat_flux_per_channel, inlet_temp
        )

        # Create convective sink field
        sink_field = np.zeros(grid.shape, dtype=np.float64)

        for edge_id, edge in network.edges.items():
            if edge_id not in h_coefficients:
                continue

            h = h_coefficients[edge_id]
            h_mm = h * 1e-6

            node_start = network.nodes[edge.node_start]
            node_end = network.nodes[edge.node_end]
            start_pos = node_start.position_mm
            end_pos = node_end.position_mm
            length_mm = edge.length_mm

            for i in range(20):
                t = (i + 0.5) / 20
                pos = start_pos + t * (end_pos - start_pos)

                try:
                    idx = grid.point_to_index(pos)
                    T_solid = temperature_field[idx]
                    T_outlet = fluid_outlet_temps.get(edge_id, inlet_temp + 30.0)
                    T_fluid = inlet_temp + t * (T_outlet - inlet_temp)
                    q_conv = h_mm * (T_solid - T_fluid) * 0.8

                    w_mm = edge.width_um / 1000
                    h_mm_channel = edge.depth_um / 1000
                    perimeter_mm = 2 * (w_mm + h_mm_channel)
                    q_per_length = q_conv * perimeter_mm
                    segment_length = length_mm / 20
                    q_total = q_per_length * segment_length

                    cell_vol = grid.cell_volume()
                    q_volumetric = q_total / cell_vol
                    sink_field[idx] -= q_volumetric
                except (ValueError, IndexError):
                    continue

        q_original = heat_sources.get_combined_field(grid)
        q_combined = q_original + sink_field
        q_si = q_combined * 1e9
        k = chip_config.thermal.material.thermal_conductivity_w_per_m_k
        rhs = -q_si / k

        laplacian = thermal_solver._build_laplacian()
        A, b = thermal_solver._apply_boundary_conditions(laplacian, rhs.ravel())

        from scipy.sparse.linalg import spsolve
        T_flat = spsolve(A, b)
        T_new = T_flat.reshape(grid.shape)
        T_new += chip_config.thermal.boundaries.ambient_temp_c

        temperature_field = 0.1 * T_new + 0.9 * temperature_field
        temperature_field = np.clip(temperature_field, 0.0, 500.0)

        T_max_new = temperature_field.max()
        relative_change = abs(T_max_new - T_max_prev) / T_max_prev

        if relative_change < 0.01:
            break

        T_max_prev = T_max_new

    # Final fluid temperatures
    h_coefficients = convective_coupling.calculate_heat_transfer_coefficients(edge_flows)
    heat_flux_per_channel = convective_coupling.estimate_heat_flux_per_channel(
        temperature_field, grid, h_coefficients, inlet_temp
    )
    fluid_outlet_temps, _ = convective_coupling.calculate_fluid_temperatures(
        edge_flows, heat_flux_per_channel, inlet_temp
    )

    # Calculate total flow rate
    total_flow_m3_s = sum(edge_flows.values())
    total_flow_ml_min = total_flow_m3_s * 1e6 * 60

    max_outlet = max(fluid_outlet_temps.values()) if fluid_outlet_temps else inlet_temp
    avg_outlet = np.mean(list(fluid_outlet_temps.values())) if fluid_outlet_temps else inlet_temp

    return {
        'n_channels': n_channels,
        'pressure_drop_kpa': pressure_drop_kpa,
        'channel_size_um': channel_size_um,
        'max_outlet_temp_c': max_outlet,
        'avg_outlet_temp_c': avg_outlet,
        'max_chip_temp_c': temperature_field.max(),
        'total_flow_ml_min': total_flow_ml_min,
    }


def main():
    print("=" * 70)
    print("Finding parameters for ~60°C max outlet temperature")
    print("=" * 70)

    target_temp = 60.0
    tolerance = 5.0

    # Current: 40 channels, 40 kPa, 300μm -> ~25°C outlet
    # Need higher outlet temp -> less cooling -> fewer channels or lower pressure

    results = []

    # Parameter sweep
    channel_counts = [8, 10, 12, 15, 20, 25, 30, 40]
    pressure_drops = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]

    print(f"\n{'Channels':>10} {'ΔP (kPa)':>10} {'Flow (ml/min)':>14} {'Max T_out':>10} {'Avg T_out':>10} {'Max T_chip':>10}")
    print("-" * 70)

    for n_ch in channel_counts:
        for dp in pressure_drops:
            try:
                result = run_simulation(n_ch, dp)
                results.append(result)

                marker = ""
                if abs(result['max_outlet_temp_c'] - target_temp) <= tolerance:
                    marker = " <-- TARGET"

                print(f"{n_ch:>10} {dp:>10.1f} {result['total_flow_ml_min']:>14.1f} "
                      f"{result['max_outlet_temp_c']:>10.1f} {result['avg_outlet_temp_c']:>10.1f} "
                      f"{result['max_chip_temp_c']:>10.1f}{marker}")
            except Exception as e:
                print(f"{n_ch:>10} {dp:>10.1f} ERROR: {str(e)[:30]}")

    # Find best matches
    print("\n" + "=" * 70)
    print("Best matches for ~60°C outlet temperature:")
    print("=" * 70)

    best = sorted(results, key=lambda r: abs(r['max_outlet_temp_c'] - target_temp))[:5]

    for i, r in enumerate(best, 1):
        diff = r['max_outlet_temp_c'] - target_temp
        print(f"\n{i}. {r['n_channels']} channels, {r['pressure_drop_kpa']:.0f} kPa, {r['channel_size_um']}μm:")
        print(f"   Max outlet: {r['max_outlet_temp_c']:.1f}°C ({diff:+.1f}°C from target)")
        print(f"   Avg outlet: {r['avg_outlet_temp_c']:.1f}°C")
        print(f"   Max chip:   {r['max_chip_temp_c']:.1f}°C")
        print(f"   Flow rate:  {r['total_flow_ml_min']:.1f} ml/min")


if __name__ == "__main__":
    main()
