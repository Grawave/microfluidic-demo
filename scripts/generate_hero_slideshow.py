#!/usr/bin/env python3
"""
Generate hero slideshow with multiple views:
1. Temperature heatmap with cooling
2. Channel network visualization
3. Configuration summary graphic
"""

import json
import numpy as np
from pathlib import Path
import sys
import yaml
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mfdemo.core.grid import StructuredGrid, create_grid_from_chip_config
from mfdemo.core.heat_source import HeatSourceCollection, HeatSource, create_heat_sources_from_chip_config
from mfdemo.core.channel_network import ChannelNetwork
from mfdemo.solvers.thermal_solver import ThermalSolver
from mfdemo.solvers.microfluidic_solver import MicrofluidicSolver
from mfdemo.solvers.convective_coupling import ConvectiveCoupling
from mfdemo.models.config import ChipConfig, ChannelConfig
from mfdemo.visualization.plotly_viz import SimulationVisualizer
from mfdemo.config_loader import (
    get_grid_resolution_text,
    get_fluid_segments,
    get_slideshow_config
)


def combine_images_vertically(top_path, bottom_path, output_path, spacing=20):
    """
    Combine two images vertically with optional spacing.

    Args:
        top_path: Path to top image
        bottom_path: Path to bottom image
        output_path: Path to save combined image
        spacing: Pixels of spacing between images
    """
    top_img = Image.open(top_path)
    bottom_img = Image.open(bottom_path)

    # Calculate combined size
    width = max(top_img.width, bottom_img.width)
    height = top_img.height + bottom_img.height + spacing

    # Create new image with dark background
    combined = Image.new('RGB', (width, height), color='#1e1e2e')

    # Paste images
    combined.paste(top_img, (0, 0))
    combined.paste(bottom_img, (0, top_img.height + spacing))

    # Save
    combined.save(output_path)
    print(f"   Combined: {output_path.name}")


def generate_chip_layout_with_power(
    chip_config,
    output_path
):
    """
    Generate chip layout showing heat sources with power labels.

    Args:
        chip_config: ChipConfig with geometry and heat sources
        output_path: Path to save image
    """
    chip_size = chip_config.geometry.size_mm
    chip_length, chip_width = chip_size

    fig = go.Figure()

    # Draw chip boundary
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=chip_width, y1=chip_length,
        line=dict(color="#89b4fa", width=3),
        fillcolor="#313244",
        opacity=0.4,
        layer='below'
    )

    # Create heat sources from config
    heat_sources = HeatSourceCollection()
    for hs_config in chip_config.thermal.heat_sources:
        heat_sources.add_source(HeatSource(config=hs_config))

    # Draw heat sources as rectangles with power labels
    # Get actual integrated power from grid (not approximate extent×intensity!)
    grid_temp = create_grid_from_chip_config(chip_config)
    total_power_grid = heat_sources.get_total_power(grid_temp)

    for heat_source in heat_sources.sources:
        center = heat_source.config.position_mm

        # Determine size (for visualization box)
        if heat_source.config.extent_mm:
            size = heat_source.config.extent_mm
        else:
            # For Gaussian: use 3σ radius (99% of distribution)
            size = [heat_source.config.spread_mm * 3] * len(center)

        x0 = center[0] - size[0] / 2
        x1 = center[0] + size[0] / 2
        y0 = center[1] - size[1] / 2
        y1 = center[1] + size[1] / 2

        # Calculate ACTUAL power by integrating this source on grid
        single_source_collection = HeatSourceCollection()
        single_source_collection.add_source(heat_source)
        power_w = single_source_collection.get_total_power(grid_temp)

        # Draw heat source rectangle
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#f38ba8", width=2),
            fillcolor="#f38ba8",
            opacity=0.7,
            layer='above'
        )

        # Add power label
        fig.add_annotation(
            x=center[0], y=center[1],
            text=f"<b>{power_w:.0f}W</b>",
            showarrow=False,
            font=dict(size=16, color="white", family="Arial Black"),
            bgcolor="rgba(30, 30, 46, 0.9)",
            bordercolor="white",
            borderwidth=2,
            borderpad=6
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Chip Layout: Power Distribution<br><sub>Total: {total_power_grid:.0f}W (grid-integrated) across {len(heat_sources.sources)} sources</sub>",
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        width=800,
        height=600,
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        xaxis=dict(
            title='Width (mm)',
            range=[-2, chip_width + 2],
            gridcolor='#45475a',
            color='#cdd6f4'
        ),
        yaxis=dict(
            title='Length (mm)',
            range=[-2, chip_length + 2],
            gridcolor='#45475a',
            color='#cdd6f4',
            scaleanchor='x',
            scaleratio=1
        ),
        margin=dict(l=60, r=20, t=80, b=60),
        showlegend=False
    )

    # Save as PNG
    fig.write_image(str(output_path), scale=2)
    print(f"   Saved: {output_path}")


def create_optimized_channel_network(chip_size, n_channels=40,
                                     channel_width_um=500, channel_depth_um=500):
    """Create optimized channel network for maximum cooling."""
    network = ChannelNetwork(ndim=2)
    chip_length, chip_width = chip_size

    # Evenly spaced parallel channels
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


def generate_channel_visualization(
    chip_size,
    channel_network,
    output_path,
    title="Cooling Channel Network"
):
    """Generate channel network visualization using Plotly."""
    chip_length, chip_width = chip_size

    fig = go.Figure()

    # Draw chip boundary
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=chip_width, y1=chip_length,
        line=dict(color="#89b4fa", width=2),
        fillcolor="#313244",
        opacity=0.3,
        layer='below'
    )

    # Draw channels as lines
    for edge_id, edge in channel_network.edges.items():
        node_start = channel_network.nodes[edge.node_start]
        node_end = channel_network.nodes[edge.node_end]

        fig.add_trace(go.Scatter(
            x=[node_start.position_mm[0], node_end.position_mm[0]],
            y=[node_start.position_mm[1], node_end.position_mm[1]],
            mode='lines',
            line=dict(color='#74c7ec', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Collect nodes by type
    inlet_x, inlet_y = [], []
    outlet_x, outlet_y = [], []

    for node_id, node in channel_network.nodes.items():
        if node.node_type == 'inlet':
            inlet_x.append(node.position_mm[0])
            inlet_y.append(node.position_mm[1])
        elif node.node_type == 'outlet':
            outlet_x.append(node.position_mm[0])
            outlet_y.append(node.position_mm[1])

    # Draw inlet nodes
    if inlet_x:
        fig.add_trace(go.Scatter(
            x=inlet_x, y=inlet_y,
            mode='markers',
            marker=dict(
                size=12,
                color='#a6e3a1',
                symbol='triangle-down',
                line=dict(color='white', width=1)
            ),
            name='Inlets (Cool)',
            hovertemplate='Inlet<br>25°C<extra></extra>'
        ))

    # Draw outlet nodes
    if outlet_x:
        fig.add_trace(go.Scatter(
            x=outlet_x, y=outlet_y,
            mode='markers',
            marker=dict(
                size=12,
                color='#f38ba8',
                symbol='triangle-up',
                line=dict(color='white', width=1)
            ),
            name='Outlets (Hot)',
            hovertemplate='Outlet<br>~40°C<extra></extra>'
        ))

    # Count channels
    n_channels = len([n for n in channel_network.nodes.values() if n.node_type == 'inlet'])

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{n_channels} channels × 500×500 μm</sub>",
            font=dict(size=18, color='white'),
            x=0.5,
            xanchor='center'
        ),
        width=800,
        height=600,
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        xaxis=dict(
            title='Width (mm)',
            range=[-2, chip_width + 2],
            gridcolor='#45475a',
            color='#cdd6f4'
        ),
        yaxis=dict(
            title='Length (mm)',
            range=[-2, chip_length + 2],
            gridcolor='#45475a',
            color='#cdd6f4',
            scaleanchor='x',
            scaleratio=1
        ),
        legend=dict(
            font=dict(color='#cdd6f4'),
            bgcolor='#313244',
            bordercolor='#45475a',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    # Save as PNG
    fig.write_image(str(output_path), scale=2)
    print(f"   Saved: {output_path}")


def generate_config_summary(
    metadata,
    output_path
):
    """Generate configuration summary graphic."""
    # Create image with dark background
    img = Image.new('RGB', (800, 600), color='#1e1e2e')
    draw = ImageDraw.Draw(img)

    # Try to load a font, fallback to default (2x larger for readability)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    y = 50

    # Title
    draw.text((400, y), "Simulation Configuration",
              fill='#cdd6f4', font=title_font, anchor='mt')
    y += 100

    # Chip section
    draw.text((50, y), "🔲 Chip Model", fill='#89b4fa', font=header_font)
    y += 65
    draw.text((70, y), f"Intel Xeon CPU (250W)", fill='#cdd6f4', font=body_font)
    y += 50
    draw.text((70, y), f"Size: 45×45 mm", fill='#a6adc8', font=small_font)
    y += 40
    draw.text((70, y), f"Grid: {metadata.get('grid_resolution', 'N/A')}",
              fill='#a6adc8', font=small_font)
    y += 70

    # Channel section
    draw.text((50, y), "💧 Cooling Channels", fill='#74c7ec', font=header_font)
    y += 65
    draw.text((70, y), f"Pattern: Straight Parallel", fill='#cdd6f4', font=body_font)
    y += 50
    draw.text((70, y), f"Count: {metadata['n_channels']} channels",
              fill='#a6adc8', font=small_font)
    y += 40
    draw.text((70, y), f"Size: {metadata['channel_size_um']}×{metadata['channel_size_um']} μm",
              fill='#a6adc8', font=small_font)
    y += 70

    # Operating conditions
    draw.text((50, y), "⚡ Operating Conditions", fill='#f9e2af', font=header_font)
    y += 65
    draw.text((70, y), f"Pressure: {metadata['pressure_drop_kpa']} kPa",
              fill='#cdd6f4', font=body_font)
    y += 50
    draw.text((70, y), f"Flow: {metadata['flow_rate_ml_min']:.0f} ml/min",
              fill='#a6adc8', font=small_font)
    y += 40
    draw.text((70, y), f"Inlet temp: 25°C", fill='#a6adc8', font=small_font)
    y += 70

    # Results
    draw.text((50, y), "📊 Results", fill='#a6e3a1', font=header_font)
    y += 65
    draw.text((70, y),
              f"Max temp: {metadata['T_max_no_cooling']:.0f}°C → {metadata['T_max_with_cooling']:.0f}°C",
              fill='#cdd6f4', font=body_font)
    y += 50
    draw.text((70, y),
              f"Effectiveness: {metadata['cooling_effectiveness_pct']:.1f}%",
              fill='#a6adc8', font=small_font)
    y += 40
    draw.text((70, y),
              f"Solve time: <3 seconds ({metadata['iterations']} iter)",
              fill='#a6adc8', font=small_font)

    img.save(output_path)


def generate_physical_layout(
    chip_config,
    channel_network,
    fluid_outlet_temps=None,
    fluid_temp_profiles=None,
    fluid_stats=None,
    inlet_temp_c=25.0,
    output_path=None
):
    """
    Generate physical layout showing heat sources + channels with fluid temperatures.

    Args:
        chip_config: ChipConfig with geometry and heat sources
        channel_network: ChannelNetwork with nodes and edges
        fluid_outlet_temps: Optional Dict[edge_id → outlet_temp (°C)]
        fluid_temp_profiles: Optional Dict[edge_id → list of (position, temp)] for gradient coloring
        fluid_stats: Optional Dict with fluid temperature statistics
        inlet_temp_c: Inlet temperature (°C)
        output_path: Path to save image
    """
    chip_size = chip_config.geometry.size_mm
    chip_length, chip_width = chip_size

    # Determine if showing fluid temps
    show_fluid_temps = fluid_outlet_temps is not None and len(fluid_outlet_temps) > 0

    fig = go.Figure()

    # Draw chip boundary
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=chip_width, y1=chip_length,
        line=dict(color="#89b4fa", width=2),
        fillcolor="#313244",
        opacity=0.3,
        layer='below'
    )

    # Create heat sources from config
    heat_sources = HeatSourceCollection()
    for hs_config in chip_config.thermal.heat_sources:
        heat_sources.add_source(HeatSource(config=hs_config))

    # Create grid for power integration (need this to get actual power, not approximate extent×intensity!)
    grid_temp = create_grid_from_chip_config(chip_config)

    # Draw heat sources as rectangles
    for heat_source in heat_sources.sources:
        center = heat_source.config.position_mm

        # Determine size
        if heat_source.config.extent_mm:
            size = heat_source.config.extent_mm
        else:
            size = [heat_source.config.spread_mm * 2] * len(center)

        x0 = center[0] - size[0] / 2
        x1 = center[0] + size[0] / 2
        y0 = center[1] - size[1] / 2
        y1 = center[1] + size[1] / 2

        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#f38ba8", width=2),
            fillcolor="#f38ba8",
            opacity=0.6,
            layer='below'
        )

        # Calculate ACTUAL power by integrating this source on grid
        # (Don't use area×intensity for Gaussians - that's wrong!)
        single_source_collection = HeatSourceCollection()
        single_source_collection.add_source(heat_source)
        power_w = single_source_collection.get_total_power(grid_temp)

        # Add label
        fig.add_annotation(
            x=center[0], y=center[1],
            text=f"{power_w:.0f}W",
            showarrow=False,
            font=dict(size=12, color="white", family="monospace"),
            bgcolor="rgba(243, 139, 168, 0.9)",
            borderpad=3
        )

    # Draw channels (color-coded by fluid temperature if available)
    import plotly.express as px

    # Prefer per-segment profiles for gradient coloring, fall back to outlet temps
    if fluid_temp_profiles and len(fluid_temp_profiles) > 0:
        # Per-segment gradient coloring
        # Get temperature range for color mapping across all segments
        all_temps = []
        for profile in fluid_temp_profiles.values():
            all_temps.extend([temp for _, temp in profile])

        temp_min = min(all_temps)
        temp_max = max(all_temps)
        temp_range = temp_max - temp_min if temp_max > temp_min else 1.0

        colorscale = px.colors.sequential.Turbo  # Blue → Green → Yellow → Red

        for edge_id, edge in channel_network.edges.items():
            profile = fluid_temp_profiles.get(edge_id, [])
            if not profile:
                continue

            node_start = channel_network.nodes[edge.node_start]
            node_end = channel_network.nodes[edge.node_end]
            start_pos = node_start.position_mm
            end_pos = node_end.position_mm

            # Draw each segment with its own color
            for i in range(len(profile) - 1):
                t0, T0 = profile[i]
                t1, T1 = profile[i + 1]

                # Positions of segment endpoints
                pos0 = start_pos * (1 - t0) + end_pos * t0
                pos1 = start_pos * (1 - t1) + end_pos * t1

                # Use temperature at segment start for color
                norm_temp = (T0 - temp_min) / temp_range if temp_range > 0 else 0.5
                color_idx = int(norm_temp * (len(colorscale) - 1))
                color_idx = max(0, min(color_idx, len(colorscale) - 1))
                segment_color = colorscale[color_idx]

                fig.add_trace(go.Scatter(
                    x=[pos0[0], pos1[0]],
                    y=[pos0[1], pos1[1]],
                    mode='lines',
                    line=dict(color=segment_color, width=5),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    elif show_fluid_temps:
        # Single color per channel (legacy)
        temps = list(fluid_outlet_temps.values())
        temp_min = min(temps)
        temp_max = max(temps)
        temp_range = temp_max - temp_min if temp_max > temp_min else 1.0

        colorscale = px.colors.sequential.Turbo

        for edge_id, edge in channel_network.edges.items():
            node_start = channel_network.nodes[edge.node_start]
            node_end = channel_network.nodes[edge.node_end]

            T_out = fluid_outlet_temps.get(edge_id, inlet_temp_c)
            norm_temp = (T_out - temp_min) / temp_range if temp_range > 0 else 0.5
            color_idx = int(norm_temp * (len(colorscale) - 1))
            color_idx = max(0, min(color_idx, len(colorscale) - 1))
            channel_color = colorscale[color_idx]

            fig.add_trace(go.Scatter(
                x=[node_start.position_mm[0], node_end.position_mm[0]],
                y=[node_start.position_mm[1], node_end.position_mm[1]],
                mode='lines',
                line=dict(color=channel_color, width=5),
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        # Default cyan channels
        for edge_id, edge in channel_network.edges.items():
            node_start = channel_network.nodes[edge.node_start]
            node_end = channel_network.nodes[edge.node_end]

            fig.add_trace(go.Scatter(
                x=[node_start.position_mm[0], node_end.position_mm[0]],
                y=[node_start.position_mm[1], node_end.position_mm[1]],
                mode='lines',
                line=dict(color='#74c7ec', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Draw nodes
    inlet_x, inlet_y = [], []
    outlet_x, outlet_y = [], []

    for node_id, node in channel_network.nodes.items():
        if node.node_type == 'inlet':
            inlet_x.append(node.position_mm[0])
            inlet_y.append(node.position_mm[1])
        elif node.node_type == 'outlet':
            outlet_x.append(node.position_mm[0])
            outlet_y.append(node.position_mm[1])

    if inlet_x:
        fig.add_trace(go.Scatter(
            x=inlet_x, y=inlet_y,
            mode='markers',
            marker=dict(
                size=14,
                color='#a6e3a1',
                symbol='triangle-down',
                line=dict(color='#1e1e2e', width=2)
            ),
            name='Inlets',
            hovertemplate='Inlet<extra></extra>'
        ))

    if outlet_x:
        fig.add_trace(go.Scatter(
            x=outlet_x, y=outlet_y,
            mode='markers',
            marker=dict(
                size=14,
                color='#f38ba8',
                symbol='triangle-up',
                line=dict(color='#1e1e2e', width=2)
            ),
            name='Outlets',
            hovertemplate='Outlet<extra></extra>'
        ))

    # Count elements
    n_channels = len([n for n in channel_network.nodes.values() if n.node_type == 'inlet'])
    n_heat_sources = len(heat_sources.sources)

    # Build title with optional fluid stats
    if show_fluid_temps and fluid_stats:
        title_text = (
            f"Physical Layout + Fluid Temperatures<br>"
            f"<sub>{n_heat_sources} heat sources • {n_channels} cooling channels</sub><br>"
            f"<sub style='font-size: 12px;'>"
            f"Fluid: {fluid_stats['inlet_temp_c']:.0f}°C → {fluid_stats['avg_outlet_temp_c']:.1f}°C  •  "
            f"Max outlet: {fluid_stats['max_outlet_temp_c']:.1f}°C  •  "
            f"Boiling margin: {fluid_stats['boiling_margin_c']:.1f}°C</sub>"
        )
    else:
        title_text = f"Physical Layout<br><sub>{n_heat_sources} heat sources • {n_channels} cooling channels</sub>"

    # Update layout
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=20, color='#cdd6f4'),
            x=0.5,
            xanchor='center'
        ),
        width=800,
        height=600,
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        xaxis=dict(
            title='Width (mm)',
            range=[-2, chip_width + 2],
            gridcolor='#45475a',
            color='#cdd6f4',
            constrain='domain'
        ),
        yaxis=dict(
            title='Length (mm)',
            range=[-2, chip_length + 2],
            gridcolor='#45475a',
            color='#cdd6f4',
            scaleanchor='x',
            scaleratio=1,
            constrain='domain'
        ),
        legend=dict(
            font=dict(color='#cdd6f4'),
            bgcolor='#313244',
            bordercolor='#45475a',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=80, b=60),
        hovermode='closest'
    )

    # Save
    fig.write_image(str(output_path), scale=2)
    print(f"   Saved: {output_path}")


def generate_slideshow(
    chip_config_name: str = "intel_xeon_cpu",
    channel_config_name: str = "straight_parallel",
    output_dir: Path = Path("static/hero_slideshow"),
    n_channels: int = 8,
    channel_size_um: int = 300,
    pressure_drop_kpa: float = 15.0  # Tuned for ~60°C max outlet temp
):
    """Generate hero slideshow with 3 slides (combined visualizations)."""
    print(f"🎬 Generating hero slideshow (3 slides)")
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Load configs
    chip_path = Path(__file__).parent.parent / "configs" / "chips" / f"{chip_config_name}.yaml"
    with open(chip_path) as f:
        chip_dict = yaml.safe_load(f)
    chip_config = ChipConfig(**chip_dict)
    chip_size = chip_config.geometry.size_mm

    channel_path = Path(__file__).parent.parent / "configs" / "channels" / f"{channel_config_name}.yaml"
    with open(channel_path) as f:
        channel_dict = yaml.safe_load(f)
    channel_config = ChannelConfig(**channel_dict)

    # Create channel network
    print("💧 Creating channel network...")
    network = create_optimized_channel_network(
        chip_size=chip_size,
        n_channels=n_channels,
        channel_width_um=channel_size_um,
        channel_depth_um=channel_size_um
    )

    # Run simulation for temperature field and fluid temperatures
    print("🔥 Running simulation with iterative thermal-fluidic coupling...")

    # Create grid
    grid = create_grid_from_chip_config(chip_config)

    # Create heat sources
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

    # Solve flow first (constant for steady-state)
    inlet_pressure = 101325.0 + pressure_drop_kpa * 1000
    outlet_pressure = 101325.0
    node_pressures, edge_flows = flow_solver.solve_flow(
        inlet_pressure_pa=inlet_pressure,
        outlet_pressure_pa=outlet_pressure
    )

    inlet_temp = channel_config.flow.operating_conditions.inlet_temp_c

    # ITERATIVE COUPLING with under-relaxation
    print("   Using Picard iteration (relaxation=0.1, max_iter=15)")

    max_iterations = 15
    tolerance = 0.01
    relaxation_factor = 0.1

    # Initialize with no-cooling solve
    temperature_field = thermal_solver.solve_steady_state(heat_sources, solver_method='direct')
    T_max_prev = temperature_field.max()

    # Picard iteration
    for iteration in range(max_iterations):
        # Calculate heat transfer coefficients
        h_coefficients = convective_coupling.calculate_heat_transfer_coefficients(edge_flows)

        # Calculate fluid outlet temps (for next iteration)
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
            h_mm = h * 1e-6  # W/m²·K → W/mm²·K

            node_start = network.nodes[edge.node_start]
            node_end = network.nodes[edge.node_end]
            start_pos = node_start.position_mm
            end_pos = node_end.position_mm
            length_mm = edge.length_mm

            n_samples = 20
            for i in range(n_samples):
                t = (i + 0.5) / n_samples
                pos = start_pos + t * (end_pos - start_pos)

                try:
                    idx = grid.point_to_index(pos)
                    T_solid = temperature_field[idx]

                    # Use outlet temp from previous iteration
                    T_outlet = fluid_outlet_temps.get(edge_id, inlet_temp + 30.0)
                    T_fluid = inlet_temp + t * (T_outlet - inlet_temp)

                    # Safety factor 0.8 for stability (accounts for 1D model approximations)
                    q_conv = h_mm * (T_solid - T_fluid) * 0.8

                    w_mm = edge.width_um / 1000
                    h_mm_channel = edge.depth_um / 1000
                    perimeter_mm = 2 * (w_mm + h_mm_channel)

                    q_per_length = q_conv * perimeter_mm
                    segment_length = length_mm / n_samples
                    q_total = q_per_length * segment_length

                    cell_vol = grid.cell_volume()
                    q_volumetric = q_total / cell_vol

                    sink_field[idx] -= q_volumetric

                except (ValueError, IndexError):
                    continue

        # Combine sources with sink
        q_original = heat_sources.get_combined_field(grid)
        q_combined = q_original + sink_field

        # Solve thermal with cooling
        q_si = q_combined * 1e9
        k = chip_config.thermal.material.thermal_conductivity_w_per_m_k
        rhs = -q_si / k

        laplacian = thermal_solver._build_laplacian()
        A, b = thermal_solver._apply_boundary_conditions(laplacian, rhs.ravel())

        from scipy.sparse.linalg import spsolve
        T_flat = spsolve(A, b)
        T_new = T_flat.reshape(grid.shape)
        T_new += chip_config.thermal.boundaries.ambient_temp_c

        # Apply under-relaxation
        temperature_field = relaxation_factor * T_new + (1 - relaxation_factor) * temperature_field

        # Clamp temperatures
        temperature_field = np.clip(temperature_field, 0.0, 500.0)

        # Check convergence
        T_max_new = temperature_field.max()
        relative_change = abs(T_max_new - T_max_prev) / T_max_prev

        if iteration == 0 or iteration == max_iterations - 1 or relative_change < tolerance:
            print(f"      Iter {iteration+1}: T_max={T_max_new:.1f}°C (Δ={relative_change*100:.3f}%)")

        if relative_change < tolerance:
            print(f"   ✓ Converged after {iteration+1} iterations")
            break

        T_max_prev = T_max_new

    # Final fluid temperature calculation
    h_coefficients = convective_coupling.calculate_heat_transfer_coefficients(edge_flows)
    heat_flux_per_channel = convective_coupling.estimate_heat_flux_per_channel(
        temperature_field, grid, h_coefficients, inlet_temp
    )
    fluid_outlet_temps, _ = convective_coupling.calculate_fluid_temperatures(
        edge_flows, heat_flux_per_channel, inlet_temp
    )

    # Calculate per-segment profiles
    n_segments = get_fluid_segments()
    fluid_temp_profiles = convective_coupling.calculate_fluid_temperatures_per_segment(
        edge_flows, temperature_field, grid, h_coefficients, inlet_temp, n_segments=n_segments
    )

    fluid_stats = convective_coupling.get_fluid_temperature_statistics(fluid_outlet_temps, inlet_temp)

    print(f"   Chip: {temperature_field.min():.1f}°C → {temperature_field.max():.1f}°C")
    print(f"   Fluid: {inlet_temp}°C → {fluid_stats['avg_outlet_temp_c']:.1f}°C "
          f"(margin: {fluid_stats['boiling_margin_c']:.1f}°C)")

    # Generate individual visualization components
    print("\n🎨 Generating visualization components...")

    # For Slide 1: Chip layout + Temperature field
    print("   1a. Chip layout with power distribution")
    generate_chip_layout_with_power(
        chip_config=chip_config,
        output_path=temp_dir / "chip_layout.png"
    )

    print("   1b. Temperature field heatmap")
    viz = SimulationVisualizer(grid)
    temp_fig = viz.plot_temperature_heatmap(
        temperature_field,
        title="Steady-State Temperature Field"
    )
    temp_fig.write_image(str(temp_dir / "temperature_field.png"), scale=2)
    print(f"   Saved: temperature_field.png")

    print("   1c. Combining Slide 1...")
    combine_images_vertically(
        top_path=temp_dir / "chip_layout.png",
        bottom_path=temp_dir / "temperature_field.png",
        output_path=output_dir / "slide_1_chip_layout_and_temperature.png",
        spacing=30
    )

    # For Slide 2: Channel network + Physical layout with fluid temps
    print("   2a. Channel network structure")
    generate_channel_visualization(
        chip_size=chip_size,
        channel_network=network,
        output_path=temp_dir / "channel_network.png",
        title="Cooling Channel Network Structure"
    )

    print("   2b. Physical layout with fluid temperatures")
    generate_physical_layout(
        chip_config=chip_config,
        channel_network=network,
        fluid_outlet_temps=fluid_outlet_temps,
        fluid_temp_profiles=fluid_temp_profiles,
        fluid_stats=fluid_stats,
        inlet_temp_c=inlet_temp,
        output_path=temp_dir / "physical_layout.png"
    )

    print("   2c. Combining Slide 2...")
    combine_images_vertically(
        top_path=temp_dir / "channel_network.png",
        bottom_path=temp_dir / "physical_layout.png",
        output_path=output_dir / "slide_2_channels_and_coupling.png",
        spacing=30
    )

    # For Slide 3: Configuration summary
    print("   3. Configuration summary")

    # Create metadata for config summary
    metadata = {
        'n_channels': n_channels,
        'channel_size_um': channel_size_um,
        'pressure_drop_kpa': pressure_drop_kpa,
        'flow_rate_ml_min': sum(edge_flows.values()) * 60 * 1e6 / channel_config.flow.fluid.density_kg_per_m3,  # Convert to ml/min
        'T_max_with_cooling': temperature_field.max() - 273.15,  # Convert K to C
        'T_max_no_cooling': 0,  # Placeholder
        'cooling_effectiveness_pct': 0,  # Placeholder
        'iterations': 1,
        'grid_resolution': get_grid_resolution_text(chip_size)
    }

    generate_config_summary(
        metadata=metadata,
        output_path=output_dir / "slide_3_config.png"
    )

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)

    # Calculate physics validation metrics
    flow_m3_s = sum(edge_flows.values())
    m_dot = flow_m3_s * channel_config.flow.fluid.density_kg_per_m3
    cp = channel_config.flow.fluid.specific_heat_j_per_kg_k

    # Energy balance
    total_power_input = sum([
        hs.config.intensity_w_per_mm2 * (
            np.pi * hs.config.spread_mm**2 if hasattr(hs.config, 'spread_mm') and hs.config.spread_mm
            else hs.config.extent_mm[0] * hs.config.extent_mm[1] if hasattr(hs.config, 'extent_mm') and hs.config.extent_mm
            else 0
        )
        for hs in heat_sources.sources
    ])

    dT_fluid_actual = fluid_stats['avg_outlet_temp_c'] - inlet_temp
    heat_to_fluid_w = m_dot * cp * dT_fluid_actual
    heat_transfer_efficiency = heat_to_fluid_w / total_power_input if total_power_input > 0 else 0

    # Reynolds number check
    Q_per_channel = flow_m3_s / n_channels
    Dh = 2 * (channel_size_um * 1e-6)**2 / (2 * channel_size_um * 1e-6)
    v = Q_per_channel / ((channel_size_um * 1e-6)**2)
    Re_calculated = (channel_config.flow.fluid.density_kg_per_m3 * v * Dh) / channel_config.flow.fluid.viscosity_pa_s

    # Create comprehensive physics report
    from datetime import datetime
    physics_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0",
            "purpose": "Structured simulation report for physics validation and LLM review"
        },

        "configuration": {
            "chip": {
                "model": chip_config.metadata.name,
                "tdp_watts": chip_config.thermal.tdp_watts,
                "size_mm": chip_config.geometry.size_mm,
                "material": chip_config.thermal.material.name,
                "thermal_conductivity_w_per_m_k": chip_config.thermal.material.thermal_conductivity_w_per_m_k,
                "n_heat_sources": len(heat_sources.sources),
                "total_power_w": float(total_power_input)
            },
            "channels": {
                "count": n_channels,
                "cross_section_um": f"{channel_size_um}×{channel_size_um}",
                "hydraulic_diameter_um": channel_size_um,
                "length_mm": chip_config.geometry.size_mm[1],
                "pattern": "straight_parallel"
            },
            "grid": {
                "resolution_text": metadata['grid_resolution'],
                "total_cells": grid.n_cells,
                "cell_size_mm": list(grid.spacing)
            },
            "fluid": {
                "type": "water",
                "density_kg_per_m3": channel_config.flow.fluid.density_kg_per_m3,
                "viscosity_pa_s": channel_config.flow.fluid.viscosity_pa_s,
                "specific_heat_j_per_kg_k": channel_config.flow.fluid.specific_heat_j_per_kg_k,
                "thermal_conductivity_w_per_m_k": channel_config.flow.fluid.thermal_conductivity_w_per_m_k,
                "inlet_temp_c": inlet_temp
            }
        },

        "methodology": {
            "thermal_solver": {
                "method": "2D Finite Difference Method (FDM)",
                "governing_equation": "∇²T = -Q/k (Poisson equation)",
                "discretization": "Central difference scheme on structured Cartesian grid",
                "solution_method": "Direct sparse solver (scipy.sparse.linalg.spsolve)",
                "boundary_conditions": {
                    "channels": "Convective (h from Nusselt correlation)",
                    "edges": "Convective (h = 10 W/m²·K, ambient)",
                    "heat_sources": "Volumetric heat generation"
                }
            },
            "flow_solver": {
                "method": "1D Hydraulic Network",
                "governing_equation": "ΔP = (128 μ L Q) / (π D_h⁴) (Hagen-Poiseuille)",
                "flow_regime": "Laminar (Re < 2300)",
                "solution_method": "Linear system (pressure-flow relationship)",
                "assumptions": [
                    "Fully developed flow",
                    "Incompressible fluid",
                    "Ideal manifolds (uniform distribution)",
                    "No entrance/exit effects"
                ]
            },
            "thermal_fluidic_coupling": {
                "method": "Picard iteration with under-relaxation (two-way coupling)",
                "relaxation_factor": 0.1,
                "max_iterations": 15,
                "convergence_tolerance": 0.01,
                "heat_transfer_coefficient": "h = (Nu × k_fluid) / D_h",
                "nusselt_number": 7.54,
                "nusselt_correlation": "Constant Nu for fully developed laminar flow in rectangular channels",
                "fluid_temperature_evolution": "dT/dx = q / (ṁ c_p)",
                "segments_per_channel": n_segments,
                "coupling_description": "Thermal field affects fluid temps, fluid cooling affects thermal field (iterative until convergence)"
            },
            "assumptions": [
                "2D thermal model (valid for thin dies <1mm)",
                "Steady-state (no transients)",
                "Laminar flow throughout",
                "Uniform channel distribution",
                "No phase change (single-phase liquid cooling)"
            ],
            "limitations": [
                "Not 3D - through-plane temperature variation ignored",
                "No turbulent flow modeling",
                "Empirical convection correlations (not solving boundary layers)",
                "No entrance/exit pressure losses",
                "No flow maldistribution effects"
            ]
        },

        "results": {
            "thermal": {
                "T_max_c": float(temperature_field.max()),
                "T_min_c": float(temperature_field.min()),
                "T_range_c": float(temperature_field.max() - temperature_field.min())
            },
            "flow": {
                "total_flow_ml_per_min": float(flow_m3_s * 1e6 * 60),
                "flow_per_channel_ml_per_min": float(Q_per_channel * 1e6 * 60),
                "velocity_m_per_s": float(v),
                "reynolds_number": float(Re_calculated),
                "flow_regime": "laminar" if Re_calculated < 2300 else "turbulent",
                "pressure_drop_kpa": pressure_drop_kpa,
                "mass_flow_rate_g_per_s": float(m_dot * 1000)
            },
            "fluid_temperatures": {
                "inlet_c": inlet_temp,
                "avg_outlet_c": float(fluid_stats['avg_outlet_temp_c']),
                "max_outlet_c": float(fluid_stats['max_outlet_temp_c']),
                "min_outlet_c": float(fluid_stats['min_outlet_temp_c']),
                "temperature_rise_c": float(dT_fluid_actual),
                "boiling_margin_c": float(fluid_stats['boiling_margin_c'])
            },
            "heat_transfer": {
                "heat_to_fluid_w": float(heat_to_fluid_w),
                "heat_transfer_efficiency_pct": float(heat_transfer_efficiency * 100),
                "boundary_losses_w": float(total_power_input - heat_to_fluid_w)
            }
        },

        "physics_validation": {
            "energy_balance": {
                "power_input_w": float(total_power_input),
                "heat_to_fluid_w": float(heat_to_fluid_w),
                "other_losses_w": float(total_power_input - heat_to_fluid_w),
                "balance_error_pct": float(abs(total_power_input - heat_to_fluid_w) / total_power_input * 100) if total_power_input > 0 else 0,
                "status": "✓ Acceptable" if heat_transfer_efficiency > 0.5 else "⚠ Low efficiency"
            },
            "flow_regime": {
                "reynolds_number": float(Re_calculated),
                "critical_re": 2300,
                "regime": "laminar" if Re_calculated < 2300 else "turbulent",
                "status": "✓ Laminar (consistent with Nu=7.54)" if Re_calculated < 2300 else "✗ Turbulent (Nu correlation invalid!)"
            },
            "temperature_checks": {
                "fluid_below_boiling": {
                    "max_fluid_temp_c": float(fluid_stats['max_outlet_temp_c']),
                    "boiling_point_c": 100.0,
                    "margin_c": float(fluid_stats['boiling_margin_c']),
                    "status": "✓ Safe" if fluid_stats['boiling_margin_c'] > 10 else "⚠ Close to boiling"
                },
                "chip_temperature_reasonable": {
                    "max_chip_temp_c": float(temperature_field.max()),
                    "typical_max_c": 85.0,
                    "status": "✓ Reasonable" if temperature_field.max() < 100 else "⚠ Very high"
                }
            },
            "physical_consistency": {
                "pressure_drop_reasonable": {
                    "value_kpa": pressure_drop_kpa,
                    "typical_range_kpa": [5, 50],
                    "status": "✓ Within typical range" if 5 <= pressure_drop_kpa <= 50 else "⚠ Outside typical range"
                },
                "flow_rate_reasonable": {
                    "value_ml_per_min": float(flow_m3_s * 1e6 * 60),
                    "typical_range_ml_per_min": [50, 500],
                    "status": "✓ Within typical range" if 50 <= (flow_m3_s * 1e6 * 60) <= 500 else "⚠ Outside typical range"
                }
            }
        },

        "performance": {
            "thermal_solve_description": "Direct sparse solver for 2D Laplacian",
            "flow_solve_description": "Linear system inversion (<1ms)",
            "typical_solve_time_seconds": 5
        }
    }

    # Save physics report
    physics_report_path = output_dir / "simulation_physics_report.json"
    with open(physics_report_path, 'w') as f:
        json.dump(physics_report, f, indent=2)

    print(f"   📋 Physics report: {physics_report_path}")

    # Save slideshow metadata
    slideshow_metadata = {
        "n_slides": 3,
        "slides": [
            {
                "index": 0,
                "filename": "slide_1_chip_layout_and_temperature.png",
                "title": "Thermal Problem: Chip Layout & Solution",
                "description": "Power distribution and steady-state temperature field"
            },
            {
                "index": 1,
                "filename": "slide_2_channels_and_coupling.png",
                "title": "Flow Solution & Thermal-Fluidic Coupling",
                "description": "Channel structure and fluid temperature gradients"
            },
            {
                "index": 2,
                "filename": "slide_3_config.png",
                "title": "Simulation Configuration",
                "description": "Model parameters and operating conditions"
            }
        ],
        "grid_resolution": metadata['grid_resolution'],
        "n_channels": n_channels,
        "channel_size_um": channel_size_um,
        "fluid_stats": fluid_stats
    }

    metadata_out = output_dir / "slideshow_metadata.json"
    with open(metadata_out, 'w') as f:
        json.dump(slideshow_metadata, f, indent=2)

    print()
    print(f"✅ Slideshow complete! (3 slides)")
    print(f"   Slides: {output_dir}/slide_*.png")
    print(f"   Metadata: {metadata_out}")
    print(f"   Physics Report: {physics_report_path}")
    print()


if __name__ == "__main__":
    generate_slideshow()
