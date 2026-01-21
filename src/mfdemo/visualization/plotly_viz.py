"""
Plotly-based visualization for simulation results.

Provides interactive visualizations for:
- Temperature heatmaps
- Pressure contours
- Flow vector fields
- Channel network overlays
- Comparative analysis plots
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..core.grid import StructuredGrid
from ..core.channel_network import ChannelNetwork


class SimulationVisualizer:
    """
    Interactive visualization for simulation results.

    Creates Plotly figures that work for 2D and 3D simulations.
    """

    def __init__(self, grid: StructuredGrid):
        """
        Initialize visualizer.

        Args:
            grid: StructuredGrid instance
        """
        self.grid = grid
        self.ndim = grid.ndim

    def plot_temperature_heatmap(self,
                                temperature: np.ndarray,
                                title: str = "Temperature Distribution",
                                colorscale: str = "Jet",
                                show_colorbar: bool = True) -> go.Figure:
        """
        Create temperature heatmap.

        Args:
            temperature: Temperature field (shape: grid.shape)
            title: Plot title
            colorscale: Plotly colorscale name
            show_colorbar: Whether to show colorbar

        Returns:
            Plotly Figure
        """
        if self.ndim == 2:
            return self._plot_2d_heatmap(temperature, title, colorscale, show_colorbar)
        elif self.ndim == 3:
            return self._plot_3d_volume(temperature, title, colorscale)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

    def _plot_2d_heatmap(self,
                        field: np.ndarray,
                        title: str,
                        colorscale: str,
                        show_colorbar: bool) -> go.Figure:
        """Create 2D heatmap."""
        # Get coordinate arrays
        coords = self.grid.cell_centers()
        x = coords[0][:, 0]  # X coordinates (1D)
        y = coords[1][0, :]  # Y coordinates (1D)

        fig = go.Figure(data=go.Heatmap(
            z=field.T,  # Transpose for correct orientation
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title="°C") if show_colorbar else None,
            hoverongaps=False,
            hovertemplate='x: %{x:.2f} mm<br>y: %{y:.2f} mm<br>T: %{z:.2f} °C<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            width=800,
            height=700,
            template="plotly_white"
        )

        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    def _plot_3d_volume(self,
                       field: np.ndarray,
                       title: str,
                       colorscale: str) -> go.Figure:
        """Create 3D volume rendering."""
        # For 3D, create slice views
        mid_x = field.shape[0] // 2
        mid_y = field.shape[1] // 2
        mid_z = field.shape[2] // 2

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('XY Slice (mid-Z)', 'XZ Slice (mid-Y)', 'YZ Slice (mid-X)')
        )

        # XY slice
        fig.add_trace(
            go.Heatmap(z=field[:, :, mid_z].T, colorscale=colorscale, showscale=False),
            row=1, col=1
        )

        # XZ slice
        fig.add_trace(
            go.Heatmap(z=field[:, mid_y, :].T, colorscale=colorscale, showscale=False),
            row=1, col=2
        )

        # YZ slice
        fig.add_trace(
            go.Heatmap(z=field[mid_x, :, :].T, colorscale=colorscale, showscale=True),
            row=1, col=3
        )

        fig.update_layout(
            title=title,
            width=1400,
            height=500,
            template="plotly_white"
        )

        return fig

    def plot_with_channel_overlay(self,
                                  temperature: np.ndarray,
                                  channel_network: ChannelNetwork,
                                  title: str = "Temperature with Channel Network") -> go.Figure:
        """
        Plot temperature with channel network overlay.

        Args:
            temperature: Temperature field
            channel_network: ChannelNetwork instance
            title: Plot title

        Returns:
            Plotly Figure
        """
        if self.ndim != 2:
            raise NotImplementedError("Channel overlay only supported for 2D")

        # Create base heatmap
        fig = self._plot_2d_heatmap(temperature, title, "Jet", True)

        # Add channel edges
        for edge in channel_network.edges.values():
            node_start = channel_network.nodes[edge.node_start]
            node_end = channel_network.nodes[edge.node_end]

            x = [node_start.position_mm[0], node_end.position_mm[0]]
            y = [node_start.position_mm[1], node_end.position_mm[1]]

            # Line width proportional to channel width
            line_width = max(1, edge.width_um / 20)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='white', width=line_width),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add channel nodes
        inlet_nodes = [n for n in channel_network.nodes.values() if n.node_type == 'inlet']
        outlet_nodes = [n for n in channel_network.nodes.values() if n.node_type == 'outlet']

        if inlet_nodes:
            fig.add_trace(go.Scatter(
                x=[n.position_mm[0] for n in inlet_nodes],
                y=[n.position_mm[1] for n in inlet_nodes],
                mode='markers',
                marker=dict(size=10, color='blue', symbol='circle'),
                name='Inlets',
                hovertemplate='Inlet<br>x: %{x:.2f} mm<br>y: %{y:.2f} mm<extra></extra>'
            ))

        if outlet_nodes:
            fig.add_trace(go.Scatter(
                x=[n.position_mm[0] for n in outlet_nodes],
                y=[n.position_mm[1] for n in outlet_nodes],
                mode='markers',
                marker=dict(size=10, color='green', symbol='square'),
                name='Outlets',
                hovertemplate='Outlet<br>x: %{x:.2f} mm<br>y: %{y:.2f} mm<extra></extra>'
            ))

        return fig

    def plot_flow_vectors(self,
                         velocity_x: np.ndarray,
                         velocity_y: np.ndarray,
                         title: str = "Flow Field",
                         subsample: int = 5) -> go.Figure:
        """
        Plot flow vector field (quiver plot).

        Args:
            velocity_x: X-component of velocity
            velocity_y: Y-component of velocity
            title: Plot title
            subsample: Subsample factor for vectors (show every Nth vector)

        Returns:
            Plotly Figure
        """
        if self.ndim != 2:
            raise NotImplementedError("Flow vectors only supported for 2D")

        # Get coordinate arrays
        coords = self.grid.cell_centers()
        X = coords[0][::subsample, ::subsample]
        Y = coords[1][::subsample, ::subsample]
        U = velocity_x[::subsample, ::subsample]
        V = velocity_y[::subsample, ::subsample]

        # Compute magnitude
        magnitude = np.sqrt(U**2 + V**2)

        fig = go.Figure()

        # Add quiver plot using scatter
        # (Plotly doesn't have native quiver, so we create arrows manually)
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                if magnitude[i, j] > 0:
                    # Arrow from (x, y) to (x+u, y+v)
                    x0, y0 = X[i, j], Y[i, j]
                    u, v = U[i, j], V[i, j]

                    # Scale arrows
                    scale = 0.5
                    x1, y1 = x0 + u * scale, y0 + v * scale

                    fig.add_trace(go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode='lines',
                        line=dict(color='blue', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Add arrowhead
                    fig.add_annotation(
                        x=x1, y=y1,
                        ax=x0, ay=y0,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor='blue'
                    )

        # Add magnitude as background
        coords_full = self.grid.cell_centers()
        fig.add_trace(go.Heatmap(
            z=magnitude.T,
            x=coords_full[0][:, 0],
            y=coords_full[1][0, :],
            colorscale='Blues',
            opacity=0.3,
            showscale=True,
            colorbar=dict(title="Velocity<br>Magnitude"),
            hoverinfo='skip'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            width=800,
            height=700,
            template="plotly_white"
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    def plot_pressure_contours(self,
                              pressure: np.ndarray,
                              title: str = "Pressure Distribution",
                              n_contours: int = 20) -> go.Figure:
        """
        Plot pressure contours.

        Args:
            pressure: Pressure field
            title: Plot title
            n_contours: Number of contour levels

        Returns:
            Plotly Figure
        """
        if self.ndim != 2:
            raise NotImplementedError("Pressure contours only for 2D")

        coords = self.grid.cell_centers()
        x = coords[0][:, 0]
        y = coords[1][0, :]

        fig = go.Figure(data=go.Contour(
            z=pressure.T,
            x=x,
            y=y,
            colorscale='Viridis',
            ncontours=n_contours,
            colorbar=dict(title="Pa"),
            hovertemplate='x: %{x:.2f} mm<br>y: %{y:.2f} mm<br>P: %{z:.1f} Pa<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            width=800,
            height=700,
            template="plotly_white"
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    def plot_multi_view(self,
                       temperature: np.ndarray,
                       channel_network: Optional[ChannelNetwork] = None,
                       pressure: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create multi-panel view with temperature, channels, and pressure.

        Args:
            temperature: Temperature field
            channel_network: Optional channel network
            pressure: Optional pressure field

        Returns:
            Plotly Figure with subplots
        """
        if self.ndim != 2:
            raise NotImplementedError("Multi-view only for 2D")

        # Determine number of subplots
        n_plots = 1  # Temperature always shown
        if channel_network:
            n_plots += 1
        if pressure is not None:
            n_plots += 1

        # Create subplots
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols

        subplot_titles = []
        if True:  # Temperature
            subplot_titles.append("Temperature (°C)")
        if channel_network:
            subplot_titles.append("Channel Network")
        if pressure is not None:
            subplot_titles.append("Pressure (Pa)")

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )

        coords = self.grid.cell_centers()
        x = coords[0][:, 0]
        y = coords[1][0, :]

        col = 1
        row = 1

        # Temperature heatmap
        fig.add_trace(
            go.Heatmap(
                z=temperature.T,
                x=x, y=y,
                colorscale='Jet',
                colorbar=dict(x=0.3, title="°C"),
                hovertemplate='T: %{z:.2f} °C<extra></extra>'
            ),
            row=row, col=col
        )
        col += 1
        if col > cols:
            col = 1
            row += 1

        # Channel network
        if channel_network:
            # Create channel mask
            mask = channel_network.rasterize_to_grid(self.grid).astype(float)

            fig.add_trace(
                go.Heatmap(
                    z=mask.T,
                    x=x, y=y,
                    colorscale=[[0, 'white'], [1, 'blue']],
                    showscale=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            col += 1
            if col > cols:
                col = 1
                row += 1

        # Pressure contours
        if pressure is not None:
            fig.add_trace(
                go.Contour(
                    z=pressure.T,
                    x=x, y=y,
                    colorscale='Viridis',
                    ncontours=15,
                    colorbar=dict(x=1.0, title="Pa"),
                    hovertemplate='P: %{z:.1f} Pa<extra></extra>'
                ),
                row=row, col=col
            )

        fig.update_layout(
            title="Simulation Results Overview",
            width=1200,
            height=400 * rows,
            template="plotly_white"
        )

        return fig


def plot_comparative_analysis(results_list: List[Dict[str, Any]],
                              metric: str = "max_temp",
                              title: Optional[str] = None) -> go.Figure:
    """
    Create comparative bar chart for multiple simulations.

    Args:
        results_list: List of result dictionaries from simulations
        metric: Metric to compare ('max_temp', 'pressure_drop', 'flow_rate')
        title: Optional plot title

    Returns:
        Plotly Figure
    """
    # Extract data
    labels = []
    values = []

    for result in results_list:
        chip_name = result.get('chip', {}).get('name', 'Unknown')
        channel_name = result.get('channel', {}).get('name', 'Unknown')
        label = f"{chip_name[:15]}<br>{channel_name[:15]}"
        labels.append(label)

        if metric == "max_temp":
            value = result.get('performance', {}).get('max_temp_c', 0)
        elif metric == "pressure_drop":
            value = result.get('performance', {}).get('max_pressure_drop_pa', 0)
        elif metric == "flow_rate":
            value = result.get('performance', {}).get('total_flow_ml_min', 0)
        else:
            value = 0

        values.append(value)

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color='steelblue',
            hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
        )
    ])

    # Set title and labels based on metric
    metric_info = {
        'max_temp': ('Maximum Temperature', '°C'),
        'pressure_drop': ('Pressure Drop', 'Pa'),
        'flow_rate': ('Flow Rate', 'ml/min')
    }

    metric_title, unit = metric_info.get(metric, ('Metric', ''))

    if title is None:
        title = f"Comparative Analysis: {metric_title}"

    fig.update_layout(
        title=title,
        xaxis_title="Configuration",
        yaxis_title=f"{metric_title} ({unit})",
        width=1000,
        height=600,
        template="plotly_white"
    )

    return fig


def plot_performance_matrix(results_dict: Dict[Tuple[str, str], Dict],
                           metric: str = "max_temp") -> go.Figure:
    """
    Create heatmap matrix of performance across chip×channel combinations.

    Args:
        results_dict: Dictionary keyed by (chip_name, channel_name) tuples
        metric: Metric to visualize

    Returns:
        Plotly Figure
    """
    # Extract unique chips and channels
    chips = sorted(set(k[0] for k in results_dict.keys()))
    channels = sorted(set(k[1] for k in results_dict.keys()))

    # Create matrix
    matrix = np.zeros((len(chips), len(channels)))

    for i, chip in enumerate(chips):
        for j, channel in enumerate(channels):
            key = (chip, channel)
            if key in results_dict:
                result = results_dict[key]
                if metric == "max_temp":
                    matrix[i, j] = result.get('performance', {}).get('max_temp_c', np.nan)
                elif metric == "pressure_drop":
                    matrix[i, j] = result.get('performance', {}).get('max_pressure_drop_pa', np.nan)
                elif metric == "flow_rate":
                    matrix[i, j] = result.get('performance', {}).get('total_flow_ml_min', np.nan)
            else:
                matrix[i, j] = np.nan

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[c[:20] for c in channels],
        y=[c[:20] for c in chips],
        colorscale='RdYlGn_r' if metric == 'max_temp' else 'Viridis',
        hovertemplate='Chip: %{y}<br>Channel: %{x}<br>Value: %{z:.2f}<extra></extra>',
        text=np.round(matrix, 1),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))

    metric_info = {
        'max_temp': 'Maximum Temperature (°C)',
        'pressure_drop': 'Pressure Drop (Pa)',
        'flow_rate': 'Flow Rate (ml/min)'
    }

    title = f"Performance Matrix: {metric_info.get(metric, metric)}"

    fig.update_layout(
        title=title,
        xaxis_title="Channel Configuration",
        yaxis_title="Chip Configuration",
        width=900,
        height=600,
        template="plotly_white"
    )

    return fig
