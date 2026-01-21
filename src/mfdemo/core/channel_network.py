"""
Channel network representation using graph structure.

Dimension-agnostic channel network for microfluidic cooling systems.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from ..models.config import ChannelConfig
from .grid import StructuredGrid


@dataclass
class ChannelNode:
    """
    Node in the channel network (junction, inlet, outlet, or terminal).

    Attributes:
        id: Unique node identifier
        position_mm: Physical position (ndim-length array)
        node_type: Type of node ('inlet', 'outlet', 'junction', 'terminal')
        pressure_pa: Pressure at this node (Pa) - computed by solver
        temperature_c: Temperature at this node (°C) - computed by solver
    """
    id: int
    position_mm: np.ndarray
    node_type: str  # inlet, outlet, junction, terminal
    pressure_pa: Optional[float] = None
    temperature_c: Optional[float] = None

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.position_mm)

    def __hash__(self):
        """Make hashable for use in sets."""
        return hash(self.id)


@dataclass
class ChannelEdge:
    """
    Edge in the channel network (channel segment).

    Represents a physical channel connecting two nodes.

    Attributes:
        id: Unique edge identifier
        node_start: Starting node ID
        node_end: Ending node ID
        width_um: Channel width (μm)
        depth_um: Channel depth (μm)
        length_mm: Channel length (mm)
        path_points: Optional array of waypoints for curved channels (shape: n_points × ndim)
        flow_rate_ml_per_min: Flow rate through channel (ml/min) - computed by solver
        reynolds_number: Reynolds number - computed by solver
    """
    id: int
    node_start: int
    node_end: int
    width_um: float
    depth_um: float
    length_mm: float
    path_points: Optional[np.ndarray] = None
    flow_rate_ml_per_min: Optional[float] = None
    reynolds_number: Optional[float] = None

    @property
    def hydraulic_diameter_m(self) -> float:
        """
        Hydraulic diameter for rectangular channel (m).

        D_h = 2WH / (W + H) where W = width, H = depth
        """
        w = self.width_um * 1e-6  # Convert to meters
        h = self.depth_um * 1e-6
        return 2 * w * h / (w + h)

    @property
    def cross_sectional_area_m2(self) -> float:
        """Cross-sectional area (m²)."""
        w = self.width_um * 1e-6
        h = self.depth_um * 1e-6
        return w * h

    @property
    def hydraulic_resistance(self) -> float:
        """
        Hydraulic resistance for Poiseuille flow.

        R = 32 μ L / D_h² (for laminar flow)
        Returns resistance in Pa·s/m³
        """
        # Default viscosity (water at 20°C): 0.001 Pa·s
        mu = 0.001  # Pa·s
        L = self.length_mm * 1e-3  # Convert to meters
        D_h = self.hydraulic_diameter_m

        return 32 * mu * L / (D_h ** 2)

    def __hash__(self):
        """Make hashable for use in sets."""
        return hash(self.id)


class ChannelNetwork:
    """
    Graph-based representation of microfluidic channel network.

    Channels are represented as a directed graph where:
    - Nodes represent junctions, inlets, outlets
    - Edges represent channel segments with physical properties

    The network is dimension-agnostic and works for 2D or 3D layouts.
    """

    def __init__(self, ndim: int):
        """
        Initialize empty channel network.

        Args:
            ndim: Number of spatial dimensions (2 or 3)
        """
        self.ndim = ndim
        self.nodes: Dict[int, ChannelNode] = {}
        self.edges: Dict[int, ChannelEdge] = {}

        # Next IDs for auto-generation
        self._next_node_id = 0
        self._next_edge_id = 0

        # Cached adjacency information
        self._adjacency: Optional[Dict[int, Set[int]]] = None

    def add_node(self, position: np.ndarray, node_type: str,
                 node_id: Optional[int] = None) -> int:
        """
        Add a node to the network.

        Args:
            position: Physical position (ndim-length array)
            node_type: Type ('inlet', 'outlet', 'junction', 'terminal')
            node_id: Optional explicit node ID (auto-generated if None)

        Returns:
            Node ID

        Raises:
            ValueError: If position dimensions don't match network or node_id exists
        """
        position = np.asarray(position, dtype=np.float64)
        if len(position) != self.ndim:
            raise ValueError(
                f"Position has {len(position)} dimensions, expected {self.ndim}"
            )

        if node_id is None:
            node_id = self._next_node_id
            self._next_node_id += 1
        elif node_id in self.nodes:
            raise ValueError(f"Node ID {node_id} already exists")

        self.nodes[node_id] = ChannelNode(
            id=node_id,
            position_mm=position,
            node_type=node_type
        )

        # Invalidate cached adjacency
        self._adjacency = None

        return node_id

    def add_edge(self, node_start: int, node_end: int,
                 width_um: float, depth_um: float,
                 path_points: Optional[np.ndarray] = None,
                 edge_id: Optional[int] = None) -> int:
        """
        Add a channel edge between two nodes.

        Args:
            node_start: Starting node ID
            node_end: Ending node ID
            width_um: Channel width (μm)
            depth_um: Channel depth (μm)
            path_points: Optional waypoints for curved path (shape: n_points × ndim)
            edge_id: Optional explicit edge ID (auto-generated if None)

        Returns:
            Edge ID

        Raises:
            ValueError: If nodes don't exist or edge_id exists
        """
        if node_start not in self.nodes or node_end not in self.nodes:
            raise ValueError("Both nodes must exist before adding edge")

        if edge_id is None:
            edge_id = self._next_edge_id
            self._next_edge_id += 1
        elif edge_id in self.edges:
            raise ValueError(f"Edge ID {edge_id} already exists")

        # Compute length
        pos_start = self.nodes[node_start].position_mm
        pos_end = self.nodes[node_end].position_mm

        if path_points is not None:
            # Curved path: sum of segment lengths
            points = np.vstack([pos_start, path_points, pos_end])
            length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        else:
            # Straight path
            length = np.linalg.norm(pos_end - pos_start)
            path_points = None

        self.edges[edge_id] = ChannelEdge(
            id=edge_id,
            node_start=node_start,
            node_end=node_end,
            width_um=width_um,
            depth_um=depth_um,
            length_mm=length,
            path_points=path_points
        )

        # Invalidate cached adjacency
        self._adjacency = None

        return edge_id

    def get_adjacency(self) -> Dict[int, Set[int]]:
        """
        Get adjacency list representation.

        Returns:
            Dictionary mapping node_id → set of connected node_ids
        """
        if self._adjacency is not None:
            return self._adjacency

        adjacency: Dict[int, Set[int]] = {nid: set() for nid in self.nodes.keys()}

        for edge in self.edges.values():
            adjacency[edge.node_start].add(edge.node_end)
            # For undirected graph, also add reverse:
            # adjacency[edge.node_end].add(edge.node_start)

        self._adjacency = adjacency
        return adjacency

    def get_inlet_nodes(self) -> List[int]:
        """Get list of inlet node IDs."""
        return [nid for nid, node in self.nodes.items()
                if node.node_type == 'inlet']

    def get_outlet_nodes(self) -> List[int]:
        """Get list of outlet node IDs."""
        return [nid for nid, node in self.nodes.items()
                if node.node_type == 'outlet']

    def get_edges_from_node(self, node_id: int) -> List[int]:
        """
        Get all edge IDs originating from a node.

        Args:
            node_id: Node ID

        Returns:
            List of edge IDs
        """
        return [eid for eid, edge in self.edges.items()
                if edge.node_start == node_id]

    def get_edges_to_node(self, node_id: int) -> List[int]:
        """
        Get all edge IDs terminating at a node.

        Args:
            node_id: Node ID

        Returns:
            List of edge IDs
        """
        return [eid for eid, edge in self.edges.items()
                if edge.node_end == node_id]

    def rasterize_to_grid(self, grid: StructuredGrid) -> np.ndarray:
        """
        Convert channel network to binary mask on grid.

        Marks grid cells that contain channels.

        Args:
            grid: StructuredGrid instance

        Returns:
            Boolean array (shape: grid.shape) where True = channel present

        Raises:
            ValueError: If grid dimensions don't match network
        """
        if grid.ndim != self.ndim:
            raise ValueError(
                f"Grid is {grid.ndim}D but network is {self.ndim}D"
            )

        mask = np.zeros(grid.shape, dtype=bool)

        # Rasterize each edge
        for edge in self.edges.values():
            node_start = self.nodes[edge.node_start]
            node_end = self.nodes[edge.node_end]

            # Get path points (or just endpoints for straight channels)
            if edge.path_points is not None:
                points = np.vstack([
                    node_start.position_mm,
                    edge.path_points,
                    node_end.position_mm
                ])
            else:
                points = np.array([node_start.position_mm, node_end.position_mm])

            # Draw line segments between consecutive points
            for i in range(len(points) - 1):
                mask |= self._draw_line_on_grid(
                    grid, points[i], points[i+1], edge.width_um * 1e-3  # Convert to mm
                )

        return mask

    def _draw_line_on_grid(self, grid: StructuredGrid,
                          p1: np.ndarray, p2: np.ndarray,
                          width_mm: float) -> np.ndarray:
        """
        Draw a line segment on grid with specified width.

        Uses Bresenham-like algorithm generalized to N dimensions.

        Args:
            grid: StructuredGrid instance
            p1, p2: Endpoints (mm)
            width_mm: Line width (mm)

        Returns:
            Boolean mask with line drawn
        """
        mask = np.zeros(grid.shape, dtype=bool)

        # Number of steps proportional to distance
        distance = np.linalg.norm(p2 - p1)
        n_steps = int(distance / np.min(grid.spacing)) + 1

        # Interpolate points along line
        t = np.linspace(0, 1, n_steps)
        for ti in t:
            point = p1 + ti * (p2 - p1)

            # Convert to grid indices
            try:
                idx = grid.point_to_index(point)

                # Mark cell and neighbors within width
                width_cells = int(width_mm / np.min(grid.spacing)) + 1

                # Create hypercube of indices around center
                ranges = [range(max(0, idx[d] - width_cells),
                               min(grid.shape[d], idx[d] + width_cells + 1))
                         for d in range(grid.ndim)]

                for multi_idx in np.ndindex(*[len(r) for r in ranges]):
                    actual_idx = tuple(ranges[d][multi_idx[d]] for d in range(grid.ndim))
                    mask[actual_idx] = True

            except ValueError:
                # Point outside grid, skip
                continue

        return mask

    def compute_total_channel_volume(self) -> float:
        """
        Compute total volume of all channels.

        Returns:
            Total volume (mm³)
        """
        total_volume = 0.0
        for edge in self.edges.values():
            # Volume = cross_section × length
            width_mm = edge.width_um * 1e-3
            depth_mm = edge.depth_um * 1e-3
            cross_section = width_mm * depth_mm
            total_volume += cross_section * edge.length_mm

        return total_volume

    def get_statistics(self) -> Dict[str, any]:
        """
        Get network statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'n_inlets': len(self.get_inlet_nodes()),
            'n_outlets': len(self.get_outlet_nodes()),
            'total_channel_length_mm': sum(e.length_mm for e in self.edges.values()),
            'total_channel_volume_mm3': self.compute_total_channel_volume(),
            'avg_channel_width_um': np.mean([e.width_um for e in self.edges.values()]),
            'avg_channel_depth_um': np.mean([e.depth_um for e in self.edges.values()]),
        }

    def __len__(self) -> int:
        """Number of nodes in network."""
        return len(self.nodes)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChannelNetwork(ndim={self.ndim}, "
            f"nodes={len(self.nodes)}, edges={len(self.edges)})"
        )
