"""
Microfluidic flow solver using network flow model.

Implements 1D network flow using circuit analogy (Hagen-Poiseuille).
This model is computationally efficient and works regardless of the
spatial dimension of the channel network embedding.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from ..core.channel_network import ChannelNetwork
from ..models.config import (
    ChannelConfig, nusselt_rectangular_channel, water_viscosity, water_thermal_conductivity
)


class MicrofluidicSolver:
    """
    1D network flow solver for microfluidic channels.

    Uses electrical circuit analogy:
    - Pressure drop ΔP analogous to voltage drop ΔV
    - Flow rate Q analogous to current I
    - Hydraulic resistance R_h analogous to electrical resistance R

    For laminar flow in rectangular channels (Hagen-Poiseuille):
    ΔP = R_h * Q
    where R_h = 32 μ L / D_h²

    Solves the network flow equations to find pressure at each node
    and flow rate through each channel.
    """

    def __init__(self, network: ChannelNetwork, config: ChannelConfig):
        """
        Initialize microfluidic solver.

        Args:
            network: ChannelNetwork instance
            config: ChannelConfig with fluid properties
        """
        self.network = network
        self.config = config

        # Fluid properties
        self.fluid = config.flow.fluid
        self.operating = config.flow.operating_conditions

        # Viscosity (Pa·s) - can be updated for temperature dependence
        self.mu = self.fluid.viscosity_pa_s

        # Fluid thermal conductivity (W/m·K) - can be updated for temperature dependence
        self.k_fluid = self.fluid.thermal_conductivity_w_per_m_k

        # Reference temperature for property calculations
        self._reference_temp_c = 25.0

        # Solver statistics
        self.solve_time = 0.0

    def update_fluid_properties(self, T_avg_celsius: float) -> None:
        """
        Update temperature-dependent fluid properties.

        Updates both viscosity and thermal conductivity based on
        average fluid temperature. Should be called during coupling
        iterations to account for property changes as fluid heats up.

        Args:
            T_avg_celsius: Average fluid temperature (°C)
        """
        self.mu = water_viscosity(T_avg_celsius)
        self.k_fluid = water_thermal_conductivity(T_avg_celsius)
        self._reference_temp_c = T_avg_celsius

    def _compute_hydraulic_resistance(self, edge_id: int) -> float:
        """
        Compute hydraulic resistance for a rectangular channel.

        For laminar flow in rectangular channel (exact solution):
        R_h = 12 μ L / (W H³ f(α))

        where:
          W, H = width and height (W >= H)
          α = H/W = aspect ratio
          f(α) = 1 - 0.63α  (correction factor for α < 1)

        This is exact for rectangular channels, unlike the circular pipe
        hydraulic diameter approximation which can be off by orders of magnitude.

        Reference: White, F.M. "Fluid Mechanics" (2016), Section 6.9

        Args:
            edge_id: Channel edge ID

        Returns:
            Hydraulic resistance (Pa·s/m³)
        """
        edge = self.network.edges[edge_id]

        # Convert to SI units
        width_m = edge.width_um * 1e-6
        depth_m = edge.depth_um * 1e-6
        length_m = edge.length_mm * 1e-3

        # For rectangular channels: W is larger dimension, H is smaller
        W = max(width_m, depth_m)
        H = min(width_m, depth_m)

        # Aspect ratio correction factor
        alpha = H / W
        if alpha < 1.0:
            # Correction factor for aspect ratio (empirical)
            # f(α) = 1 - 0.63α for α < 1
            # For α → 0 (very flat): f → 1 (formula exact)
            # For α → 1 (square): f → 0.37 (needs correction)
            f_alpha = 1.0 - 0.63 * alpha
        else:
            # Square channel (α = 1)
            f_alpha = 0.42  # Correction for square channels

        # Hydraulic resistance (rectangular channel, laminar flow)
        # R_h = 12 μ L / (W H³ f(α))
        R_h = 12 * self.mu * length_m / (W * H**3 * f_alpha)

        return R_h

    def solve_flow(self, inlet_pressure_pa: Optional[float] = None,
                  outlet_pressure_pa: Optional[float] = None) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Solve network flow equations.

        Applies Kirchhoff's laws:
        1. Pressure drops sum to zero around loops
        2. Flow rates sum to zero at junctions

        Args:
            inlet_pressure_pa: Inlet pressure (Pa), defaults from config
            outlet_pressure_pa: Outlet pressure (Pa), defaults to atmospheric

        Returns:
            (node_pressures, edge_flows)
            - node_pressures: Dict[node_id → pressure (Pa)]
            - edge_flows: Dict[edge_id → flow_rate (m³/s)]
        """
        if inlet_pressure_pa is None:
            inlet_pressure_pa = self.operating.inlet_pressure_pa

        if outlet_pressure_pa is None:
            outlet_pressure_pa = 101325.0  # Atmospheric pressure

        print(f"\nSolving microfluidic network flow...")
        print(f"Nodes: {len(self.network.nodes)}, Edges: {len(self.network.edges)}")

        # Get inlet and outlet nodes
        inlet_nodes = self.network.get_inlet_nodes()
        outlet_nodes = self.network.get_outlet_nodes()

        print(f"Inlets: {len(inlet_nodes)}, Outlets: {len(outlet_nodes)}")

        # Build system of equations: A * P = b
        # where P is vector of unknown pressures

        n_nodes = len(self.network.nodes)
        n_edges = len(self.network.edges)

        # Create mapping from node_id to matrix index
        node_to_idx = {nid: i for i, nid in enumerate(self.network.nodes.keys())}

        # Build conductance matrix (inverse resistance)
        # For each edge: G * (P_start - P_end) = Q
        # At each junction: Σ Q = 0

        A = lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        # Compute resistances for all edges
        edge_resistances = {}
        for eid in self.network.edges.keys():
            edge_resistances[eid] = self._compute_hydraulic_resistance(eid)

        # Build equations from edges
        for eid, edge in self.network.edges.items():
            i_start = node_to_idx[edge.node_start]
            i_end = node_to_idx[edge.node_end]

            R_h = edge_resistances[eid]
            G = 1.0 / R_h  # Conductance

            # Flow: Q = G * (P_start - P_end)
            # Add to node equations: dP/dt contributes to mass conservation
            A[i_start, i_start] += G
            A[i_start, i_end] -= G
            A[i_end, i_end] += G
            A[i_end, i_start] -= G

        # Apply boundary conditions (fix pressures at inlets and outlets)
        #
        # Strategy: Check network connectivity
        # - If connected network (most inlets/outlets have degree > 1): Fix one inlet, one outlet
        # - If disconnected (parallel independent channels): Fix all inlets and outlets
        #
        # Physical justification:
        # - Connected: Pressure distributes through network resistances
        # - Disconnected: Each channel needs its own boundary conditions

        # Check connectivity: are most boundary nodes connected to other nodes?
        boundary_degrees = []
        for nid in inlet_nodes + outlet_nodes:
            # Count edges connected to this node
            degree = sum(1 for e in self.network.edges.values()
                        if e.node_start == nid or e.node_end == nid)
            boundary_degrees.append(degree)

        avg_degree = np.mean(boundary_degrees) if boundary_degrees else 0

        # If average degree is ~1, we have disconnected parallel channels
        # If average degree > 1, we have a connected network
        is_disconnected = avg_degree <= 1.5

        if is_disconnected:
            # Disconnected parallel channels: fix all boundary pressures
            print(f"Note: Detected disconnected parallel channels")
            print(f"      Setting boundary conditions on all {len(inlet_nodes)} inlets and {len(outlet_nodes)} outlets")

            for nid in inlet_nodes:
                i = node_to_idx[nid]
                A[i, :] = 0
                A[i, i] = 1.0
                b[i] = inlet_pressure_pa

            for nid in outlet_nodes:
                i = node_to_idx[nid]
                A[i, :] = 0
                A[i, i] = 1.0
                b[i] = outlet_pressure_pa
        else:
            # Connected network: fix only primary inlet/outlet, let others float
            print(f"Note: Detected connected network")

            if len(inlet_nodes) > 0:
                main_inlet = inlet_nodes[0]
                i = node_to_idx[main_inlet]
                A[i, :] = 0
                A[i, i] = 1.0
                b[i] = inlet_pressure_pa

                if len(inlet_nodes) > 1:
                    print(f"      Fixed pressure at primary inlet (node {main_inlet})")
                    print(f"      Other {len(inlet_nodes)-1} inlet(s) pressure will be determined by flow")

            if len(outlet_nodes) > 0:
                main_outlet = outlet_nodes[0]
                i = node_to_idx[main_outlet]
                A[i, :] = 0
                A[i, i] = 1.0
                b[i] = outlet_pressure_pa

                if len(outlet_nodes) > 1:
                    print(f"      Fixed pressure at primary outlet (node {main_outlet})")
                    print(f"      Other {len(outlet_nodes)-1} outlet(s) pressure will be determined by flow")

        # Convert to CSR for efficient solving
        A_csr = A.tocsr()

        # Solve
        import time
        start = time.time()
        P = spsolve(A_csr, b)
        self.solve_time = time.time() - start

        print(f"Solved in {self.solve_time:.4f}s")

        # Extract node pressures
        node_pressures = {}
        for nid, i in node_to_idx.items():
            node_pressures[nid] = P[i]
            # Store in network node
            self.network.nodes[nid].pressure_pa = P[i]

        # Compute flow rates through edges
        edge_flows = {}
        for eid, edge in self.network.edges.items():
            P_start = node_pressures[edge.node_start]
            P_end = node_pressures[edge.node_end]
            R_h = edge_resistances[eid]

            # Flow rate (m³/s)
            Q = (P_start - P_end) / R_h

            edge_flows[eid] = Q

            # Convert to ml/min for storage
            Q_ml_per_min = Q * 1e6 * 60  # m³/s → ml/min
            edge.flow_rate_ml_per_min = Q_ml_per_min

            # Compute Reynolds number
            # Re = ρ V D_h / μ
            # where V = Q / A
            A_cross = edge.cross_sectional_area_m2
            V = Q / A_cross if A_cross > 0 else 0
            D_h = edge.hydraulic_diameter_m
            Re = self.fluid.density_kg_per_m3 * V * D_h / self.mu
            edge.reynolds_number = Re

        # Print statistics
        total_flow_m3_s = sum(edge_flows.values())
        total_flow_ml_min = total_flow_m3_s * 1e6 * 60
        avg_pressure_drop = np.mean([
            abs(node_pressures[e.node_start] - node_pressures[e.node_end])
            for e in self.network.edges.values()
        ])

        print(f"Total flow: {total_flow_ml_min:.3f} ml/min")
        print(f"Avg pressure drop: {avg_pressure_drop:.1f} Pa")

        # Check Reynolds numbers (should be laminar, Re < 2300)
        max_re = max(e.reynolds_number for e in self.network.edges.values() if e.reynolds_number is not None)
        print(f"Max Reynolds number: {max_re:.1f} (laminar if < 2300)")

        return node_pressures, edge_flows

    def compute_heat_transfer_coefficient(self, edge_id: int) -> float:
        """
        Compute convective heat transfer coefficient for a channel.

        Uses Nusselt number correlation for laminar flow:
        Nu = 3.66 (fully developed, uniform wall temperature)

        h = Nu * k_fluid / D_h

        Args:
            edge_id: Channel edge ID

        Returns:
            Heat transfer coefficient (W/m²·K)
        """
        edge = self.network.edges[edge_id]

        # Nusselt number for laminar flow in rectangular channel
        # (constant wall temperature, fully developed flow)
        # Using Shah & London (1978) correlation for aspect ratio dependence
        width = edge.width_um
        depth = edge.depth_um
        aspect_ratio = min(width, depth) / max(width, depth)
        Nu = nusselt_rectangular_channel(aspect_ratio)

        # Hydraulic diameter
        D_h = edge.hydraulic_diameter_m

        # Heat transfer coefficient using temperature-dependent k_fluid
        h = Nu * self.k_fluid / D_h

        return h

    def get_statistics(self) -> Dict[str, any]:
        """
        Get flow statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.network.edges or self.network.edges[next(iter(self.network.edges))].flow_rate_ml_per_min is None:
            return {'error': 'Flow not yet solved'}

        flow_rates = [e.flow_rate_ml_per_min for e in self.network.edges.values() if e.flow_rate_ml_per_min is not None]
        reynolds = [e.reynolds_number for e in self.network.edges.values() if e.reynolds_number is not None]
        pressures = [n.pressure_pa for n in self.network.nodes.values() if n.pressure_pa is not None]

        return {
            'total_flow_ml_per_min': sum(flow_rates),
            'avg_flow_ml_per_min': np.mean(flow_rates),
            'max_flow_ml_per_min': np.max(flow_rates),
            'avg_reynolds_number': np.mean(reynolds),
            'max_reynolds_number': np.max(reynolds),
            'min_pressure_pa': np.min(pressures),
            'max_pressure_pa': np.max(pressures),
            'max_pressure_drop_pa': np.max(pressures) - np.min(pressures),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MicrofluidicSolver(fluid={self.config.flow.fluid_type.value}, "
            f"μ={self.mu:.2e} Pa·s)"
        )
