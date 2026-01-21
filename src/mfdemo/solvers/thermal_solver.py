"""
Dimension-agnostic thermal solver using finite difference methods.

Implements steady-state and transient heat conduction with heat sources.
Uses Kronecker product construction for N-dimensional Laplacian operator.
"""

from typing import Optional, Tuple, Callable, TYPE_CHECKING
import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve, cg, gmres
import time

from ..core.grid import StructuredGrid
from ..core.heat_source import HeatSourceCollection
from ..models.config import ChipConfig, silicon_thermal_conductivity

if TYPE_CHECKING:
    from ..core.channel_network import ChannelNetwork
    from .microfluidic_solver import MicrofluidicSolver


class ThermalSolver:
    """
    Dimension-agnostic thermal solver for heat conduction.

    Solves the heat equation:
    - Steady-state: ∇²T = -q/k
    - Transient: ρc ∂T/∂t = k∇²T + q

    where T = temperature, q = heat source, k = thermal conductivity,
    ρ = density, c = specific heat

    The solver uses Kronecker product construction to build the Laplacian
    matrix in arbitrary dimensions, making it work seamlessly for 2D and 3D.
    """

    def __init__(self, grid: StructuredGrid, chip_config: ChipConfig):
        """
        Initialize thermal solver.

        Args:
            grid: StructuredGrid instance
            chip_config: ChipConfig with material properties
        """
        self.grid = grid
        self.config = chip_config

        # Material properties
        material = chip_config.thermal.material
        self.k = material.thermal_conductivity_w_per_m_k
        self.rho = material.density_kg_per_m3
        self.cp = material.specific_heat_j_per_kg_k
        self.alpha = material.thermal_diffusivity

        # Boundary conditions
        self.bc_config = chip_config.thermal.boundaries

        # Laplacian matrix (lazy initialization)
        self._laplacian: Optional[csr_matrix] = None
        self._laplacian_with_bc: Optional[csr_matrix] = None

        # Solver statistics
        self.solve_time = 0.0
        self.n_iterations = 0

    def update_thermal_conductivity(self, T_avg_celsius: float) -> None:
        """
        Update silicon thermal conductivity for temperature-dependent calculations.

        Uses power law correlation for silicon:
        k(T) = k_ref * (T_ref / T)^n
        where T_ref = 300K, k_ref = 150 W/m·K, n ≈ 1.3

        This should be called during coupling iterations to account
        for conductivity changes as the chip heats up.

        Note: Silicon k decreases with temperature (~0.4%/K near room temp),
        which means hotter chips conduct heat worse - a positive feedback loop.

        Args:
            T_avg_celsius: Average chip temperature (°C)
        """
        self.k = silicon_thermal_conductivity(T_avg_celsius)

    def _build_1d_laplacian(self, n: int, dx: float) -> csr_matrix:
        """
        Build 1D Laplacian matrix for dimension with n cells and spacing dx.

        Uses second-order central differences:
        d²f/dx² ≈ (f[i+1] - 2*f[i] + f[i-1]) / dx²

        Args:
            n: Number of cells in this dimension
            dx: Cell spacing (mm)

        Returns:
            Sparse matrix (n × n)
        """
        dx2 = (dx * 1e-3) ** 2  # Convert mm to m and square

        # Main diagonal: -2/dx²
        main_diag = np.full(n, -2.0 / dx2)

        # Off-diagonals: 1/dx²
        off_diag = np.full(n - 1, 1.0 / dx2)

        # Create sparse matrix
        laplacian_1d = diags(
            [main_diag, off_diag, off_diag],
            [0, 1, -1],
            shape=(n, n),
            format='csr'
        )

        return laplacian_1d

    def _build_laplacian(self) -> csr_matrix:
        """
        Build N-dimensional Laplacian using Kronecker products.

        For 2D: L = L_x ⊗ I_y + I_x ⊗ L_y
        For 3D: L = L_x ⊗ I_y ⊗ I_z + I_x ⊗ L_y ⊗ I_z + I_x ⊗ I_y ⊗ L_z

        This construction is dimension-agnostic and automatically
        handles any ndim.

        Returns:
            Sparse Laplacian matrix
        """
        if self._laplacian is not None:
            return self._laplacian

        print(f"Building {self.grid.ndim}D Laplacian matrix...")
        start_time = time.time()

        # Build 1D Laplacians for each dimension
        laplacians_1d = []
        for d in range(self.grid.ndim):
            n = self.grid.shape[d]
            dx = self.grid.spacing[d]
            laplacians_1d.append(self._build_1d_laplacian(n, dx))

        # Build identities for each dimension
        identities = []
        for d in range(self.grid.ndim):
            n = self.grid.shape[d]
            identities.append(eye(n, format='csr'))

        # Assemble N-dimensional Laplacian using Kronecker products
        # For each dimension d: L_d ⊗ I_d+1 ⊗ ... ⊗ I_ndim
        laplacian_nd = None

        for d in range(self.grid.ndim):
            # Build Kronecker product for this dimension
            term = laplacians_1d[d]

            # Multiply with identities for other dimensions
            for other_d in range(self.grid.ndim):
                if other_d != d:
                    if other_d < d:
                        # Identity comes before Laplacian
                        term = kron(identities[other_d], term, format='csr')
                    else:
                        # Identity comes after Laplacian
                        term = kron(term, identities[other_d], format='csr')

            # Add to total Laplacian
            if laplacian_nd is None:
                laplacian_nd = term
            else:
                laplacian_nd = laplacian_nd + term

        elapsed = time.time() - start_time
        print(f"Laplacian built in {elapsed:.3f}s, shape: {laplacian_nd.shape}, "
              f"nnz: {laplacian_nd.nnz}")

        self._laplacian = laplacian_nd
        return laplacian_nd

    def _apply_boundary_conditions(self, laplacian: csr_matrix,
                                   source: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """
        Apply boundary conditions to Laplacian and source term.

        Implements convective (Robin) boundary conditions:
        -k ∇T·n = h(T - T_ambient) at boundaries

        This is discretized using a one-sided difference approximation.

        Args:
            laplacian: Laplacian matrix
            source: Source term (flattened)

        Returns:
            (modified_laplacian, modified_source)
        """
        if self.grid.ndim != 2:
            # Only implemented for 2D currently
            return laplacian, source

        # Get boundary parameters from config
        if hasattr(self.config.thermal, 'boundaries'):
            bc_type = self.config.thermal.boundaries.type
            if bc_type == 'convective':
                h = self.config.thermal.boundaries.heat_transfer_coefficient_w_per_m2_k

                # Note: We're solving for temperature difference from ambient,
                # so T_amb = 0 in the equation (ambient offset added at end)
                T_amb_rel = 0.0

                # Apply convective BCs
                return self._apply_convective_bc_2d(laplacian, source, h, T_amb_rel)

        # Default: return unchanged (Neumann BC)
        return laplacian, source

    def _apply_convective_bc_2d(self, laplacian: csr_matrix, source: np.ndarray,
                                h: float, T_amb: float) -> Tuple[csr_matrix, np.ndarray]:
        """
        Apply convective boundary conditions for 2D grid using ghost point method.

        At boundaries: -k ∂T/∂n = h(T - T_amb)

        Using ghost points and central difference for the BC:
        For left boundary (i=0): -k(T[0] - T[-1])/(2*dx) = h(T[0] - T_amb)
        Solving for ghost: T[-1] = T[0] + (2*dx*h/k)*(T[0] - T_amb)

        Substituting into interior Laplacian approximation:
        ∇²T[0] ≈ (T[1] - 2*T[0] + T[-1])/dx²
               = (T[1] - 2*T[0] + T[0] + (2*dx*h/k)*(T[0] - T_amb))/dx²
               = (T[1] - T[0])/dx² + (2*h/k)*(T[0] - T_amb)/dx

        Rearranging for ∇²T = -Q/k:
        (T[1] - T[0])/dx² + (2*h/k)*T[0]/dx = -Q/k + (2*h/k)*T_amb/dx

        Args:
            laplacian: Original Laplacian matrix
            source: Original source term
            h: Heat transfer coefficient (W/m²·K)
            T_amb: Ambient temperature (relative, usually 0)

        Returns:
            (modified_laplacian, modified_source)
        """
        nx, ny = self.grid.shape
        dx, dy = self.grid.spacing  # mm
        dx_m = dx * 1e-3  # Convert to meters
        dy_m = dy * 1e-3

        # Convert to LIL format for efficient row modification
        A_mod = laplacian.tolil()
        b_mod = source.copy()

        def get_flat_index(i, j):
            """Convert 2D index to flattened index."""
            return i * ny + j

        # Process ALL boundary nodes
        for i in range(nx):
            for j in range(ny):
                is_boundary = (i == 0) or (i == nx-1) or (j == 0) or (j == ny-1)

                if not is_boundary:
                    continue

                idx = get_flat_index(i, j)

                # For boundary nodes, modify the diagonal coefficient
                # The standard Laplacian has: diag[idx] = -2/dx² - 2/dy²
                # We need to add the Robin BC contribution

                # X-direction contribution
                if i == 0 or i == nx-1:
                    # Add 2h/(k*dx) to diagonal (note: Laplacian diagonal is negative)
                    # This makes the diagonal LESS negative, reducing cooling
                    A_mod[idx, idx] += -2.0 * h / (self.k * dx_m)
                    # Add source contribution: -2h*T_amb/(k*dx)
                    b_mod[idx] += -2.0 * h * T_amb / (self.k * dx_m)

                # Y-direction contribution
                if j == 0 or j == ny-1:
                    A_mod[idx, idx] += -2.0 * h / (self.k * dy_m)
                    b_mod[idx] += -2.0 * h * T_amb / (self.k * dy_m)

        return A_mod.tocsr(), b_mod

    def solve_steady_state(self, heat_sources: HeatSourceCollection,
                          initial_temp: Optional[np.ndarray] = None,
                          solver_method: str = 'direct',
                          tolerance: float = 1e-6) -> np.ndarray:
        """
        Solve steady-state heat equation: ∇²T = -q/k

        Args:
            heat_sources: Heat source collection
            initial_temp: Initial guess (optional, used for iterative solvers)
            solver_method: 'direct' (spsolve), 'cg' (conjugate gradient), or 'gmres'
            tolerance: Convergence tolerance for iterative solvers

        Returns:
            Temperature field (shape: grid.shape)
        """
        print(f"\nSolving steady-state thermal problem ({solver_method})...")
        start_time = time.time()

        # Get heat source field
        q = heat_sources.get_combined_field(self.grid)  # W/mm²  or W/mm³

        # Convert to W/m³ (volumetric)
        # For 2D: W/mm² → W/m³ (assuming 1mm reference thickness): multiply by 1e9
        # For 3D: W/mm³ → W/m³: multiply by 1e9
        q_si = q * 1e9

        # Right-hand side: -q/k
        rhs = -q_si / self.k

        # Flatten
        rhs_flat = rhs.ravel()

        # Build Laplacian
        laplacian = self._build_laplacian()

        # Apply boundary conditions
        A, b = self._apply_boundary_conditions(laplacian, rhs_flat)

        # Solve linear system
        if solver_method == 'direct':
            T_flat = spsolve(A, b)
            self.n_iterations = 1

        elif solver_method == 'cg':
            # Conjugate gradient (for symmetric positive definite)
            # Note: Our Laplacian is negative definite, so we solve -A x = -b
            x0 = initial_temp.ravel() if initial_temp is not None else None
            T_flat, info = cg(-A, -b, x0=x0, tol=tolerance, maxiter=10000)
            if info > 0:
                print(f"Warning: CG did not converge ({info} iterations)")
            self.n_iterations = info if info > 0 else 10000

        elif solver_method == 'gmres':
            # Generalized minimal residual (works for any matrix)
            x0 = initial_temp.ravel() if initial_temp is not None else None
            T_flat, info = gmres(A, b, x0=x0, tol=tolerance, maxiter=10000)
            if info > 0:
                print(f"Warning: GMRES did not converge ({info} iterations)")
            self.n_iterations = info if info > 0 else 10000

        else:
            raise ValueError(f"Unknown solver method: {solver_method}")

        # Reshape to grid
        T = T_flat.reshape(self.grid.shape)

        # Add ambient temperature offset
        T_ambient = self.bc_config.ambient_temp_c
        T += T_ambient

        self.solve_time = time.time() - start_time
        print(f"Solved in {self.solve_time:.3f}s, "
              f"T_min={np.min(T):.2f}°C, T_max={np.max(T):.2f}°C")

        return T

    def solve_transient(self, heat_sources: HeatSourceCollection,
                       T_initial: np.ndarray,
                       dt: float,
                       n_steps: int,
                       method: str = 'implicit_euler') -> np.ndarray:
        """
        Solve transient heat equation: ρc ∂T/∂t = k∇²T + q

        Args:
            heat_sources: Heat source collection
            T_initial: Initial temperature field (shape: grid.shape)
            dt: Time step (s)
            n_steps: Number of time steps
            method: 'implicit_euler' or 'crank_nicolson'

        Returns:
            Final temperature field (shape: grid.shape)
        """
        print(f"\nSolving transient thermal problem ({method})...")
        print(f"Time steps: {n_steps}, dt: {dt}s, total time: {dt*n_steps}s")

        # Build Laplacian
        L = self._build_laplacian()

        # Time integration coefficient
        # ∂T/∂t = α∇²T + q/(ρc)
        # where α = k/(ρc) = thermal diffusivity

        rho_cp = self.rho * self.cp
        cell_volume = self.grid.cell_volume() * 1e-9  # mm³ to m³

        # Current temperature
        T = T_initial.copy()

        for step in range(n_steps):
            # Get heat source at current time
            time = step * dt
            q = heat_sources.get_combined_field(self.grid, time=time)
            q_si = q * 1e6 if self.grid.ndim == 2 else q * 1e9  # W/m²  or W/m³

            if method == 'implicit_euler':
                # (I - dt*α*L) T^{n+1} = T^n + dt*q/(ρc)
                A = eye(self.grid.n_cells) - dt * self.alpha * L
                b = T.ravel() + dt * q_si.ravel() / rho_cp

                T_new_flat = spsolve(A, b)
                T = T_new_flat.reshape(self.grid.shape)

            elif method == 'crank_nicolson':
                # (I - dt/2*α*L) T^{n+1} = (I + dt/2*α*L) T^n + dt*q/(ρc)
                A = eye(self.grid.n_cells) - (dt / 2) * self.alpha * L
                B = eye(self.grid.n_cells) + (dt / 2) * self.alpha * L
                b = B @ T.ravel() + dt * q_si.ravel() / rho_cp

                T_new_flat = spsolve(A, b)
                T = T_new_flat.reshape(self.grid.shape)

            else:
                raise ValueError(f"Unknown time integration method: {method}")

            # Progress update
            if (step + 1) % max(1, n_steps // 10) == 0:
                print(f"Step {step+1}/{n_steps}, "
                      f"T_min={np.min(T):.2f}°C, T_max={np.max(T):.2f}°C")

        return T

    def compute_heat_flux(self, T: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute heat flux vector field from temperature.

        q⃗ = -k ∇T

        Args:
            T: Temperature field (shape: grid.shape)

        Returns:
            Tuple of heat flux components (qx, qy) or (qx, qy, qz)
            Each has shape matching grid.shape
        """
        flux_components = []

        for d in range(self.grid.ndim):
            # Gradient in direction d
            grad_T = self.grid.compute_gradient(T, d)

            # Heat flux: q = -k * grad(T)
            flux = -self.k * grad_T

            flux_components.append(flux)

        return tuple(flux_components)

    def compute_total_heat_flow(self, T: np.ndarray) -> float:
        """
        Compute total heat flow out of domain boundaries.

        Args:
            T: Temperature field

        Returns:
            Total heat flow (W)
        """
        flux_components = self.compute_heat_flux(T)

        total_flow = 0.0

        # Sum flux across all boundaries
        for d in range(self.grid.ndim):
            # Flux at lower boundary (outward normal is negative direction)
            slices_lower = [slice(None)] * self.grid.ndim
            slices_lower[d] = 0
            face_area = self.grid.cell_volume() / self.grid.spacing[d]  # mm to mm²
            total_flow += np.sum(flux_components[d][tuple(slices_lower)]) * face_area

            # Flux at upper boundary (outward normal is positive direction)
            slices_upper = [slice(None)] * self.grid.ndim
            slices_upper[d] = -1
            total_flow -= np.sum(flux_components[d][tuple(slices_upper)]) * face_area

        # Convert to Watts (flux is in W/m², area is in mm²)
        total_flow *= 1e-6  # mm² to m²

        return total_flow

    def solve_steady_state_with_convection(
        self,
        heat_sources: HeatSourceCollection,
        channel_network: 'ChannelNetwork',
        flow_solver: 'MicrofluidicSolver',
        initial_temp: Optional[np.ndarray] = None,
        max_iterations: int = 10,
        temperature_tolerance: float = 0.5,
        solver_method: str = 'direct',
        relaxation_factor: float = 0.5
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve steady-state heat equation with convective cooling.

        Iteratively couples thermal and flow solutions:
        1. Solve conduction with current convective sink estimate
        2. Update convective sink based on new temperature
        3. Apply under-relaxation to prevent oscillations
        4. Repeat until temperature converges

        The heat equation with convection:
        ∇²T = -(q - Q_conv) / k

        where Q_conv is the volumetric heat removal by fluid (W/m³).

        Args:
            heat_sources: Heat source collection
            channel_network: ChannelNetwork with solved flow
            flow_solver: MicrofluidicSolver with computed flow rates
            initial_temp: Initial temperature guess (optional)
            max_iterations: Maximum coupling iterations
            temperature_tolerance: Convergence criterion (°C)
            solver_method: 'direct', 'cg', or 'gmres'
            relaxation_factor: Under-relaxation factor (0 < α ≤ 1)
                α = 1.0: no relaxation (may oscillate)
                α = 0.5: moderate relaxation (recommended)
                α = 0.3: heavy relaxation (slower but more stable)

        Returns:
            (temperature_field, convergence_info)
            - temperature_field: Final temperature (shape: grid.shape)
            - convergence_info: Dict with iteration count, residual, etc.
        """
        print(f"\nSolving steady-state thermal with convective cooling...")
        print(f"Max iterations: {max_iterations}, tolerance: {temperature_tolerance}°C")
        start_time = time.time()

        # Initial thermal solution (conduction only, for first estimate)
        if initial_temp is None:
            print("Computing initial temperature field (conduction only)...")
            T_old = self.solve_steady_state(heat_sources, solver_method=solver_method)
        else:
            T_old = initial_temp.copy()

        convergence_info = {
            'iterations': 0,
            'converged': False,
            'final_residual': np.inf,
            'residual_history': []
        }

        # Initialize convective sink
        Q_conv_old = np.zeros(self.grid.shape)

        # Iterative coupling loop
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Update temperature-dependent material properties
            if iteration > 0:
                # Average chip temperature for silicon conductivity
                T_chip_avg = np.mean(T_old)
                self.update_thermal_conductivity(T_chip_avg)

                # Average fluid temperature for water properties
                # Estimate as inlet temp + half the chip-to-inlet difference
                T_inlet = self.bc_config.ambient_temp_c
                T_fluid_avg = T_inlet + 0.5 * (T_chip_avg - T_inlet)
                flow_solver.update_fluid_properties(T_fluid_avg)

                print(f"Updated properties: k_Si={self.k:.1f} W/m·K, "
                      f"μ_water={flow_solver.mu:.2e} Pa·s")

            # Compute new convective sink based on current temperature
            # Enable fluid temperature rise model for more accurate physics
            Q_conv_new = self._compute_convective_sink(
                T_old, channel_network, flow_solver, model_fluid_heating=True
            )

            # Apply under-relaxation to prevent oscillations
            # Q_conv = α * Q_conv_new + (1 - α) * Q_conv_old
            if iteration == 0:
                Q_conv = Q_conv_new
            else:
                Q_conv = (relaxation_factor * Q_conv_new +
                         (1 - relaxation_factor) * Q_conv_old)

            Q_conv_old = Q_conv.copy()  # Store for next iteration

            # Get heat generation field
            q = heat_sources.get_combined_field(self.grid)  # W/mm² or W/mm³
            q_si = q * 1e6 if self.grid.ndim == 2 else q * 1e9  # W/m² or W/m³

            # Right-hand side: -(q - Q_conv) / k
            # Q_conv is heat REMOVAL (positive), so it reduces the heat generation
            rhs = -(q_si - Q_conv) / self.k
            rhs_flat = rhs.ravel()

            # Build Laplacian
            laplacian = self._build_laplacian()

            # Apply boundary conditions
            A, b = self._apply_boundary_conditions(laplacian, rhs_flat)

            # Solve linear system
            T_flat = spsolve(A, b)
            T_new = T_flat.reshape(self.grid.shape)

            # Add ambient temperature offset
            T_ambient = self.bc_config.ambient_temp_c
            T_new += T_ambient

            # Enforce physical temperature constraint:
            # Temperature cannot go below coolant inlet temperature
            T_fluid_inlet = flow_solver.operating.inlet_temp_c
            T_new = np.maximum(T_new, T_fluid_inlet)

            # Check convergence
            delta_T = np.abs(T_new - T_old)
            max_delta = np.max(delta_T)
            avg_delta = np.mean(delta_T)

            convergence_info['residual_history'].append(max_delta)

            print(f"T_min={np.min(T_new):.2f}°C, T_max={np.max(T_new):.2f}°C")
            print(f"ΔT: max={max_delta:.3f}°C, avg={avg_delta:.3f}°C")
            print(f"Convective cooling: {np.sum(Q_conv) * self.grid.cell_volume() * 1e-9:.2f} W")

            if max_delta < temperature_tolerance:
                convergence_info['converged'] = True
                convergence_info['iterations'] = iteration + 1
                convergence_info['final_residual'] = max_delta
                print(f"\n✅ Converged after {iteration + 1} iterations (ΔT = {max_delta:.3f}°C)")
                break

            T_old = T_new

        else:
            # Loop completed without break (did not converge)
            convergence_info['iterations'] = max_iterations
            convergence_info['final_residual'] = max_delta
            print(f"\n⚠️  Did not converge after {max_iterations} iterations (ΔT = {max_delta:.3f}°C)")

        self.solve_time = time.time() - start_time
        print(f"\nTotal solve time: {self.solve_time:.3f}s")

        return T_new, convergence_info

    def _compute_convective_sink(
        self,
        T_field: np.ndarray,
        channel_network: 'ChannelNetwork',
        flow_solver: 'MicrofluidicSolver',
        model_fluid_heating: bool = True
    ) -> np.ndarray:
        """
        Compute convective heat removal as volumetric sink term.

        For each channel edge:
        1. Get heat transfer coefficient: h = Nu * k_fluid / D_h
        2. Get channel wetted area in contact with silicon
        3. Compute Q_conv = h * A * (T_solid - T_fluid)
        4. Model fluid temperature rise along channel (optional)
        5. Distribute to grid cells adjacent to channel

        Args:
            T_field: Current temperature field (°C)
            channel_network: ChannelNetwork instance
            flow_solver: MicrofluidicSolver with flow solution
            model_fluid_heating: If True, model fluid temperature rise along channel

        Returns:
            Q_conv: Volumetric heat sink (W/m³), shape: grid.shape
        """
        Q_conv = np.zeros(self.grid.shape)

        # Get fluid properties
        T_fluid_inlet = flow_solver.operating.inlet_temp_c
        cp_fluid = flow_solver.fluid.specific_heat_j_per_kg_k
        rho_fluid = flow_solver.fluid.density_kg_per_m3

        for edge_id, edge in channel_network.edges.items():
            # Heat transfer coefficient (W/m²·K)
            h = flow_solver.compute_heat_transfer_coefficient(edge_id)

            # Channel geometry
            width_m = edge.width_um * 1e-6
            depth_m = edge.depth_um * 1e-6
            length_m = edge.length_mm * 1e-3

            # Wetted perimeter for rectangular channel: 2*(width + depth)
            # All 4 walls participate in convective heat transfer
            wetted_perimeter_m = 2 * (width_m + depth_m)
            wetted_area_m2 = wetted_perimeter_m * length_m

            # Map channel to grid cells
            cell_indices = self._map_channel_to_cells(edge, channel_network)

            if len(cell_indices) == 0:
                continue

            # Get flow rate for this edge
            if edge.flow_rate_ml_per_min is None or edge.flow_rate_ml_per_min == 0:
                continue  # No flow, no convective cooling

            # Mass flow rate (kg/s)
            Q_flow_m3_s = edge.flow_rate_ml_per_min * 1e-6 / 60  # ml/min → m³/s
            m_dot = Q_flow_m3_s * rho_fluid  # kg/s

            # Distribute heat removal along channel segments
            n_cells = len(cell_indices)
            segment_area = wetted_area_m2 / n_cells  # Area per segment

            # Initialize fluid temperature at inlet
            T_fluid = T_fluid_inlet
            Q_total_removed = 0.0

            for i, cell_idx in enumerate(cell_indices):
                # Local solid temperature
                T_solid = T_field[cell_idx]

                if model_fluid_heating and i > 0:
                    # Update fluid temperature based on heat absorbed
                    # Energy balance: Q = m_dot * cp * dT
                    # T_fluid_out = T_fluid_in + Q / (m_dot * cp)
                    dT_fluid = Q_segment_prev / (m_dot * cp_fluid + 1e-10)
                    T_fluid += dT_fluid

                # Average fluid temperature in this segment
                # Use current T_fluid as entering temperature
                T_fluid_avg = T_fluid

                # Convective heat removal in this segment (W)
                Q_segment = h * segment_area * (T_solid - T_fluid_avg)

                # Prevent unphysical negative cooling (fluid hotter than solid)
                Q_segment = max(Q_segment, 0.0)

                Q_total_removed += Q_segment
                Q_segment_prev = Q_segment  # Store for next iteration

                # Convert to volumetric sink (W/m³)
                cell_volume_m3 = self.grid.cell_volume() * 1e-9  # mm³ to m³
                Q_conv[cell_idx] += Q_segment / cell_volume_m3

        return Q_conv

    def _map_channel_to_cells(
        self,
        edge: 'ChannelEdge',
        network: 'ChannelNetwork'
    ) -> list:
        """
        Map a channel edge to grid cell indices.

        Uses line rasterization to find all grid cells intersected by
        the channel path.

        Args:
            edge: ChannelEdge instance
            network: ChannelNetwork containing the nodes

        Returns:
            List of grid cell indices (tuples)
        """
        from ..core.channel_network import ChannelEdge

        node_start = network.nodes[edge.node_start]
        node_end = network.nodes[edge.node_end]

        # Get path points
        if edge.path_points is not None:
            points = np.vstack([
                node_start.position_mm,
                edge.path_points,
                node_end.position_mm
            ])
        else:
            points = np.array([node_start.position_mm, node_end.position_mm])

        cell_indices = []

        # Rasterize path segments
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]

            # Number of steps proportional to distance
            distance = np.linalg.norm(p2 - p1)
            n_steps = max(int(distance / np.min(self.grid.spacing)) + 1, 2)

            # Interpolate along segment
            t = np.linspace(0, 1, n_steps)
            for ti in t:
                point = p1 + ti * (p2 - p1)

                try:
                    idx = self.grid.point_to_index(point)
                    if idx not in cell_indices:
                        cell_indices.append(idx)
                except ValueError:
                    # Point outside grid
                    continue

        return cell_indices

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ThermalSolver(ndim={self.grid.ndim}, "
            f"k={self.k} W/m·K, α={self.alpha:.2e} m²/s)"
        )
