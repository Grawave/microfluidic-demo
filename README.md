# Microfluidic Cooling Simulation Demo

A technical demonstration of coupled thermal-fluidic physics simulation for microfluidic cooling of high-power processors (AI accelerators, GPUs, data center CPUs). Built to showcase simulation methodology and physics-based modeling.

**Live Demo**: https://jani-demo.com

## 🎯 Project Purpose

This is a **technical demonstration** of microfluidic cooling simulation capabilities:
- Coupled thermal-fluidic solver (2D Finite Difference + 1D network flow)
- Interactive visualization slideshow explaining methodology
- Physics-accurate results with documented assumptions and validation
- Target audience: Technical decision makers, engineering teams

**Not a product** - this is a research/demonstration framework for evaluating simulation approaches and understanding microfluidic cooling physics.

---

## ✨ What You'll See

**Hero Slideshow** with 3 technical slides:
- **Slide 1**: Thermal Problem - Chip layout and temperature field
- **Slide 2**: Flow Solution & Coupling - Channel network and fluid temperatures
- **Slide 3**: Configuration Summary - Model parameters and specifications

**Technical Features**:
- Dark/light theme toggle for professional presentation
- MathML formula rendering (native browser support)
- Pre-generated visualizations for instant page load
- Technical specifications displayed alongside visualizations

**Simulation Results** (Intel Xeon CPU example):
- 250W power dissipation, 45×45mm chip
- 8 parallel microfluidic channels (300×300 μm)
- Max outlet temperature: ~56°C (44°C boiling margin)
- Solve time: ~6 seconds on standard CPU

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager ([install instructions](https://github.com/astral-sh/uv))

### Installation

```bash
# Clone repository
git clone <repo-url>
cd microfluidic-demo

# Install dependencies
uv sync
```

### Run the Dashboard

```bash
# Start Streamlit dashboard
./start_dashboard.sh

# Or use uv directly
uv run streamlit run Home.py
```

Open your browser to: **http://localhost:8501**

---

## 📁 Project Structure

```
microfluidic-demo/
├── Home.py                     # Main Streamlit app with slideshow
│
├── configs/
│   ├── simulation_defaults.yaml    # Grid resolution, segments, presets
│   ├── slideshow_content.yaml      # Technical content for slides
│   ├── chips/                      # Chip configuration files
│   │   └── intel_xeon_cpu.yaml
│   └── channels/                   # Channel pattern configurations
│       └── straight_parallel.yaml
│
├── src/mfdemo/
│   ├── models/
│   │   └── config.py           # Pydantic config models
│   ├── core/
│   │   ├── grid.py             # StructuredGrid (dimension-agnostic)
│   │   ├── heat_source.py      # HeatSource classes
│   │   └── channel_network.py  # ChannelNetwork (graph-based)
│   ├── solvers/
│   │   ├── thermal_solver.py   # 2D FDM solver
│   │   ├── microfluidic_solver.py   # 1D network flow
│   │   └── convective_coupling.py   # Thermal-fluidic coupling
│   ├── visualization/
│   │   └── plotly_viz.py       # Temperature/flow visualizations
│   ├── slideshow_component.py  # Streamlit slideshow component
│   └── config_loader.py        # Config utilities
│
├── scripts/
│   ├── generate_hero_slideshow.py  # Generate slideshow visualizations
│   └── optimize_outlet_temp.py     # Parameter sweep script
│
├── static/
│   └── hero_slideshow/         # Pre-generated slideshow images
│
├── start_dashboard.sh          # Launch script
├── Dockerfile                  # Production container
└── pyproject.toml              # Dependencies (uv)
```

---

## 📊 Technical Specifications

### Thermal Solver
- **Method**: 2D Finite Difference Method (FDM)
- **Discretization**: 360×360 spatial cells (0.125mm resolution)
- **Matrix**: Sparse (~1 million non-zero entries)
- **Solution**: Direct solver (scipy.sparse.linalg.spsolve)
- **Solve time**: ~5-6 seconds

### Flow Solver
- **Model**: 1D hydraulic network (Hagen-Poiseuille)
- **Channels**: 8 parallel, 300×300 μm cross-section
- **Operating conditions**: 15 kPa pressure drop, ~47 ml/min total flow
- **Flow regime**: Laminar (Re ≈ 330)
- **Solve time**: <1ms

### Coupling
- **Method**: Iterative Picard coupling with under-relaxation
- **Heat transfer**: Shah & London (1978) Nusselt correlation (aspect-ratio dependent)
- **Temperature-dependent properties**: Water viscosity (NIST), thermal conductivity
- **Temperature segments**: 200 per channel for smooth gradients
- **Convergence**: Typically 9-11 iterations

---

## 🔬 Physics & Assumptions

### Governing Equations

**Thermal (2D steady-state heat equation)**:
```
∇²T = -Q/k
```

**Flow (Hagen-Poiseuille)**:
```
ΔP = 128μLQ/(πD_h⁴)
```

**Convection (Shah & London Nusselt correlation)**:
```
Nu = f(Re, Pr, aspect_ratio)
```

### Key Assumptions

1. **2D Thermal Model**: Valid for thin dies (<1mm) with high through-plane conductivity
2. **Laminar Flow**: Re < 2300 for all channels (turbulent regime not modeled)
3. **Steady-State**: No transient thermal response
4. **Empirical Convection**: Uses Nusselt correlations rather than solving boundary layers
5. **Ideal Manifolds**: Assumes perfect flow distribution

### Validation

- Analytical comparison for 1D heat conduction: <0.01% error
- Grid convergence studies performed
- Energy conservation: Input heat matches removed + boundary losses

### Limitations

- Not a replacement for high-fidelity CFD
- Suitable for parametric design exploration
- Does not model:
  - 3D effects (temperature variation through thickness)
  - Turbulent flow
  - Transient thermal response
  - Non-ideal manifold flow distribution

---

## 🔧 Configuration

### Grid Resolution

Edit `configs/simulation_defaults.yaml`:

```yaml
grid:
  resolution_cells_per_mm: [8.0, 8.0]  # 360×360 cells for 45×45mm chip
  presets:
    low: [2.0, 2.0]      # 90×90 cells (fast)
    medium: [4.0, 4.0]   # 180×180 cells
    high: [8.0, 8.0]     # 360×360 cells (default)
    ultra: [16.0, 16.0]  # 720×720 cells (slow)
```

### Fluid Temperature Segments

```yaml
fluid_visualization:
  segments_per_channel: 200  # For smooth temperature gradients
  presets:
    coarse: 20
    medium: 50
    fine: 100
    ultra: 200  # default
```

---

## 🛠️ Development

### Regenerating Slideshow

```bash
# Generate new slideshow visualizations
uv run python scripts/generate_hero_slideshow.py
```

Images saved to `static/hero_slideshow/`

### Running Simulations Programmatically

```python
from pathlib import Path
from mfdemo.config_loader import load_config
from mfdemo.solvers.thermal_solver import ThermalSolver
from mfdemo.solvers.microfluidic_solver import MicrofluidicSolver
from mfdemo.solvers.convective_coupling import solve_coupled_system

# Load configurations
chip_config = load_config("configs/chips/intel_xeon_cpu.yaml")
channel_config = load_config("configs/channels/straight_parallel.yaml")

# Run coupled simulation (see scripts/generate_hero_slideshow.py for full example)
```

---

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build image
docker build -t microfluidic-demo .

# Run container
docker run -p 8501:80 microfluidic-demo
```

Access at: http://localhost:8501

### Production Deployment

The demo is deployed using Docker on Azure Container Instances with Cloudflare DNS. See `Dockerfile` for container configuration.

---

## 📄 License

MIT License

---

## 📧 About

**Purpose**: Technical demonstration of microfluidic cooling simulation methodology
**Development**: Built with AI assistance (Claude/Grok) over ~15-20 hours

This project demonstrates:
- Physics-based modeling of coupled thermal-fluidic systems
- Rapid prototyping with modern Python tools (Streamlit, uv)
- Docker containerization and cloud deployment
- Technical communication through interactive visualization

For technical details, see the simulation methodology section above and explore the codebase in `src/mfdemo/`.
