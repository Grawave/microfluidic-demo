#!/usr/bin/env python3
"""
Microfluidic Cooling Simulation Framework

Technical demonstration of simulation capabilities for thermal-fluidic
coupled analysis of microfluidic cooling systems in AI/GPU chips.
"""

import streamlit as st
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
import subprocess
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from mfdemo.config_loader import get_technical_subtitle
from mfdemo.slideshow_component import render_slideshow

def ensure_slideshow_exists():
    """
    Ensure slideshow assets exist, generate if missing.

    This runs once at startup to generate visualization assets.
    """
    slideshow_dir = Path("static/hero_slideshow")
    slideshow_meta_path = slideshow_dir / "slideshow_metadata.json"

    if not slideshow_meta_path.exists():
        st.info("🔧 Generating slideshow assets (first run, ~30 seconds)...")

        with st.spinner("Running simulation and generating visualizations..."):
            try:
                # Run slideshow generation script
                result = subprocess.run(
                    [sys.executable, "scripts/generate_hero_slideshow.py"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    st.success("✅ Slideshow generated successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to generate slideshow: {result.stderr}")
                    st.stop()
            except subprocess.TimeoutExpired:
                st.error("Slideshow generation timed out (>5 minutes)")
                st.stop()
            except Exception as e:
                st.error(f"Error generating slideshow: {e}")
                st.stop()

# Ensure assets exist before rendering
ensure_slideshow_exists()

st.set_page_config(
    page_title="Microfluidic Cooling Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Get current theme
current_theme = st.session_state.theme

# Theme-based styling
if current_theme == 'dark':
    bg_color = "#1a202c"
    text_color = "#e2e8f0"
    header_border = "#4a5568"
    section_bg = "#2d3748"
    section_border = "#4299e1"
    title_color = "#ffffff"
    subtitle_color = "#cbd5e0"
    method_title_color = "#4299e1"
else:
    bg_color = "#ffffff"
    text_color = "#333"
    header_border = "#e0e0e0"
    section_bg = "#f8f9fa"
    section_border = "#2c3e50"
    title_color = "#1a1a1a"
    subtitle_color = "#555"
    method_title_color = "#2c3e50"

st.markdown(f"""
<style>
    /* Theme-aware background */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color} !important;
    }}
    [data-testid="stHeader"] {{
        background-color: {bg_color} !important;
    }}
    section[data-testid="stMain"] {{
        background-color: {bg_color} !important;
    }}

    /* Base text color */
    .main {{
        color: {text_color};
    }}
</style>
""", unsafe_allow_html=True)

# Technical CSS - clean, professional, LARGE TEXT, theme-aware
st.markdown(f"""
<style>
    /* Hide sidebar and adjust layout */
    [data-testid="stSidebar"] {{
        display: none;
    }}

    .main-header {{
        text-align: center;
        padding: 2rem 0 1.5rem;
        border-bottom: 3px solid {header_border};
        margin-bottom: 2rem;
    }}

    .main-title {{
        font-size: 3rem;
        font-weight: 700;
        color: {title_color};
        margin-bottom: 0.75rem;
    }}

    .main-subtitle {{
        font-size: 1.4rem;
        color: {subtitle_color};
        font-weight: 400;
    }}

    .slideshow-container {{
        width: 100%;
        margin: 0 auto;
    }}

    .method-section {{
        background: {section_bg};
        border-left: 5px solid {section_border};
        padding: 2rem;
        margin: 2.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,{'0.3' if current_theme == 'dark' else '0.08'});
    }}

    .method-title {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {method_title_color};
        margin-bottom: 1.25rem;
    }}

    .method-list {{
        line-height: 2.5;
        color: {text_color};
        font-size: 1.15rem;
    }}

    .tech-spec {{
        font-family: 'Courier New', monospace;
        background: {'#374151' if current_theme == 'dark' else '#e9ecef'};
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: {'#fbbf24' if current_theme == 'dark' else '#d63384'};
        font-weight: 600;
        font-size: 1.05rem;
    }}

    /* Ensure good contrast */
    strong {{
        color: {title_color};
        font-weight: 700;
    }}

    /* Global button styling - larger and more visible */
    button[kind="primary"], button[kind="secondary"] {{
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }}

    /* MathML theme support */
    math {{
        color: {text_color};
    }}
</style>
""", unsafe_allow_html=True)

# Header
# Get technical subtitle from config
tech_subtitle = f"Coupled Thermal-Fluidic Analysis | {get_technical_subtitle()}"
st.markdown(f"""
<div class="main-header">
    <div class="main-title">Microfluidic Cooling Simulation Framework</div>
    <div class="main-subtitle">{tech_subtitle}</div>
</div>
""", unsafe_allow_html=True)

# Slideshow with 50/50 layout (image + technical content)
render_slideshow()


# MathML styling - MINIMAL to avoid breaking rendering
st.markdown("""
<style>
    /* MathML support - let browser handle rendering, just set size */
    math {
        font-size: 1.7em;
    }
</style>
""", unsafe_allow_html=True)

# Methodology section
st.markdown("""
<div class="method-section">
    <div class="method-title">Simulation Methodology & Assumptions</div>
    <div class="method-list">
        <strong>Thermal Solver:</strong><br>
        • Finite Difference Method (FDM) with <span class="tech-spec">360×360</span> spatial discretization<br>
        • Direct solver for steady-state heat equation: <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mo>∇</mo><mn>2</mn></msup><mi>T</mi><mo>=</mo><mo>−</mo><mfrac><mi>Q</mi><mi>k</mi></mfrac></mrow></math><br>
        • Convective boundary conditions at channel interfaces<br>
        • Temperature-dependent silicon conductivity: <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>k</mi><mo>(</mo><mi>T</mi><mo>)</mo><mo>=</mo><mn>150</mn><msup><mrow><mo>(</mo><mfrac><mn>300</mn><mi>T</mi></mfrac><mo>)</mo></mrow><mn>1.3</mn></msup></math> W/m·K<br>
        <br>
        <strong>Flow Solver:</strong><br>
        • 1D network flow model for parallel microchannels<br>
        • Poiseuille flow assumption (<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Re</mi><mo>&lt;</mo><mn>2300</mn></math>, laminar regime)<br>
        • Pressure drop: <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Δ</mi><mi>P</mi><mo>=</mo><mfrac><mrow><mn>128</mn><mi>μ</mi><mi>L</mi><mi>Q</mi></mrow><mrow><mi>π</mi><msubsup><mi>D</mi><mi>h</mi><mn>4</mn></msubsup></mrow></mfrac></mrow></math><br>
        • Channel geometry: 300×300 μm rectangular cross-section (8 channels)<br>
        • Temperature-dependent water viscosity (NIST correlation)<br>
        <br>
        <strong>Thermal-Fluidic Coupling:</strong><br>
        • Shah &amp; London (1978) Nusselt correlation with aspect-ratio dependence<br>
        • Full wetted perimeter heat transfer (all 4 channel walls)<br>
        • Per-segment fluid temperature tracking (<span class="tech-spec">200 segments/channel</span>)<br>
        • Energy balance: <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mrow><mi>d</mi><mi>T</mi></mrow><mrow><mi>d</mi><mi>x</mi></mrow></mfrac><mo>=</mo><mfrac><mi>q</mi><mrow><mover><mi>m</mi><mo>˙</mo></mover><msub><mi>c</mi><mi>p</mi></msub></mrow></mfrac></mrow></math> where <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi><mo>=</mo><mi>h</mi><mo>(</mo><msub><mi>T</mi><mi>chip</mi></msub><mo>−</mo><msub><mi>T</mi><mi>fluid</mi></msub><mo>)</mo><mi>P</mi></mrow></math><br>
        • Picard iteration with property updates each iteration (<span class="tech-spec">α=0.1</span>)<br>
        <br>
        <strong>Validation:</strong><br>
        • Analytical verification for 1D heat conduction problems<br>
        • Grid independence study (90×90 → 180×180 → 360×360 cells)<br>
        • Physical consistency checks (energy conservation, boundary conditions)<br>
    </div>
</div>
""", unsafe_allow_html=True)

# Target application
st.markdown("""
<div class="method-section">
    <div class="method-title">Use Case: AI/GPU Chip Thermal Management</div>
    <div class="method-list">
        <strong>Problem Statement:</strong> High-power density AI accelerators (250-500W, 45×45mm die)
        require advanced cooling beyond conventional heat sinks. Microfluidic cooling enables
        localized heat removal with minimal thermal resistance.<br>
        <br>
        <strong>Design Exploration:</strong> This framework enables rapid evaluation of channel layouts,
        flow rates, and thermal performance without expensive CFD simulations or physical prototyping.<br>
        <br>
        <strong>Computational Performance:</strong> Steady-state solution with iterative coupling in ~10-15 seconds on standard hardware
        (compared to hours for 3D Navier-Stokes CFD). Suitable for parametric studies and optimization loops.
    </div>
</div>
""", unsafe_allow_html=True)

