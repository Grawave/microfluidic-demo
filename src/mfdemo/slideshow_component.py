"""
Slideshow component with 50% image + 50% technical content layout.
"""

import streamlit as st
from pathlib import Path
import yaml
from PIL import Image
import re


def load_slideshow_content():
    """Load technical content for slides from YAML config."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "slideshow_content.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def render_latex_content(content: str) -> str:
    """
    Convert markdown content with LaTeX to HTML with KaTeX rendering.

    Supports:
    - Block equations: $$...$$
    - Inline equations: $...$
    """
    # Convert markdown bold to HTML
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)

    # Split by block equations first ($$...$$)
    parts = []
    current = content

    while '$$' in current:
        before, rest = current.split('$$', 1)
        if '$$' in rest:
            equation, after = rest.split('$$', 1)
            parts.append(('text', before))
            parts.append(('block_eq', equation.strip()))
            current = after
        else:
            parts.append(('text', current))
            break
    else:
        if current:
            parts.append(('text', current))

    # Now process inline equations in text parts ($...$)
    final_parts = []
    for part_type, part_content in parts:
        if part_type == 'block_eq':
            final_parts.append(('block_eq', part_content))
        else:
            # Process inline equations
            text_parts = []
            current_text = part_content
            while '$' in current_text:
                before, rest = current_text.split('$', 1)
                if '$' in rest:
                    equation, after = rest.split('$', 1)
                    text_parts.append(('text', before))
                    text_parts.append(('inline_eq', equation.strip()))
                    current_text = after
                else:
                    text_parts.append(('text', current_text))
                    break
            else:
                if current_text:
                    text_parts.append(('text', current_text))

            final_parts.extend(text_parts)

    # Build HTML
    html_parts = []
    for part_type, part_content in final_parts:
        if part_type == 'block_eq':
            # Block equation - centered on its own line
            html_parts.append(f'<div class="katex-block">\\[{part_content}\\]</div>')
        elif part_type == 'inline_eq':
            # Inline equation
            html_parts.append(f'\\({part_content}\\)')
        else:
            # Regular text - preserve line breaks
            html_parts.append(part_content.replace('\n', '<br>'))

    return ''.join(html_parts)


def render_slideshow():
    """
    Render slideshow with 50/50 image + technical content layout.

    Uses Streamlit session state for slide navigation.
    """
    # MathML and theme styling - MINIMAL CSS to avoid breaking rendering
    st.markdown("""
    <style>
        /* MathML styling - let browser handle defaults, just size */
        math {
            font-size: 1.8em;
        }
        /* Larger for centered equations */
        .equation-display math {
            font-size: 2.2em;
        }
        /* Table styling for config display */
        table {
            font-size: 1rem;
        }
        table td {
            vertical-align: top;
        }

        /* Theme-aware colors */
        .slide-content-box {
            transition: background-color 0.3s, color 0.3s, border-color 0.3s;
        }

        /* Light theme (default) */
        .slide-content-box.light {
            background: #f8f9fa;
            color: #333;
            border-left: 5px solid #2c3e50;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .slide-content-box.light strong {
            color: #1a1a1a;
        }

        /* Dark theme */
        .slide-content-box.dark {
            background: transparent;
            color: #e2e8f0;
            border: none;
            box-shadow: none;
        }
        .slide-content-box.dark strong {
            color: #ffffff;
        }
        .slide-content-box.dark math {
            color: #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'slide_index' not in st.session_state:
        st.session_state.slide_index = 0
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'  # Default to dark theme

    # Load content
    content = load_slideshow_content()
    slides = content['slides']
    slideshow_dir = Path("static/hero_slideshow")

    # Get current theme
    theme = st.session_state.theme

    # Slide file mapping (now 3 slides)
    slide_files = [
        "slide_1_chip_layout_and_temperature.png",
        "slide_2_channels_and_coupling.png",
        "slide_3_config.png"
    ]

    current_slide = slides[st.session_state.slide_index]
    current_image = slideshow_dir / slide_files[st.session_state.slide_index]

    # Add custom button styling for larger buttons
    st.markdown("""
    <style>
        div[data-testid="column"] button {
            font-size: 1.1rem !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Navigation controls (theme toggle hidden from users)
    col1, col2, col3 = st.columns([1, 3, 1])

    # Theme toggle hidden but kept functional for development
    # Uncomment below to show theme toggle:
    # theme_col, col1, col2, col3 = st.columns([0.7, 1, 3, 1])
    # with theme_col:
    #     theme_icon = "☀️" if theme == "dark" else "🌙"
    #     if st.button(theme_icon, use_container_width=True, key="theme_toggle", help="Toggle theme"):
    #         st.session_state.theme = "light" if theme == "dark" else "dark"
    #         st.rerun()

    with col1:
        if st.button("◀ Prev", use_container_width=True, disabled=st.session_state.slide_index == 0):
            st.session_state.slide_index = max(0, st.session_state.slide_index - 1)
            st.rerun()

    with col2:
        # Slide indicator dots (larger and more visible)
        dots_html = ""
        for i in range(len(slides)):
            active = "background: #2c3e50; width: 40px;" if i == st.session_state.slide_index else "background: #ccc;"
            dots_html += f'<div style="display: inline-block; width: 14px; height: 14px; border-radius: 50%; margin: 0 6px; {active}"></div>'
        st.markdown(f'<div style="text-align: center; padding: 0.5rem 0;">{dots_html}</div>', unsafe_allow_html=True)

    with col3:
        if st.button("Next ▶", use_container_width=True, disabled=st.session_state.slide_index == len(slides) - 1):
            st.session_state.slide_index = min(len(slides) - 1, st.session_state.slide_index + 1)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Main content: 50% image + 50% technical content
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Image title (no visible container)
        st.markdown(f"""
        <h4 style="color: #cdd6f4; margin: 0 0 0.5rem 0; font-size: 1.2rem; font-weight: 600;">{current_slide['visualization']}</h4>
        """, unsafe_allow_html=True)

        if current_image.exists():
            image = Image.open(current_image)
            st.image(image, width="stretch")
        else:
            st.warning(f"Image not found: {current_image.name}")

    with col_right:
        # Technical content with MathML rendering (LARGE TEXT, theme-aware)
        title_color = "#2c3e50" if theme == "light" else "#ffffff"
        content_html = current_slide.get('content_html', current_slide.get('content', ''))

        st.markdown(f"""<div class="slide-content-box {theme}" style="padding: 2rem; border-radius: 10px; min-height: 450px;">
<h3 style="color: {title_color}; margin-top: 0; font-size: 1.6rem; font-weight: 700;">{current_slide['title']}</h3>
<div style="line-height: 2.2; font-size: 1.15rem;">
{content_html}
</div>
</div>""", unsafe_allow_html=True)

    # Slide counter
    st.markdown(f"""
    <div style="text-align: center; color: #666; margin-top: 1.5rem; font-size: 1.1rem; font-weight: 500;">
        Slide {st.session_state.slide_index + 1} of {len(slides)}
    </div>
    """, unsafe_allow_html=True)
