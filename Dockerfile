# Microfluidic Cooling Simulation Framework
# Production Dockerfile for standalone Streamlit dashboard
#
# This container runs the complete application including:
# - Thermal-fluidic simulation engine
# - Interactive Streamlit dashboard
# - Nginx reverse proxy (use Cloudflare DNS proxy for HTTPS)

FROM python:3.11-slim

# Metadata
LABEL description="Microfluidic Cooling Simulation Demo"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    chromium \
    chromium-driver \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome path for Plotly/Kaleido
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMIUM_PATH=/usr/bin/chromium

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files and source structure first (for better caching)
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install Python dependencies using uv
RUN uv sync --frozen

# Copy remaining source code and configuration
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY Home.py ./
COPY start_dashboard.sh ./

# Create directories for generated content
RUN mkdir -p /app/static/hero_slideshow \
    && mkdir -p /app/static/hero_slideshow/temp \
    && mkdir -p /app/data

# Pre-generate slideshow during build
RUN uv run python scripts/generate_hero_slideshow.py || echo "Warning: Slideshow generation failed, will retry on startup"

# Make startup script executable
RUN chmod +x start_dashboard.sh

# Configure nginx as reverse proxy for Streamlit
RUN echo 'server { \n\
    listen 80; \n\
    server_name _; \n\
    \n\
    location / { \n\
        proxy_pass http://127.0.0.1:8501; \n\
        proxy_http_version 1.1; \n\
        proxy_set_header Upgrade $http_upgrade; \n\
        proxy_set_header Connection "upgrade"; \n\
        proxy_set_header Host $host; \n\
        proxy_set_header X-Real-IP $remote_addr; \n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \n\
        proxy_set_header X-Forwarded-Proto $scheme; \n\
        proxy_read_timeout 86400; \n\
        proxy_buffering off; \n\
    } \n\
    \n\
    location /_stcore/health { \n\
        proxy_pass http://127.0.0.1:8501/_stcore/health; \n\
    } \n\
}' > /etc/nginx/sites-available/default

# Configure supervisor to run nginx and streamlit
RUN echo '[supervisord] \n\
nodaemon=true \n\
\n\
[program:nginx] \n\
command=/usr/sbin/nginx -g "daemon off;" \n\
autostart=true \n\
autorestart=true \n\
\n\
[program:streamlit] \n\
command=/root/.local/bin/uv run streamlit run /app/Home.py --server.port=8501 --server.address=127.0.0.1 --server.headless=true \n\
directory=/app \n\
autostart=true \n\
autorestart=true \n\
environment=STREAMLIT_SERVER_PORT="8501",STREAMLIT_SERVER_ADDRESS="127.0.0.1",STREAMLIT_SERVER_HEADLESS="true",STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"' > /etc/supervisor/conf.d/app.conf

# Expose port 80 (nginx)
EXPOSE 80

# Health check via nginx
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost/_stcore/health || exit 1

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=127.0.0.1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run supervisor (manages nginx and streamlit)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
