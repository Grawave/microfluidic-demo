#!/bin/bash
# Start the Microfluidic Cooling Simulation Framework

cd "$(dirname "$0")"

echo "========================================"
echo "Microfluidic Cooling Simulation Demo"
echo "========================================"
echo ""
echo "Technical demonstration app available at:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

# Use uv run to ensure correct environment
uv run streamlit run Home.py --server.port=8501 --server.address=localhost

echo ""
echo "App stopped."
