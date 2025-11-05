#!/bin/bash
# Interactive K-Means Clustering GUI Runner

echo "=========================================="
echo "Interactive K-Means Clustering GUI"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null && ! command -v ~/.local/bin/uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Determine uv path
if command -v uv &> /dev/null; then
    UV_CMD="uv"
else
    UV_CMD="~/.local/bin/uv"
fi

echo "Starting interactive K-Means clustering..."
echo ""
echo "Instructions:"
echo "  • Click and drag the colored squares to move group centers"
echo "  • Data will regenerate automatically when you release"
echo "  • K-Means will re-cluster the new data"
echo "  • Close the window to exit"
echo ""

# Run the interactive program
$UV_CMD run --with numpy --with matplotlib --with scikit-learn python KMeansInteractive.py

echo ""
echo "Program exited."
