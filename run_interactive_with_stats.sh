#!/bin/bash
# Interactive K-Means Clustering with Statistics Tracker

echo "=========================================="
echo "K-Means Clustering with Statistics"
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

echo "Starting interactive K-Means with statistics tracking..."
echo ""
echo "This version tracks comprehensive statistics including:"
echo "  • Accuracy & Confusion Matrix"
echo "  • Cluster Separation & Variance"
echo "  • Silhouette Scores"
echo "  • Advanced Metrics"
echo ""
echo "All results will be saved to 'interactive_run_results/' folder"
echo ""

# Run the interactive program with statistics
$UV_CMD run --with numpy --with matplotlib --with scikit-learn python KMeansInteractiveWithStats.py

echo ""
echo "Session complete! Check 'interactive_run_results/' for:"
echo "  • JSON file with all raw data"
echo "  • CSV summary of key metrics"
echo "  • Text report"
echo ""
echo "To visualize results, run:"
echo "  $UV_CMD run --with numpy --with matplotlib python visualize_stats.py"
echo ""
