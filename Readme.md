# K-Means Algorithm - Interactive & Static Visualization

This project provides both static and **interactive** K-Means clustering visualizations. It creates three distinct groups of 2D vectors with minimal within-group variance and maximal between-group variance, perfect for demonstrating K-Means clustering.

## Features

### Static Version (`KMeansAlgoExercise.py`)
- **Data Generation**: Creates 2000 samples for each of the three data groups.
- **Optimized Clusters**: Minimal within-group variance (tight clusters) with maximal between-group variance (well-separated).
- **K-Means Clustering**: Automatically applies K-Means algorithm to find cluster centers.
- **Visualization**: Plots all data points colored by cluster assignment.
- **Clear Markers**:
    - K-Means centers marked with black 'X'
    - Original group means marked with colored squares (Red, Green, Blue)
    - Semi-transparent plotting shows cluster density
- **Educational**: Heavily commented code for learning.

### Interactive Version (`KMeansInteractive.py`) **NEW!**
- **Drag-and-Drop Interface**: Move entire groups by clicking and dragging anywhere on a group's cloud of points or its center marker.
- **Blow Up Buttons**: Interactive buttons overlaid at each group center to scatter that group randomly - generates random mean and variance.
- **Real-time Updates**: Data regenerates automatically after any action.
- **Live K-Means**: K-Means re-clusters after each move or blow up.
- **Smart Visualization**:
    - Samples keep their original color (red/green/blue)
    - Reassigned samples get a colored ring showing their new cluster
    - Accuracy counter shows how many samples were reassigned
- **Visual Feedback**: Markers highlight when selected, comprehensive instructions on screen.
- **Experiment Freely**: Test how cluster separation and overlap affects K-Means accuracy.
- **Detailed Guide**: See `INTERACTIVE_GUIDE.md` for comprehensive usage instructions and experiment ideas.

### Interactive Version with Statistics (`KMeansInteractiveWithStats.py`) **ENHANCED!**
All features from the standard interactive version PLUS:
- **Comprehensive Statistics Tracking**: Automatically calculates and displays detailed metrics after each action:
    - **Accuracy Metrics**: Overall and per-group accuracy, confusion matrix, misclassification rates
    - **Separation Metrics**: Inter-cluster distances, K-Means center displacement
    - **Variance Analysis**: Within-group and between-group variance, variance ratios
    - **Silhouette Scores**: Overall and per-group silhouette analysis
    - **Advanced Metrics**: Davies-Bouldin Index, Calinski-Harabasz Score
    - **Convergence Data**: Inertia (SSE) and iteration counts
- **Automatic Export**: Saves all statistics to `interactive_run_results/` folder in three formats:
    - **JSON**: Complete raw data for programmatic analysis
    - **CSV**: Summary metrics for spreadsheet analysis
    - **Text Report**: Human-readable session summary
- **Visualization Tool** (`visualize_stats.py`): Generate plots showing metric evolution over time
- **Perfect for Research**: Track exactly when and why K-Means fails

## Requirements

- Python 3
- `numpy`
- `matplotlib`
- `scikit-learn`
- `uv` (recommended package manager)

## Installation

### Option 1: Using uv (Recommended)
`uv` automatically manages dependencies:

```sh
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# No additional installation needed - uv handles everything!
```

### Option 2: Using pip
```sh
pip install -r requirements.txt
```

## Usage

### Static Visualization
Run the static K-Means clustering visualization:

```sh
# Using uv (recommended)
uv run --with numpy --with matplotlib --with scikit-learn python KMeansAlgoExercise.py

# Or using python directly (if dependencies installed)
python KMeansAlgoExercise.py
```

This generates a PNG file (`KMeans_Clustering_Results.png`) and displays the plot.

### Interactive GUI (Standard)
Run the interactive drag-and-drop visualization:

```sh
# Using the provided script (easiest)
./run_interactive.sh

# Or using uv directly
uv run --with numpy --with matplotlib --with scikit-learn python KMeansInteractive.py

# Or using python directly
python KMeansInteractive.py
```

### Interactive GUI with Statistics (Enhanced)
Run the enhanced version with comprehensive statistics tracking:

```sh
# Using the provided script (easiest)
./run_interactive_with_stats.sh

# Or using uv directly
uv run --with numpy --with matplotlib --with scikit-learn python KMeansInteractiveWithStats.py

# Or using python directly
python KMeansInteractiveWithStats.py
```

### Visualizing Session Statistics
After running the statistics-enabled version, visualize your results:

```sh
# Visualize the latest session
uv run --with numpy --with matplotlib python visualize_stats.py

# List all available sessions
uv run --with numpy --with matplotlib python visualize_stats.py list

# Visualize a specific session
uv run --with numpy --with matplotlib python visualize_stats.py interactive_run_results/session_YYYYMMDD_HHMMSS.json
```

**How to use the Interactive GUI:**
1. The window opens with three colored square markers (Red, Green, Blue) representing group centers
2. **Drag groups**: Click and drag anywhere on a group's cloud of points OR its center marker to move the entire group
   - The marker enlarges when selected
   - Release to regenerate data at the new position
3. **Blow up groups**: Click the "💥 Blow Up" buttons overlaid at each group center to scatter that group randomly
   - Generates random mean (random position within plot area)
   - Generates random variance (1.0-3.0 range for scattered effect)
   - Console shows detailed information about the new configuration
4. After each action, data regenerates and K-Means re-clusters automatically
5. **Observe reassignments**:
   - Samples keep their original color (fill)
   - Reassigned samples show a colored ring (edge) indicating their new cluster
   - Title shows reassignment count and accuracy (% of samples correctly clustered)
6. Experiment with overlap and separation to see how K-Means performs
7. Close the window to see a summary of final positions and covariances in the console

**Note for WSL Users:** You need an X server (VcXsrv, X410, or WSLg) to display the GUI. Alternatively, you can access the GUI through Windows by running the script from Windows PowerShell:
```powershell
python KMeansInteractive.py
```

## Expected Output

### Static Version
Displays a plot titled "K-Means Clustering of Gaussian Data" with:
- Three well-separated, tight clusters (minimal overlap)
- K-Means centers (black X markers) aligned with original means
- Colored square markers showing original group centers
- Saved as `KMeans_Clustering_Results.png`

### Interactive Version (Standard)
Opens an interactive window titled "Interactive K-Means Clustering" with:
- Draggable colored square markers for each group center
- "💥 Blow Up" buttons overlaid at each group center
- Real-time visualization updates as you move markers or blow up groups
- Instructions panel at the top showing how to interact
- Reassignment visualization with colored rings on misclassified samples
- Accuracy percentage displayed in the title
- Console output showing move/blow up details and final summary when you close the window

### Interactive Version with Statistics
Same as standard version, plus:
- **Detailed Console Statistics** after each action showing:
  - Accuracy metrics and confusion matrix
  - Cluster separation distances
  - Variance analysis
  - Silhouette scores
  - Advanced clustering metrics
  - Convergence information
- **Automatic File Export** to `interactive_run_results/`:
  - JSON file with complete session data
  - CSV summary for spreadsheet analysis
  - Human-readable text report
- **Visualization Tool Output**:
  - 9-panel plot showing metric evolution over time
  - Confusion matrix evolution plot
  - Accuracy vs. separation scatter plot

*Note: The exact positions of points will vary on each run due to the random nature of data generation.*

## Project Files

### Main Programs
- `KMeansAlgoExercise.py` - Static K-Means clustering visualization
- `KMeansInteractive.py` - Interactive GUI with draggable group centers and blow up functionality
- `KMeansInteractiveWithStats.py` - **Enhanced interactive GUI with comprehensive statistics tracking**
- `visualize_stats.py` - **Statistics visualization tool for analyzing session data**

### Scripts
- `run_interactive.sh` - Convenient script to run the standard interactive version
- `run_interactive_with_stats.sh` - **Script to run the enhanced statistics version**

### Documentation
- `Readme.md` - This file
- `INTERACTIVE_GUIDE.md` - Comprehensive guide for the interactive version with tips and experiment ideas
- `STATISTICS_GUIDE.md` - **Complete guide to analyzing K-Means failures using statistics**

### Configuration
- `requirements.txt` - Python package dependencies

### Output
- `KMeans_Clustering_Results.png` - Output from static version (generated when run)
- `interactive_run_results/` - **Directory for storing session statistics and results**
  - `session_YYYYMMDD_HHMMSS.json` - Complete session data
  - `session_YYYYMMDD_HHMMSS_summary.csv` - Key metrics summary
  - `session_YYYYMMDD_HHMMSS_report.txt` - Human-readable report
  - `session_YYYYMMDD_HHMMSS_visualization.png` - Metric evolution plots
  - `session_YYYYMMDD_HHMMSS_confusion_matrices.png` - Confusion matrix evolution

## Educational Value

This project demonstrates:
- **Cluster Separation Impact**: See how distance between groups affects K-Means accuracy
- **Variance Effect**: Observe how scattered (high variance) vs. tight (low variance) clusters affect clustering
- **Reassignment Visualization**: Clearly see which samples get reassigned when clusters overlap
- **Interactive Learning**: Experiment with different configurations in real-time
- **K-Means Limitations**: Discover scenarios where K-Means struggles (overlapping clusters)
- **Data Preprocessing Importance**: Understand why scaling and normalization matter

## Key Features Explained

### Blow Up Functionality
When you click a "Blow Up" button:
1. The group's mean is randomly relocated within the plot area
2. Variance is increased (1.0-3.0 range) to create a scattered, spread-out cluster
3. Data regenerates with the new parameters
4. K-Means attempts to re-cluster the scattered data

### Reassignment Visualization
- **Original Color (fill)**: Shows which group a sample originally belonged to
- **Ring Color (edge)**: Shows which cluster K-Means assigned it to
- **No Ring**: Sample was correctly classified (original group matches K-Means cluster)
- **Colored Ring**: Sample was reassigned to a different cluster

This dual-color system makes it easy to spot misclassifications and understand K-Means behavior under different conditions.

## Analyzing K-Means Algorithm Failures

The enhanced statistics version (`KMeansInteractiveWithStats.py`) provides powerful tools for understanding when and why K-Means fails. Here are the key metrics to analyze:

### 1. **Accuracy & Misclassification**
- **Overall Accuracy**: Percentage of correctly clustered samples
- **Per-Group Accuracy**: Identifies which groups are most affected
- **Confusion Matrix**: Shows which groups get confused with each other
- **Use Case**: Track how accuracy decreases as clusters overlap

### 2. **Cluster Separation**
- **Inter-Cluster Distances**: Distance between each pair of group centers
- **Minimum Separation**: Shortest distance between any two groups
- **Center Displacement**: How far K-Means centers are from true centers
- **Use Case**: Determine the separation threshold where failures begin

### 3. **Variance Analysis**
- **Within-Group Variance**: How scattered each cluster is
- **Between-Group Variance**: How separated the groups are
- **Variance Ratio**: Higher ratio = better clustering conditions
- **Use Case**: Understand how "blown up" (high variance) clusters affect results

### 4. **Silhouette Scores**
- **Overall Score**: -1 (wrong cluster) to +1 (perfect clustering)
- **Per-Group Scores**: Identify problematic groups
- **Negative Samples**: Samples likely misclassified
- **Use Case**: Quantify cluster quality and identify boundary samples

### 5. **Advanced Metrics**
- **Davies-Bouldin Index**: Lower is better (measures separation vs compactness)
- **Calinski-Harabasz Score**: Higher is better (between/within cluster dispersion ratio)
- **Use Case**: Compare different clustering configurations objectively

### 6. **Convergence Analysis**
- **Inertia (SSE)**: Sum of squared errors - higher means looser clusters
- **Iterations**: More iterations suggests complex/ambiguous configuration
- **Use Case**: Identify when K-Means struggles to find stable centers

### Example Analysis Workflow

1. **Start with Well-Separated Clusters**
   - Initial accuracy: ~100%
   - High variance ratio
   - High silhouette scores

2. **Gradually Increase Overlap**
   - Drag groups closer together OR blow up groups
   - Watch accuracy decrease
   - Observe which metrics correlate with failures

3. **Analyze the Thresholds**
   - At what minimum separation does accuracy drop below 95%?
   - What variance ratio indicates problems?
   - Which silhouette score signals poor clustering?

4. **Identify Failure Patterns**
   - Look at confusion matrix: which pairs get confused?
   - Check center displacement: is K-Means finding wrong centers?
   - Compare scenarios using the visualization tool

### Typical Findings

- **Separation < 3.0**: Accuracy often drops below 90%
- **Variance Ratio < 5.0**: Clusters may overlap significantly
- **Silhouette < 0.5**: Boundary samples are ambiguous
- **High Blown-Up Variance (>2.0)**: Scattered samples cause reassignments even when centers are far apart

## Additional Resources

### For Interactive Usage
See **`INTERACTIVE_GUIDE.md`** for comprehensive documentation including:
- Step-by-step experiment ideas
- Understanding console output
- Technical details about the algorithms
- Troubleshooting tips

### For Statistics Analysis
See **`STATISTICS_GUIDE.md`** for complete statistics analysis documentation including:
- Understanding all metrics (accuracy, separation, variance, silhouette, etc.)
- Interpreting console output and exported files
- Using the visualization tool
- Experimental workflows and analysis patterns
- Finding critical thresholds for K-Means failures
- Tips for publication-quality analysis
