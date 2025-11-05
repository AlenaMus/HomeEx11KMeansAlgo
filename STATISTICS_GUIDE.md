# K-Means Statistics Analysis Guide

## Quick Start

```bash
# Run the enhanced interactive version with statistics
./run_interactive_with_stats.sh

# After your session, visualize the results
uv run --with numpy --with matplotlib python visualize_stats.py
```

## Understanding the Statistics

### Console Output

After each action (drag or blow up), you'll see detailed statistics printed to the console:

```
======================================================================
ACTION #1 STATISTICS
======================================================================

📊 ACCURACY METRICS:
  Overall Accuracy: 98.45%
  Correctly Classified: 5907/6000
  Misclassified: 93

  Per-Group Accuracy:
    Group 1: 97.20% (1944/2000)
    Group 2: 99.15% (1983/2000)
    Group 3: 99.00% (1980/2000)

  Confusion Matrix:
         Predicted→  G1    G2    G3
    True G1          1944   42    14
    True G2            17 1983     0
    True G3            20    0  1980

📏 SEPARATION METRICS:
  Inter-Cluster Distances:
    group_1_to_2: 8.54
    group_1_to_3: 12.73
    group_2_to_3: 5.19
  Min/Avg/Max Separation: 5.19 / 8.82 / 12.73
  Avg K-Means Center Displacement: 0.15

📈 VARIANCE METRICS:
  Within-Group Variance:
    Group 1: 0.600
    Group 2: 0.600
    Group 3: 0.600
  Between-Group Variance: 356.333
  Variance Ratio (between/within): 594.222

🎯 SILHOUETTE ANALYSIS:
  Overall Score: 0.823 (range: -1 to 1, higher is better)
  Per-Group Scores:
    Group 1: 0.801
    Group 2: 0.856
    Group 3: 0.812
  Negative Silhouette Samples: 23 (0.38%)

🔬 ADVANCED METRICS:
  Davies-Bouldin Index: 0.234 (lower is better)
  Calinski-Harabasz Score: 28945.67 (higher is better)

⚙️  CONVERGENCE:
  Total Inertia (SSE): 3598.45
  Iterations to Converge: 5
```

## Exported Files

### 1. JSON File (Complete Data)
**File**: `interactive_run_results/session_YYYYMMDD_HHMMSS.json`

Contains complete raw data for each action:
- All metrics in structured format
- Group configurations (centers and covariances)
- K-Means cluster centers
- Cluster-to-group mappings

**Use for**: Programmatic analysis, custom visualizations, machine learning

### 2. CSV Summary
**File**: `interactive_run_results/session_YYYYMMDD_HHMMSS_summary.csv`

Key metrics in spreadsheet format:

| Action | Timestamp | Overall_Accuracy | Misclassified | Min_Separation | ... |
|--------|-----------|------------------|---------------|----------------|-----|
| 0      | ...       | 100.00           | 0             | 14.14          | ... |
| 1      | ...       | 98.45            | 93            | 5.19           | ... |

**Use for**: Excel/spreadsheet analysis, creating charts, sharing results

### 3. Text Report
**File**: `interactive_run_results/session_YYYYMMDD_HHMMSS_report.txt`

Human-readable summary:
- Session overview
- Overall session statistics
- Detailed action history

**Use for**: Quick review, documentation, sharing findings

## Visualization Tool

### Basic Usage

```bash
# Visualize latest session
python visualize_stats.py

# List all sessions
python visualize_stats.py list

# Visualize specific session
python visualize_stats.py interactive_run_results/session_20250101_123456.json
```

### Generated Plots

#### Main Visualization (9-panel plot)
**File**: `session_YYYYMMDD_HHMMSS_visualization.png`

1. **Overall Accuracy Over Time**: Track how accuracy changes with each action
2. **Per-Group Accuracy**: See which groups are most affected
3. **Misclassified Samples**: Count of misclassified samples
4. **Cluster Separation**: Min, avg, and max distances between groups
5. **Center Displacement**: How far K-Means centers drift from true centers
6. **Variance Ratio**: Between-group vs within-group variance
7. **Silhouette Score**: Overall clustering quality metric
8. **Inertia (SSE)**: Sum of squared errors
9. **Accuracy vs Separation**: Scatter plot showing correlation

#### Confusion Matrix Evolution
**File**: `session_YYYYMMDD_HHMMSS_confusion_matrices.png`

Shows confusion matrices at key points (first, middle, last actions) to visualize how group confusion evolves.

## Key Metrics Explained

### Accuracy Metrics

**Overall Accuracy**
- Percentage of samples correctly assigned to their original group
- 100% = perfect clustering
- <90% = significant overlap/confusion

**Confusion Matrix**
- Rows: True group
- Columns: Predicted cluster
- Diagonal values should be high (correct assignments)
- Off-diagonal values show misassignments

### Separation Metrics

**Inter-Cluster Distances**
- Euclidean distance between group centers
- Higher is better (more separated)
- **Critical threshold**: ~3.0 - below this, expect failures

**Center Displacement**
- Distance between K-Means computed center and true center
- Low (<0.5) = K-Means found correct centers
- High (>2.0) = K-Means struggling to find true centers

### Variance Metrics

**Within-Group Variance**
- Total variance = trace of covariance matrix
- 0.6 = tight cluster (original)
- 1.0-3.0 = "blown up" scattered cluster
- Higher = more spread out

**Variance Ratio** (Between/Within)
- Higher is better for clustering
- >100 = well-separated clusters
- <10 = overlapping clusters
- **Critical threshold**: ~5.0 - below this, expect problems

### Silhouette Score

**Range**: -1 to +1
- **+1**: Perfect clustering (far from other clusters)
- **0**: On cluster boundary (ambiguous)
- **-1**: Assigned to wrong cluster

**Interpretation**:
- \>0.7: Strong clustering
- 0.5-0.7: Reasonable clustering
- 0.25-0.5: Weak clustering
- <0.25: No substantial structure

**Per-Group Scores**
- Identify which groups are well-separated
- Low score = that group overlaps with others

**Negative Samples**
- Samples with negative silhouette scores
- Likely misclassified
- Should be near cluster boundaries

### Advanced Metrics

**Davies-Bouldin Index**
- Ratio of within-cluster to between-cluster distances
- **Lower is better**
- <1.0 = good clustering
- \>2.0 = poor clustering

**Calinski-Harabasz Score**
- Ratio of between-cluster to within-cluster dispersion
- **Higher is better**
- >1000 = good clustering
- Higher values indicate denser, better separated clusters

### Convergence Metrics

**Inertia (SSE - Sum of Squared Errors)**
- Sum of squared distances from samples to their cluster center
- Lower generally means tighter clusters
- Compare across different configurations

**Iterations to Converge**
- Number of iterations K-Means took to stabilize
- 3-5 iterations = typical
- 10+ iterations = complex/ambiguous configuration

## Experimental Workflows

### Experiment 1: Find the Separation Threshold

**Goal**: Determine minimum separation for accurate clustering

**Steps**:
1. Start with well-separated groups
2. Gradually drag one group closer to another
3. Record accuracy at each step
4. Identify separation value where accuracy drops below 95%

**Analysis**:
- Plot: Accuracy vs Min_Separation
- Look for the "knee" in the curve
- Typical finding: Separation < 3.0 causes failures

### Experiment 2: Variance Impact Study

**Goal**: Understand how cluster spread affects K-Means

**Steps**:
1. Keep groups well-separated
2. Blow up one group at a time
3. Record variance and accuracy for each
4. Note how variance affects reassignments

**Analysis**:
- Plot: Variance vs Misclassified
- Check silhouette scores
- Typical finding: Variance > 2.0 causes scattered reassignments

### Experiment 3: Three-Way Overlap

**Goal**: Test K-Means with heavily overlapping clusters

**Steps**:
1. Move all three groups close together
2. Observe confusion matrix
3. Check which pairs get most confused
4. Note K-Means center displacement

**Analysis**:
- Examine confusion matrix patterns
- High center displacement indicates K-Means finds arbitrary boundaries
- Silhouette scores will be low (<0.3)

### Experiment 4: Asymmetric Configuration

**Goal**: Test with one far group, two close groups

**Steps**:
1. Keep two groups close together
2. Keep third group far away
3. Observe if far group always gets perfect accuracy
4. Check if close groups share their errors

**Analysis**:
- Per-group accuracy comparison
- Far group should have ~100% accuracy
- Close groups should have similar error rates

## Tips for Analysis

### 1. Track Trends Over Time
- Don't just look at final values
- Watch how metrics evolve with each action
- Use the visualization tool to see patterns

### 2. Compare Multiple Metrics
- No single metric tells the whole story
- Correlation between metrics indicates robustness
- Example: Low accuracy + low silhouette + high separation = specific boundary problem

### 3. Focus on Critical Thresholds
- Identify values where failures start
- Document these for future reference
- Use them to set preprocessing requirements

### 4. Use Confusion Matrix
- Tells you exactly which groups get confused
- Helps identify if it's a global or local problem
- Pattern analysis: symmetric confusion vs one-way confusion

### 5. Validate with Silhouette
- Confirms if clustering quality matches accuracy
- Identifies boundary samples
- Helps distinguish bad data from bad algorithm

## Common Patterns and What They Mean

### Pattern 1: High Accuracy, Low Silhouette
**Meaning**: Clusters are correct but have fuzzy boundaries
**Action**: Consider using different distance metric or soft clustering

### Pattern 2: Low Accuracy, High Center Displacement
**Meaning**: K-Means finding completely wrong centers
**Action**: Clusters likely heavily overlapped - preprocessing needed

### Pattern 3: Accuracy Drops Suddenly
**Meaning**: Crossed a critical separation threshold
**Action**: Document this threshold for future data requirements

### Pattern 4: One Group Low Accuracy, Others High
**Meaning**: That specific group overlaps with another
**Action**: Check confusion matrix to see which pair

### Pattern 5: High Iterations, High Inertia
**Meaning**: K-Means struggling to converge
**Action**: Clusters ambiguous - consider increasing K or different algorithm

## Exporting for Publications/Reports

### Create Publication-Quality Figures

1. Run your experiments
2. Use visualization tool to generate plots
3. Plots are saved at 150 DPI, suitable for documents

### Generate Summary Statistics

From CSV file:
- Import into Excel/Google Sheets
- Create pivot tables
- Generate custom charts
- Calculate correlation coefficients

### Write Analysis Reports

Use the text report as a starting point:
- Add your interpretations
- Include threshold findings
- Document experiment conclusions

## Troubleshooting

### "No session files found"
- Make sure you ran `KMeansInteractiveWithStats.py` first
- Check that `interactive_run_results/` directory exists
- Verify JSON files were created

### Visualization shows flat lines
- Session had only one action
- Perform more moves/blow ups to see trends
- Need at least 3-4 actions for meaningful plots

### Statistics seem incorrect
- Check if clusters overlap completely
- Verify data was regenerated after actions
- Some metrics undefined for pathological cases

## Further Reading

- **INTERACTIVE_GUIDE.md**: Detailed GUI usage instructions
- **README.md**: Overall project documentation
- Scikit-learn documentation on clustering metrics
- Original K-Means algorithm papers

## Quick Reference: When to Use Each Metric

| Metric | Use When You Want To... |
|--------|-------------------------|
| Overall Accuracy | Get a simple performance number |
| Confusion Matrix | See which specific groups get confused |
| Min Separation | Find threshold for cluster overlap |
| Variance Ratio | Assess if clusters are distinguishable |
| Silhouette Score | Validate clustering quality |
| Davies-Bouldin | Compare different clustering configurations |
| Calinski-Harabasz | Determine optimal number of clusters |
| Inertia | Measure cluster compactness |
| Center Displacement | Check if K-Means finds correct centers |

---

**Happy Analyzing! 🎯📊**
