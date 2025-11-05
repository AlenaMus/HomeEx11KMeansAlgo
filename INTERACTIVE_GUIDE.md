# Interactive K-Means Clustering - Quick Reference Guide

## Overview
This interactive visualization allows you to experiment with K-Means clustering in real-time by manipulating group positions and variance.

## Interface Elements

### Main Plot Area
- **Colored Dots**: Data points (2000 per group, 6000 total)
- **Colored Squares (Red/Green/Blue)**: Draggable group centers
- **Black X Markers**: K-Means computed cluster centers
- **Colored Rings**: Indicate samples reassigned to different clusters

### Control Buttons (Bottom of Window)
- **"Blow Up Group 1" (Red)**: Scatter Group 1 randomly
- **"Blow Up Group 2" (Green)**: Scatter Group 2 randomly
- **"Blow Up Group 3" (Blue)**: Scatter Group 3 randomly

## How to Use

### 1. Drag Group Centers
**Action**: Click and hold a colored square, then drag to new position
**Effect**:
- Group relocates to new position
- Data regenerates with tight variance (0.3)
- K-Means re-clusters all data

**Use Case**: Test how changing cluster separation affects accuracy

### 2. Blow Up Groups
**Action**: Click any "Blow Up" button
**Effect**:
- Group moves to random position
- Variance increases (1.0-3.0 range) - creates scattered cluster
- Data regenerates
- K-Means re-clusters all data

**Use Case**: Test how cluster overlap and variance affect K-Means performance

## Understanding the Visualization

### Color Coding System

#### Without Reassignment (Correct Clustering)
```
Sample: ● (solid color, no ring)
```
- Original group matches K-Means cluster
- K-Means correctly identified the sample's group

#### With Reassignment (Misclassification)
```
Sample: ◉ (original color fill + colored ring)
```
- **Fill Color**: Original group (red/green/blue)
- **Ring Color**: New cluster assignment by K-Means
- Indicates K-Means assigned this sample to a different cluster

### Title Information
```
Interactive K-Means Clustering
Reassigned: 245/6000 (Accuracy: 95.9%)
```
- **Reassigned**: Number of samples assigned to different cluster
- **Total**: Total number of samples (always 6000)
- **Accuracy**: Percentage of correctly classified samples

## Experiment Ideas

### 1. Test Cluster Separation
1. Start with initial tight clusters (high accuracy ~100%)
2. Drag one group closer to another
3. Observe how accuracy decreases as clusters approach
4. Watch rings appear on boundary samples

### 2. Test Variance Impact
1. Click "Blow Up" on one group
2. Observe scattered samples with increased variance
3. Note how many samples get reassigned
4. Compare accuracy before and after

### 3. Test Overlap Scenarios
1. Blow up all three groups
2. Some groups may heavily overlap
3. Observe how K-Means handles ambiguous boundary cases
4. Notice which samples get reassigned (usually boundary samples)

### 4. Test K-Means Convergence
1. Create heavily overlapping clusters
2. Watch K-Means find cluster centers
3. Note that centers may not match original group means
4. Understand K-Means finds "best fit" not "perfect fit"

## Console Output

### During Interaction
```bash
============================================================
BLOWING UP GROUP 1!
============================================================
New mean: [5.23, 7.89]
New covariance:
[[2.34 0.12]
 [0.12 1.87]]
Regenerating data and re-clustering...
Blow up complete!
```

### When Dragging
```bash
Group 2 moved to: [8.45, 9.12]
```

### On Exit
```bash
============================================================
SESSION SUMMARY
============================================================

Final group positions:
  Group 1: [3.25, 4.67]
  Group 2: [8.45, 9.12]
  Group 3: [1.23, 8.90]

Final group covariances:
  Group 1:
    [[2.34 0.12]
     [0.12 1.87]]
  Group 2:
    [[0.3 0. ]
     [0.  0.3]]
  Group 3:
    [[1.56 -0.23]
     [-0.23 2.11]]
```

## Tips for Best Experience

1. **Start Simple**: Begin by dragging groups to understand basic behavior
2. **Blow Up One at a Time**: Easier to observe effects on individual groups
3. **Watch the Accuracy**: Track how your actions affect clustering quality
4. **Look for Rings**: Reassigned samples appear at cluster boundaries
5. **Experiment Freely**: You can't break it - explore different scenarios!

## Technical Details

### Data Generation
- Each group: 2000 samples
- Total samples: 6000
- Distribution: Multivariate normal (Gaussian)
- Initial variance: 0.3 (tight clusters)
- Blown up variance: 1.0-3.0 (scattered clusters)

### K-Means Algorithm
- Clusters: 3
- Algorithm: Lloyd's algorithm (sklearn default)
- Initialization: k-means++ (sklearn default)
- Convergence: When cluster centers stabilize

### Cluster Mapping
The program intelligently maps K-Means cluster labels (0, 1, 2) to original groups (0, 1, 2) using majority voting:
- For each K-Means cluster, find which original group has the most members
- This ensures ring colors correctly show reassignments

## Troubleshooting

### GUI doesn't open (WSL users)
- Install X server (VcXsrv, X410, or WSLg)
- Or run from Windows PowerShell instead

### Buttons not responding
- Make sure to click directly on the button area
- Try clicking center of button

### Slow performance
- Normal for 6000 samples
- Each action regenerates all data and re-clusters
- Consider reducing `n_samples` in code if needed (line 11)

## Learning Outcomes

After using this tool, you should understand:
- ✓ How K-Means finds cluster centers
- ✓ Impact of cluster separation on accuracy
- ✓ Impact of variance on clustering quality
- ✓ K-Means works best with well-separated, low-variance clusters
- ✓ K-Means struggles with overlapping or high-variance clusters
- ✓ Boundary samples are most likely to be misclassified
