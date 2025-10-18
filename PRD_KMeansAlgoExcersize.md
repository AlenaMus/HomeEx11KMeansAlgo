# Product Requirements Document: 2D Gaussian Data Visualizer

## 1. Objective

To create a standalone Python script that generates and visualizes three distinct groups of 2D data points, each following a Gaussian (normal) distribution. The script serves as a tool for generating sample data for educational purposes, specifically for exercises related to clustering algorithms like K-Means.

## 2. Functional Requirements

### 2.1. Data Generation
- The script must generate a specified number of 2D vectors (samples) for three separate groups.
- Each group must be generated from a multivariate normal distribution with a unique mean and covariance matrix.
- The parameters (mean, covariance) for each group should be defined within the script to ensure that the groups are distinct yet have overlapping regions.

### 2.2. Visualization
- The script must produce a 2D scatter plot visualizing the generated data points.
- Each of the three groups must be represented by a different color to be easily distinguishable.
- The plot must be clearly titled and have labeled X and Y axes.
- A legend must be present to provide a clear explanation of the plotted data.
- The legend entry for each group must display its color, name (e.g., "Group 1"), the mean vector, and the calculated variance.
- The geometric mean (center) of each group should be clearly marked on the plot with a distinct symbol.

### 2.3. Code and Documentation
- The Python script must be well-commented to explain each logical step of the algorithm, from data generation to plotting.
- The project must include a `Readme.md` file explaining the project's purpose, setup, and usage instructions.

## 3. Technical Requirements

- **Language:** Python 3
- **Libraries:**
    - `numpy`: For numerical operations and data generation.
    - `matplotlib`: For data visualization and plotting.
- **Execution:** The script should be executable from a standard Python interpreter without requiring any special environment, other than the installation of the necessary libraries.

## 4. Out of Scope

- This is not a K-Means clustering implementation. It is a data generation tool.
- The script will not take command-line arguments for configuration. All parameters (number of samples, means, covariances) are to be hardcoded.
- The output is a screen display of the plot; the script is not required to save the plot to a file automatically.
