# K-Means Algorithm Data Generator

This project consists of a Python script that generates and visualizes sample data that can be used for K-Means clustering exercises. It creates three distinct groups of 2D vectors, each following a Gaussian (normal) distribution, and plots them on a scatter graph.

## Features

- **Data Generation**: Creates 2000 samples for each of the three data groups.
- **Gaussian Distribution**: Each group is centered around a specific mean and has a defined covariance, causing them to form clusters.
- **Visualization**: Plots all data points on a 2D scatter plot.
- **Clarity**: 
    - Each group is color-coded (Red, Green, Blue).
    - The geometric mean of each group is marked with an 'x'.
    - Overlapping regions between the groups are visible due to semi-transparent plotting.
- **Data Explanation**: A comprehensive legend details the name, mean, and variance for each group.
- **Educational**: The code is heavily commented to be used as a learning tool.

## Requirements

- Python 3
- `numpy`
- `matplotlib`

## Installation

1.  Ensure you have Python 3 installed.
2.  Install the required libraries using pip:
    ```sh
    pip install numpy matplotlib
    ```

## Usage

To run the script and generate the visualization, execute the following command in your terminal from the project directory:

```sh
python KMeansAlgoExercise.py
```

A window will pop up displaying the scatter plot of the generated data.

## Expected Output

The script will display a plot titled "Gaussian Distribution of 3 Groups of 2D Vectors". The plot will contain three colored clusters of points, with the centers marked. A legend will provide detailed information about each group.

![Sample Output](https://i.imgur.com/5zJ4z8c.png) *Note: The exact positions of points will vary slightly on each run due to the random nature of the data generation.*
