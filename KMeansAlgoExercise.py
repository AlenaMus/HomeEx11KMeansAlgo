# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Parameters ---
# Define the number of samples to generate for each group.
n_samples = 2000

# --- 2. Define Group Properties and Generate Data ---
# Define the properties for the first group (mean and covariance matrix).
# The mean determines the center of the cluster.
# The covariance matrix determines the shape and orientation of the cluster.
mean1 = [0, 0]
cov1 = [[2, 1], [1, 2]]
# Generate 2D vectors for Group 1 from a multivariate normal (Gaussian) distribution.
data1 = np.random.multivariate_normal(mean1, cov1, n_samples)

# Define the properties for the second group.
mean2 = [3, 3]
cov2 = [[2, -1], [-1, 2]]
# Generate 2D vectors for Group 2.
data2 = np.random.multivariate_normal(mean2, cov2, n_samples)

# Define the properties for the third group.
mean3 = [0, 4]
cov3 = [[3, 0], [0, 3]]
# Generate 2D vectors for Group 3.
data3 = np.random.multivariate_normal(mean3, cov3, n_samples)

# --- 3. Create the Visualization ---
# Create a figure and a set of subplots. This is the canvas for our plot.
fig, ax = plt.subplots(figsize=(10, 8))

# --- 4. Calculate Variances ---
# Calculate the variance for each group. For a multivariate distribution,
# the total variance can be represented by the trace of the covariance matrix.
variance1 = np.trace(cov1)
variance2 = np.trace(cov2)
variance3 = np.trace(cov3)

# --- 5. Plot the Data ---
# Create a scatter plot for each group of data.
# `alpha=0.5` makes the points semi-transparent to show overlaps.
# The label for each group includes its mean and calculated variance for the legend.
ax.scatter(data1[:, 0], data1[:, 1], c='red', alpha=0.5, label=f'Group 1\nMean: {mean1}\nVariance: {variance1:.2f}')
ax.scatter(data2[:, 0], data2[:, 1], c='green', alpha=0.5, label=f'Group 2\nMean: {mean2}\nVariance: {variance2:.2f}')
ax.scatter(data3[:, 0], data3[:, 1], c='blue', alpha=0.5, label=f'Group 3\nMean: {mean3}\nVariance: {variance3:.2f}')

# --- 6. Mark the Group Means ---
# Plot a distinct marker ('x') at the mean of each group to make them clearly visible.
ax.plot(mean1[0], mean1[1], 'x', color='darkred', markersize=10, mew=2)
ax.plot(mean2[0], mean2[1], 'x', color='darkgreen', markersize=10, mew=2)
ax.plot(mean3[0], mean3[1], 'x', color='darkblue', markersize=10, mew=2)


# --- 7. Finalize the Plot ---
# Set the title for the plot.
ax.set_title('Gaussian Distribution of 3 Groups of 2D Vectors')
# Set the labels for the x and y axes.
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
# Display the legend, which contains the explanatory text for each group.
ax.legend()
# Display a grid for better readability.
ax.grid(True)

# --- 8. Save and Show the Plot ---
# Save the plot to a PNG file before displaying it.
plt.savefig('KMeans_Visualization.png')

# Render and display the complete plot on the screen.
plt.show()