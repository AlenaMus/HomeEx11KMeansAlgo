# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- 1. Define Parameters ---
# Define the number of samples to generate for each group.
n_samples = 2000

# --- 2. Define Group Properties and Generate Data ---
# Define the properties for the first group (mean and covariance matrix).
# The mean determines the center of the cluster.
# The covariance matrix determines the shape and orientation of the cluster.
# UPDATED: Reduced covariance for minimal within-group variance (tighter cluster)
mean1 = [0, 0]
cov1 = [[0.3, 0], [0, 0.3]]  # Much smaller variance for tighter cluster
# Generate 2D vectors for Group 1 from a multivariate normal (Gaussian) distribution.
data1 = np.random.multivariate_normal(mean1, cov1, n_samples)

# Define the properties for the second group.
# UPDATED: Moved mean further away for maximal between-group variance and reduced covariance
mean2 = [10, 10]  # Moved much further from group 1
cov2 = [[0.3, 0], [0, 0.3]]  # Much smaller variance for tighter cluster
# Generate 2D vectors for Group 2.
data2 = np.random.multivariate_normal(mean2, cov2, n_samples)

# Define the properties for the third group.
# UPDATED: Moved mean for better separation and reduced covariance
mean3 = [0, 10]  # Moved further from group 1, forms triangle with other groups
cov3 = [[0.3, 0], [0, 0.3]]  # Much smaller variance for tighter cluster
# Generate 2D vectors for Group 3.
data3 = np.random.multivariate_normal(mean3, cov3, n_samples)

# --- 3. Combine Data ---
# Combine the three datasets into a single array for clustering.
data = np.concatenate((data1, data2, data3), axis=0)

# --- 4. Apply K-Means Algorithm ---
# Initialize the K-Means model to find 3 clusters.
# `n_init='auto'` is set to avoid a FutureWarning and use the default value.
kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
# Fit the model to the combined data and predict the cluster for each point.
labels = kmeans.fit_predict(data)
# Get the coordinates of the cluster centers found by the algorithm.
centers = kmeans.cluster_centers_

# --- 5. Create the Visualization ---
# Create a figure and a set of subplots.
fig, ax = plt.subplots(figsize=(10, 8))

# --- 6. Plot K-Means Results ---
# Plot the data points, coloring them by the cluster label found by K-Means.
scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5, cmap='viridis')
# Plot the cluster centers found by K-Means as black 'X's.
ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='K-Means Centers')

# --- 7. Mark Original Group Means ---
# Plot the original means to compare with K-Means results.
ax.plot(mean1[0], mean1[1], 's', color='red', markersize=10, mew=2, label='Original Mean 1')
ax.plot(mean2[0], mean2[1], 's', color='green', markersize=10, mew=2, label='Original Mean 2')
ax.plot(mean3[0], mean3[1], 's', color='blue', markersize=10, mew=2, label='Original Mean 3')

# --- 8. Finalize the Plot ---
# Set the title and labels for the plot.
ax.set_title('K-Means Clustering of Gaussian Data')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
# Create a legend for the plot.
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.legend()
# Display a grid.
ax.grid(True)

# --- 9. Save and Show the Plot ---
# Save the updated plot to a new file.
plt.savefig('KMeans_Clustering_Results.png')

# Show the plot.
plt.show()