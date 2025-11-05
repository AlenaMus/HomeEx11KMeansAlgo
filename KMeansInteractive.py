# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.widgets import Button
from matplotlib.patches import FancyBboxPatch
from collections import Counter

# --- Interactive K-Means Clustering GUI ---
class InteractiveKMeans:
    def __init__(self):
        # Define the number of samples to generate for each group
        self.n_samples = 2000

        # Initialize group means (centers)
        self.means = [
            np.array([0.0, 0.0]),    # Group 1
            np.array([10.0, 10.0]),  # Group 2
            np.array([0.0, 10.0])    # Group 3
        ]

        # Define covariance for each group (can be changed by blow up)
        self.covs = [
            [[0.3, 0], [0, 0.3]],  # Group 1
            [[0.3, 0], [0, 0.3]],  # Group 2
            [[0.3, 0], [0, 0.3]]   # Group 3
        ]

        # Colors for each group (RGB)
        self.colors = ['red', 'green', 'blue']
        self.group_names = ['Group 1', 'Group 2', 'Group 3']

        # Track original group assignment for each sample
        self.original_groups = None

        # Dragging state
        self.dragging_group_id = None
        self.drag_offset = None

        # Create figure with main plot area
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('Interactive K-Means Clustering')

        # Initialize plot elements
        self.scatter = None
        self.center_markers = []
        self.kmeans_markers = None
        self.button_texts = []

        # Generate initial data and plot
        self.generate_data()
        self.apply_kmeans()
        self.plot()

        # Connect event handlers for dragging
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Add instructions
        self.add_instructions()

    def generate_data(self):
        """Generate data for all three groups based on current means and covariances."""
        data_groups = []
        original_groups = []

        for group_id, (mean, cov) in enumerate(zip(self.means, self.covs)):
            data = np.random.multivariate_normal(mean, cov, self.n_samples)
            data_groups.append(data)
            # Track which original group each sample belongs to
            original_groups.extend([group_id] * self.n_samples)

        # Combine all data
        self.data = np.concatenate(data_groups, axis=0)
        self.original_groups = np.array(original_groups)

    def apply_kmeans(self):
        """Apply K-Means clustering to the current data."""
        kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
        self.labels = kmeans.fit_predict(self.data)
        self.centers = kmeans.cluster_centers_

        # Map K-Means cluster labels to original groups
        self.cluster_to_group_map = self.map_clusters_to_groups()

    def map_clusters_to_groups(self):
        """
        Map K-Means cluster labels to original groups based on majority voting.
        Returns a dict mapping kmeans_cluster_id -> original_group_id
        """
        cluster_map = {}

        for cluster_id in range(3):
            cluster_mask = self.labels == cluster_id
            original_groups_in_cluster = self.original_groups[cluster_mask]

            if len(original_groups_in_cluster) > 0:
                most_common = Counter(original_groups_in_cluster).most_common(1)[0][0]
                cluster_map[cluster_id] = most_common
            else:
                cluster_map[cluster_id] = cluster_id

        return cluster_map

    def blow_up_group(self, group_id):
        """
        Blow up a specific group by scattering it with random mean and variance.

        Args:
            group_id: Index of the group to blow up (0, 1, or 2)
        """
        print(f"\n{'='*60}")
        print(f"BLOWING UP {self.group_names[group_id].upper()}! (Group ID: {group_id})")
        print(f"{'='*60}")

        # Generate random new mean within reasonable bounds
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        new_mean_x = np.random.uniform(x_min + 2, x_max - 2)
        new_mean_y = np.random.uniform(y_min + 2, y_max - 2)
        self.means[group_id] = np.array([new_mean_x, new_mean_y])

        # Generate random covariance (increased variance for "blown up" effect)
        var_x = np.random.uniform(1.0, 3.0)
        var_y = np.random.uniform(1.0, 3.0)
        covar = np.random.uniform(-0.5, 0.5) * np.sqrt(var_x * var_y)

        self.covs[group_id] = [[var_x, covar], [covar, var_y]]

        print(f"New mean: [{new_mean_x:.2f}, {new_mean_y:.2f}]")
        print(f"New covariance:\n{np.array(self.covs[group_id])}")

        # Regenerate all data and re-cluster
        self.generate_data()
        self.apply_kmeans()
        self.plot()

        print(f"Blow up complete!")

    def plot(self):
        """Plot or update the visualization with original colors and reassignment rings."""
        self.ax.clear()

        # Prepare colors for each sample based on original group
        face_colors = [self.colors[group_id] for group_id in self.original_groups]

        # Prepare edge colors: if sample was reassigned, show new cluster color
        edge_colors = []
        edge_widths = []

        for i, (original_group, kmeans_label) in enumerate(zip(self.original_groups, self.labels)):
            mapped_group = self.cluster_to_group_map.get(kmeans_label, kmeans_label)

            if mapped_group != original_group:
                edge_colors.append(self.colors[mapped_group])
                edge_widths.append(1.5)
            else:
                edge_colors.append('none')
                edge_widths.append(0)

        # Plot data points with original colors and reassignment rings
        self.scatter = self.ax.scatter(
            self.data[:, 0], self.data[:, 1],
            c=face_colors, edgecolors=edge_colors, linewidths=edge_widths,
            alpha=0.6, s=30, picker=5  # Enable picking for dragging groups
        )

        # Plot K-Means centers
        self.kmeans_markers = self.ax.scatter(
            self.centers[:, 0], self.centers[:, 1],
            c='black', marker='x', s=300, linewidths=4,
            label='K-Means Centers', zorder=5
        )

        # Plot original group means
        self.center_markers = []
        for i, (mean, color, name) in enumerate(zip(self.means, self.colors, self.group_names)):
            marker = self.ax.plot(
                mean[0], mean[1],
                marker='s', color=color, markersize=15,
                markeredgewidth=2, markeredgecolor='black',
                label=f'{name} Mean',
                picker=10, zorder=10
            )[0]
            self.center_markers.append(marker)

        # Add "Blow Up" buttons at group centers
        self.button_texts = []
        for i, (mean, color, name) in enumerate(zip(self.means, self.colors, self.group_names)):
            # Create button-like text annotation at group center
            button_text = self.ax.text(
                mean[0], mean[1] - 1.5,  # Offset below center
                f'💥 Blow Up',
                fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                         edgecolor='black', linewidth=2, alpha=0.9),
                picker=10, zorder=11
            )
            self.button_texts.append(button_text)

        # Count reassignments
        reassigned_count = sum(1 for i in range(len(self.original_groups))
                              if self.cluster_to_group_map.get(self.labels[i], self.labels[i]) != self.original_groups[i])
        total_samples = len(self.original_groups)
        accuracy = ((total_samples - reassigned_count) / total_samples) * 100

        # Set labels and title
        self.ax.set_xlabel('X-axis', fontsize=12)
        self.ax.set_ylabel('Y-axis', fontsize=12)
        title = (f'Interactive K-Means Clustering\n'
                f'Reassigned: {reassigned_count}/{total_samples} '
                f'(Accuracy: {accuracy:.1f}%)')
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=9)

        # Set axis limits with padding
        all_points = np.vstack([self.data, self.means])
        x_min, x_max = all_points[:, 0].min() - 2, all_points[:, 0].max() + 2
        y_min, y_max = all_points[:, 1].min() - 2, all_points[:, 1].max() + 2
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        self.fig.canvas.draw()

    def add_instructions(self):
        """Add instruction text to the plot."""
        instruction_text = (
            "Instructions:\n"
            "• Click and drag anywhere on a group (cloud of points) to move it\n"
            "• Click '💥 Blow Up' buttons (in group centers) to scatter that group\n"
            "• Samples keep original color, rings show reassignment\n"
            "• K-Means re-clusters after each action"
        )
        self.fig.text(
            0.02, 0.98, instruction_text,
            transform=self.fig.transFigure,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        )

    def find_clicked_group(self, event):
        """Find which group was clicked based on click position."""
        if event.xdata is None or event.ydata is None:
            return None

        click_point = np.array([event.xdata, event.ydata])

        # Check each group's data to see if click is within its bounds
        for group_id in range(3):
            # Get data points for this group
            group_mask = self.original_groups == group_id
            group_data = self.data[group_mask]

            if len(group_data) == 0:
                continue

            # Calculate distance from click to group center
            group_center = self.means[group_id]
            dist_to_center = np.linalg.norm(click_point - group_center)

            # Check if click is close to any point in the group
            distances = np.linalg.norm(group_data - click_point, axis=1)
            min_distance = np.min(distances)

            # If clicked within a reasonable radius of the group
            if min_distance < 2.0:  # Adjust threshold as needed
                return group_id

        return None

    def on_press(self, event):
        """Handle mouse press event."""
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Check if "Blow Up" button was clicked
        for i, button_text in enumerate(self.button_texts):
            contains, _ = button_text.contains(event)
            if contains:
                self.blow_up_group(i)
                return

        # Check if center marker was clicked for precise dragging
        for i, marker in enumerate(self.center_markers):
            contains, _ = marker.contains(event)
            if contains:
                self.dragging_group_id = i
                self.drag_offset = self.means[i] - np.array([event.xdata, event.ydata])
                marker.set_markersize(20)
                self.fig.canvas.draw()
                return

        # Check if anywhere on a group was clicked
        clicked_group = self.find_clicked_group(event)
        if clicked_group is not None:
            self.dragging_group_id = clicked_group
            self.drag_offset = self.means[clicked_group] - np.array([event.xdata, event.ydata])
            self.center_markers[clicked_group].set_markersize(20)
            self.fig.canvas.draw()

    def on_motion(self, event):
        """Handle mouse motion event (dragging)."""
        if self.dragging_group_id is None or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Update the position of the dragged group
        new_position = np.array([event.xdata, event.ydata]) + self.drag_offset
        self.means[self.dragging_group_id] = new_position

        # Update the marker position
        self.center_markers[self.dragging_group_id].set_data(
            [new_position[0]], [new_position[1]]
        )

        # Update button position
        self.button_texts[self.dragging_group_id].set_position(
            (new_position[0], new_position[1] - 1.5)
        )

        self.fig.canvas.draw()

    def on_release(self, event):
        """Handle mouse release event."""
        if self.dragging_group_id is None:
            return

        # Reset marker size
        self.center_markers[self.dragging_group_id].set_markersize(15)

        # Regenerate data and re-cluster
        print(f"\nGroup {self.dragging_group_id + 1} moved to: "
              f"[{self.means[self.dragging_group_id][0]:.2f}, {self.means[self.dragging_group_id][1]:.2f}]")

        # Keep tight covariance when manually moving
        self.covs[self.dragging_group_id] = [[0.3, 0], [0, 0.3]]

        self.generate_data()
        self.apply_kmeans()
        self.plot()

        # Reset dragging state
        self.dragging_group_id = None
        self.drag_offset = None

    def show(self):
        """Display the interactive plot."""
        plt.show()


# --- Main Program ---
if __name__ == "__main__":
    print("=" * 60)
    print("Interactive K-Means Clustering with Blow Up Feature")
    print("=" * 60)
    print("\nStarting interactive GUI...")
    print("• Click and drag anywhere on a group to move it")
    print("• Click '💥 Blow Up' buttons to scatter groups randomly")
    print("• Watch how K-Means reassigns samples!\n")
    print("Close the window to exit.\n")

    app = InteractiveKMeans()
    app.show()

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print("\nFinal group positions:")
    for i, mean in enumerate(app.means):
        print(f"  Group {i+1}: [{mean[0]:.2f}, {mean[1]:.2f}]")

    print("\nFinal group covariances:")
    for i, cov in enumerate(app.covs):
        print(f"  Group {i+1}:")
        print(f"    {np.array(cov)}")
