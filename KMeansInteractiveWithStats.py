# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from matplotlib.widgets import Button
from matplotlib.patches import FancyBboxPatch
from collections import Counter
from datetime import datetime
import json
import csv
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Statistics Tracker Class ---
class KMeansStatistics:
    """Tracks and analyzes K-Means clustering performance statistics."""

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history = []
        self.action_count = 0

    def calculate_statistics(self, data, original_groups, labels, centers, means, covs,
                            cluster_to_group_map, kmeans_model):
        """Calculate comprehensive statistics for current clustering state."""

        stats = {
            'action_number': self.action_count,
            'timestamp': datetime.now().isoformat(),
        }

        # 1. ACCURACY & MISCLASSIFICATION METRICS
        total_samples = len(original_groups)

        # Calculate per-group accuracy and confusion matrix
        confusion_matrix = np.zeros((3, 3), dtype=int)
        per_group_correct = [0, 0, 0]
        per_group_total = [0, 0, 0]

        for i, (orig_group, kmeans_label) in enumerate(zip(original_groups, labels)):
            mapped_group = cluster_to_group_map.get(kmeans_label, kmeans_label)
            confusion_matrix[orig_group][mapped_group] += 1
            per_group_total[orig_group] += 1
            if orig_group == mapped_group:
                per_group_correct[orig_group] += 1

        per_group_accuracy = [(correct / total * 100) if total > 0 else 0
                             for correct, total in zip(per_group_correct, per_group_total)]

        total_correct = sum(per_group_correct)
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        misclassified = total_samples - total_correct

        stats['accuracy'] = {
            'overall_accuracy': overall_accuracy,
            'overall_correct': total_correct,
            'total_misclassified': misclassified,
            'misclassification_rate': (misclassified / total_samples * 100),
            'per_group_accuracy': per_group_accuracy,
            'per_group_correct': per_group_correct,
            'per_group_total': per_group_total,
            'confusion_matrix': confusion_matrix.tolist()
        }

        # 2. CLUSTER SEPARATION METRICS
        # Calculate pairwise distances between group centers
        inter_cluster_distances = {}
        for i in range(3):
            for j in range(i+1, 3):
                dist = np.linalg.norm(means[i] - means[j])
                inter_cluster_distances[f'group_{i+1}_to_{j+1}'] = dist

        min_separation = min(inter_cluster_distances.values())
        max_separation = max(inter_cluster_distances.values())
        avg_separation = np.mean(list(inter_cluster_distances.values()))

        # Calculate center displacement (K-Means center vs true center)
        center_displacements = []
        for cluster_id in range(3):
            mapped_group = cluster_to_group_map.get(cluster_id, cluster_id)
            displacement = np.linalg.norm(centers[cluster_id] - means[mapped_group])
            center_displacements.append(displacement)

        stats['separation'] = {
            'inter_cluster_distances': inter_cluster_distances,
            'min_separation': min_separation,
            'max_separation': max_separation,
            'avg_separation': avg_separation,
            'center_displacements': center_displacements,
            'avg_center_displacement': np.mean(center_displacements),
            'max_center_displacement': np.max(center_displacements)
        }

        # 3. VARIANCE METRICS
        # Calculate within-group variance
        within_group_variance = []
        for group_id in range(3):
            cov_matrix = np.array(covs[group_id])
            # Total variance is trace of covariance matrix
            total_var = np.trace(cov_matrix)
            within_group_variance.append(total_var)

        # Calculate between-group variance
        overall_mean = np.mean(means, axis=0)
        between_group_variance = sum([np.sum((mean - overall_mean)**2) for mean in means])

        # Variance ratio (higher is better for clustering)
        avg_within_var = np.mean(within_group_variance)
        variance_ratio = between_group_variance / avg_within_var if avg_within_var > 0 else 0

        stats['variance'] = {
            'within_group_variance': within_group_variance,
            'avg_within_group_variance': avg_within_var,
            'between_group_variance': between_group_variance,
            'variance_ratio': variance_ratio
        }

        # 4. SILHOUETTE ANALYSIS
        try:
            overall_silhouette = silhouette_score(data, labels)
            sample_silhouette = silhouette_samples(data, labels)

            # Calculate per-group silhouette scores
            per_group_silhouette = []
            for group_id in range(3):
                group_mask = original_groups == group_id
                if np.any(group_mask):
                    group_silhouette = np.mean(sample_silhouette[group_mask])
                    per_group_silhouette.append(group_silhouette)
                else:
                    per_group_silhouette.append(0.0)

            # Find samples with negative silhouette (likely misclassified)
            negative_silhouette_count = np.sum(sample_silhouette < 0)

            stats['silhouette'] = {
                'overall_score': overall_silhouette,
                'per_group_scores': per_group_silhouette,
                'negative_samples': int(negative_silhouette_count),
                'negative_percentage': (negative_silhouette_count / total_samples * 100)
            }
        except Exception as e:
            stats['silhouette'] = {'error': str(e)}

        # 5. ADVANCED CLUSTERING METRICS
        try:
            davies_bouldin = davies_bouldin_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)

            stats['advanced_metrics'] = {
                'davies_bouldin_index': davies_bouldin,  # Lower is better
                'calinski_harabasz_score': calinski_harabasz  # Higher is better
            }
        except Exception as e:
            stats['advanced_metrics'] = {'error': str(e)}

        # 6. INERTIA & CONVERGENCE
        stats['inertia'] = {
            'total_inertia': kmeans_model.inertia_,
            'n_iterations': int(kmeans_model.n_iter_)
        }

        # 7. CONFIGURATION
        stats['configuration'] = {
            'group_centers': [mean.tolist() for mean in means],
            'group_covariances': [[list(row) for row in cov] for cov in covs],
            'kmeans_centers': centers.tolist(),
            'cluster_to_group_mapping': cluster_to_group_map
        }

        return stats

    def add_statistics(self, stats):
        """Add statistics entry to history."""
        self.action_count += 1
        self.history.append(stats)
        return stats

    def print_statistics(self, stats):
        """Print formatted statistics to console."""
        print("\n" + "="*70)
        print(f"ACTION #{stats['action_number']} STATISTICS")
        print("="*70)

        # Accuracy
        acc = stats['accuracy']
        print("\n📊 ACCURACY METRICS:")
        print(f"  Overall Accuracy: {acc['overall_accuracy']:.2f}%")
        print(f"  Correctly Classified: {acc['overall_correct']}/{acc['per_group_total'][0]*3}")
        print(f"  Misclassified: {acc['total_misclassified']}")
        print("\n  Per-Group Accuracy:")
        for i, accuracy in enumerate(acc['per_group_accuracy']):
            print(f"    Group {i+1}: {accuracy:.2f}% ({acc['per_group_correct'][i]}/{acc['per_group_total'][i]})")

        # Confusion Matrix
        print("\n  Confusion Matrix:")
        print("         Predicted→  G1    G2    G3")
        cm = acc['confusion_matrix']
        for i in range(3):
            print(f"    True G{i+1}          {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}")

        # Separation
        sep = stats['separation']
        print("\n📏 SEPARATION METRICS:")
        print("  Inter-Cluster Distances:")
        for pair, dist in sep['inter_cluster_distances'].items():
            print(f"    {pair}: {dist:.2f}")
        print(f"  Min/Avg/Max Separation: {sep['min_separation']:.2f} / {sep['avg_separation']:.2f} / {sep['max_separation']:.2f}")
        print(f"  Avg K-Means Center Displacement: {sep['avg_center_displacement']:.2f}")

        # Variance
        var = stats['variance']
        print("\n📈 VARIANCE METRICS:")
        print("  Within-Group Variance:")
        for i, v in enumerate(var['within_group_variance']):
            print(f"    Group {i+1}: {v:.3f}")
        print(f"  Between-Group Variance: {var['between_group_variance']:.3f}")
        print(f"  Variance Ratio (between/within): {var['variance_ratio']:.3f}")

        # Silhouette
        if 'error' not in stats['silhouette']:
            sil = stats['silhouette']
            print("\n🎯 SILHOUETTE ANALYSIS:")
            print(f"  Overall Score: {sil['overall_score']:.3f} (range: -1 to 1, higher is better)")
            print("  Per-Group Scores:")
            for i, score in enumerate(sil['per_group_scores']):
                print(f"    Group {i+1}: {score:.3f}")
            print(f"  Negative Silhouette Samples: {sil['negative_samples']} ({sil['negative_percentage']:.2f}%)")

        # Advanced metrics
        if 'error' not in stats['advanced_metrics']:
            adv = stats['advanced_metrics']
            print("\n🔬 ADVANCED METRICS:")
            print(f"  Davies-Bouldin Index: {adv['davies_bouldin_index']:.3f} (lower is better)")
            print(f"  Calinski-Harabasz Score: {adv['calinski_harabasz_score']:.2f} (higher is better)")

        # Inertia
        inertia = stats['inertia']
        print("\n⚙️  CONVERGENCE:")
        print(f"  Total Inertia (SSE): {inertia['total_inertia']:.2f}")
        print(f"  Iterations to Converge: {inertia['n_iterations']}")

        print("="*70 + "\n")

    def save_session_summary(self, output_dir="interactive_run_results"):
        """Save comprehensive session summary to files."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = self.session_id

        # 1. Save JSON with all raw data
        json_file = os.path.join(output_dir, f"session_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'total_actions': self.action_count,
                'history': self.history
            }, f, indent=2, cls=NumpyEncoder)
        print(f"✅ Saved detailed JSON: {json_file}")

        # 2. Save CSV with key metrics
        csv_file = os.path.join(output_dir, f"session_{timestamp}_summary.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Action', 'Timestamp', 'Overall_Accuracy', 'Misclassified',
                'Min_Separation', 'Avg_Separation', 'Variance_Ratio',
                'Silhouette_Score', 'Davies_Bouldin', 'Calinski_Harabasz',
                'Inertia', 'Iterations'
            ])

            for stats in self.history:
                row = [
                    stats['action_number'],
                    stats['timestamp'],
                    f"{stats['accuracy']['overall_accuracy']:.2f}",
                    stats['accuracy']['total_misclassified'],
                    f"{stats['separation']['min_separation']:.2f}",
                    f"{stats['separation']['avg_separation']:.2f}",
                    f"{stats['variance']['variance_ratio']:.3f}",
                    f"{stats['silhouette'].get('overall_score', 'N/A'):.3f}" if 'overall_score' in stats['silhouette'] else 'N/A',
                    f"{stats['advanced_metrics'].get('davies_bouldin_index', 'N/A'):.3f}" if 'davies_bouldin_index' in stats['advanced_metrics'] else 'N/A',
                    f"{stats['advanced_metrics'].get('calinski_harabasz_score', 'N/A'):.2f}" if 'calinski_harabasz_score' in stats['advanced_metrics'] else 'N/A',
                    f"{stats['inertia']['total_inertia']:.2f}",
                    stats['inertia']['n_iterations']
                ]
                writer.writerow(row)
        print(f"✅ Saved CSV summary: {csv_file}")

        # 3. Save human-readable text report
        txt_file = os.path.join(output_dir, f"session_{timestamp}_report.txt")
        with open(txt_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("K-MEANS CLUSTERING SESSION REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Total Actions: {self.action_count}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

            # Summary statistics across all actions
            if self.history:
                accuracies = [s['accuracy']['overall_accuracy'] for s in self.history]
                separations = [s['separation']['min_separation'] for s in self.history]

                f.write("OVERALL SESSION SUMMARY:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Accuracy Range: {min(accuracies):.2f}% - {max(accuracies):.2f}%\n")
                f.write(f"Average Accuracy: {np.mean(accuracies):.2f}%\n")
                f.write(f"Final Accuracy: {accuracies[-1]:.2f}%\n")
                f.write(f"\nSeparation Range: {min(separations):.2f} - {max(separations):.2f}\n")
                f.write(f"Average Separation: {np.mean(separations):.2f}\n")
                f.write(f"Final Separation: {separations[-1]:.2f}\n")
                f.write("\n" + "="*70 + "\n\n")

                # Detail for each action
                f.write("DETAILED ACTION HISTORY:\n")
                f.write("="*70 + "\n\n")
                for stats in self.history:
                    f.write(f"Action #{stats['action_number']}:\n")
                    f.write(f"  Accuracy: {stats['accuracy']['overall_accuracy']:.2f}%\n")
                    f.write(f"  Misclassified: {stats['accuracy']['total_misclassified']}\n")
                    f.write(f"  Min Separation: {stats['separation']['min_separation']:.2f}\n")
                    if 'overall_score' in stats['silhouette']:
                        f.write(f"  Silhouette: {stats['silhouette']['overall_score']:.3f}\n")
                    f.write("\n")

        print(f"✅ Saved text report: {txt_file}")

        return json_file, csv_file, txt_file


# --- Enhanced Interactive K-Means Clustering GUI ---
class InteractiveKMeansWithStats:
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

        # Statistics tracker
        self.stats_tracker = KMeansStatistics()

        # Create figure with main plot area
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('Interactive K-Means Clustering with Statistics')

        # Initialize plot elements
        self.scatter = None
        self.center_markers = []
        self.kmeans_markers = None
        self.button_texts = []
        self.kmeans_model = None

        # Generate initial data and plot
        self.generate_data()
        self.apply_kmeans()
        self.plot()

        # Calculate and save initial statistics
        self.calculate_and_save_statistics()

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
        self.kmeans_model = KMeans(n_clusters=3, random_state=0, n_init='auto')
        self.labels = self.kmeans_model.fit_predict(self.data)
        self.centers = self.kmeans_model.cluster_centers_

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

    def calculate_and_save_statistics(self):
        """Calculate statistics and save them."""
        stats = self.stats_tracker.calculate_statistics(
            self.data, self.original_groups, self.labels, self.centers,
            self.means, self.covs, self.cluster_to_group_map, self.kmeans_model
        )
        self.stats_tracker.add_statistics(stats)
        self.stats_tracker.print_statistics(stats)

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

        # Calculate and save statistics
        self.calculate_and_save_statistics()

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
        title = (f'Interactive K-Means Clustering with Statistics\n'
                f'Reassigned: {reassigned_count}/{total_samples} '
                f'(Accuracy: {accuracy:.1f}%) | Action: {self.stats_tracker.action_count}')
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
            "• K-Means re-clusters after each action\n"
            "• Statistics are tracked and saved automatically"
        )
        self.fig.text(
            0.02, 0.98, instruction_text,
            transform=self.fig.transFigure,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
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

        # Calculate and save statistics
        self.calculate_and_save_statistics()

        # Reset dragging state
        self.dragging_group_id = None
        self.drag_offset = None

    def show(self):
        """Display the interactive plot."""
        plt.show()

    def save_final_results(self):
        """Save final results and statistics."""
        print("\n" + "="*70)
        print("SAVING SESSION RESULTS")
        print("="*70)
        self.stats_tracker.save_session_summary()


# --- Main Program ---
if __name__ == "__main__":
    print("=" * 70)
    print("Interactive K-Means Clustering with Statistics Tracking")
    print("=" * 70)
    print("\nThis enhanced version tracks comprehensive statistics including:")
    print("  • Accuracy & Confusion Matrix")
    print("  • Cluster Separation Metrics")
    print("  • Variance Analysis")
    print("  • Silhouette Scores")
    print("  • Advanced Clustering Metrics (Davies-Bouldin, Calinski-Harabasz)")
    print("  • Inertia & Convergence Data")
    print("\nAll statistics are saved to 'interactive_run_results/' folder.")
    print("\nStarting interactive GUI...")
    print("• Click and drag anywhere on a group to move it")
    print("• Click '💥 Blow Up' buttons to scatter groups randomly")
    print("• Watch statistics in the console!\n")
    print("Close the window to save session results and exit.\n")

    app = InteractiveKMeansWithStats()
    app.show()

    # Save final results
    app.save_final_results()

    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print("\nFinal Configuration:")
    print("\nFinal group positions:")
    for i, mean in enumerate(app.means):
        print(f"  Group {i+1}: [{mean[0]:.2f}, {mean[1]:.2f}]")

    print("\nFinal group covariances:")
    for i, cov in enumerate(app.covs):
        print(f"  Group {i+1}:")
        print(f"    {np.array(cov)}")

    print("\n" + "=" * 70)
    print("Thank you for using K-Means Interactive Statistics!")
    print("=" * 70)
