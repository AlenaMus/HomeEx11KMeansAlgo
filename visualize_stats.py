#!/usr/bin/env python3
"""
Visualization tool for K-Means clustering session statistics.
Generates plots showing how clustering performance changed over time.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from glob import glob

def load_session_data(json_file):
    """Load session data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_session_statistics(session_data, output_dir="interactive_run_results"):
    """Generate comprehensive visualization plots for a session."""

    history = session_data['history']
    session_id = session_data['session_id']

    if not history:
        print("No data to visualize!")
        return

    # Extract data series
    actions = [s['action_number'] for s in history]

    # Accuracy metrics
    overall_accuracy = [s['accuracy']['overall_accuracy'] for s in history]
    misclassified = [s['accuracy']['total_misclassified'] for s in history]
    group_accuracies = [
        [s['accuracy']['per_group_accuracy'][i] for s in history]
        for i in range(3)
    ]

    # Separation metrics
    min_separation = [s['separation']['min_separation'] for s in history]
    avg_separation = [s['separation']['avg_separation'] for s in history]
    max_separation = [s['separation']['max_separation'] for s in history]
    avg_displacement = [s['separation']['avg_center_displacement'] for s in history]

    # Variance metrics
    variance_ratio = [s['variance']['variance_ratio'] for s in history]
    within_variance = [
        [s['variance']['within_group_variance'][i] for s in history]
        for i in range(3)
    ]

    # Silhouette scores (if available)
    silhouette_scores = []
    for s in history:
        if 'overall_score' in s.get('silhouette', {}):
            silhouette_scores.append(s['silhouette']['overall_score'])
        else:
            silhouette_scores.append(None)

    # Advanced metrics (if available)
    davies_bouldin = []
    calinski_harabasz = []
    for s in history:
        adv = s.get('advanced_metrics', {})
        davies_bouldin.append(adv.get('davies_bouldin_index', None))
        calinski_harabasz.append(adv.get('calinski_harabasz_score', None))

    # Inertia
    inertia = [s['inertia']['total_inertia'] for s in history]
    iterations = [s['inertia']['n_iterations'] for s in history]

    # Create comprehensive figure with subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'K-Means Clustering Session Analysis\nSession ID: {session_id}',
                 fontsize=16, fontweight='bold')

    # 1. Overall Accuracy Over Time
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(actions, overall_accuracy, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')
    ax1.set_xlabel('Action Number')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Overall Accuracy Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # 2. Per-Group Accuracy
    ax2 = plt.subplot(3, 3, 2)
    colors = ['red', 'green', 'blue']
    for i, (acc, color) in enumerate(zip(group_accuracies, colors)):
        ax2.plot(actions, acc, 'o-', linewidth=2, markersize=6,
                color=color, label=f'Group {i+1}', alpha=0.7)
    ax2.set_xlabel('Action Number')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Per-Group Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # 3. Misclassified Samples
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(actions, misclassified, 'o-', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Action Number')
    ax3.set_ylabel('Misclassified Samples')
    ax3.set_title('Total Misclassified Samples', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Cluster Separation
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(actions, min_separation, 'o-', linewidth=2, markersize=6,
            color='red', label='Min Separation')
    ax4.plot(actions, avg_separation, 's-', linewidth=2, markersize=6,
            color='orange', label='Avg Separation')
    ax4.plot(actions, max_separation, '^-', linewidth=2, markersize=6,
            color='green', label='Max Separation')
    ax4.set_xlabel('Action Number')
    ax4.set_ylabel('Distance')
    ax4.set_title('Cluster Separation', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Center Displacement
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(actions, avg_displacement, 'o-', linewidth=2, markersize=8, color='purple')
    ax5.set_xlabel('Action Number')
    ax5.set_ylabel('Average Displacement')
    ax5.set_title('K-Means Center Displacement', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Variance Ratio
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(actions, variance_ratio, 'o-', linewidth=2, markersize=8, color='brown')
    ax6.set_xlabel('Action Number')
    ax6.set_ylabel('Variance Ratio')
    ax6.set_title('Between/Within Variance Ratio', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Silhouette Score (if available)
    ax7 = plt.subplot(3, 3, 7)
    if any(s is not None for s in silhouette_scores):
        valid_actions = [a for a, s in zip(actions, silhouette_scores) if s is not None]
        valid_scores = [s for s in silhouette_scores if s is not None]
        ax7.plot(valid_actions, valid_scores, 'o-', linewidth=2, markersize=8, color='teal')
        ax7.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax7.set_ylim([-1, 1])
    ax7.set_xlabel('Action Number')
    ax7.set_ylabel('Silhouette Score')
    ax7.set_title('Silhouette Score (-1 to 1)', fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # 8. Inertia (SSE)
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(actions, inertia, 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax8.set_xlabel('Action Number')
    ax8.set_ylabel('Inertia (SSE)')
    ax8.set_title('Total Inertia (Sum of Squared Errors)', fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9. Accuracy vs Separation Scatter Plot
    ax9 = plt.subplot(3, 3, 9)
    scatter = ax9.scatter(min_separation, overall_accuracy,
                         c=actions, cmap='viridis', s=100, alpha=0.7)
    ax9.set_xlabel('Minimum Separation')
    ax9.set_ylabel('Overall Accuracy (%)')
    ax9.set_title('Accuracy vs Separation', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax9)
    cbar.set_label('Action #')

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, f'session_{session_id}_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_file}")

    # Create a second figure for confusion matrices (if interesting)
    if len(history) > 1:
        fig2, axes = plt.subplots(1, min(4, len(history)), figsize=(16, 4))
        if len(history) == 1:
            axes = [axes]

        fig2.suptitle(f'Confusion Matrix Evolution\nSession ID: {session_id}',
                     fontsize=14, fontweight='bold')

        # Show confusion matrices at key points
        indices_to_show = [0]  # First action
        if len(history) > 2:
            mid = len(history) // 2
            indices_to_show.append(mid)  # Middle action
        if len(history) > 1:
            indices_to_show.append(len(history) - 1)  # Last action

        for idx, action_idx in enumerate(indices_to_show[:4]):
            ax = axes[idx] if len(indices_to_show) > 1 else axes[0]
            cm = np.array(history[action_idx]['accuracy']['confusion_matrix'])

            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(['G1', 'G2', 'G3'])
            ax.set_yticklabels(['G1', 'G2', 'G3'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Action #{history[action_idx]["action_number"]}\n'
                        f'Acc: {history[action_idx]["accuracy"]["overall_accuracy"]:.1f}%')

            # Add text annotations
            for i in range(3):
                for j in range(3):
                    text = ax.text(j, i, cm[i, j],
                                 ha="center", va="center", color="black", fontweight='bold')

            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        output_file2 = os.path.join(output_dir, f'session_{session_id}_confusion_matrices.png')
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"✅ Saved confusion matrices: {output_file2}")

    plt.show()

def visualize_latest_session(results_dir="interactive_run_results"):
    """Find and visualize the latest session."""
    json_files = glob(os.path.join(results_dir, "session_*.json"))

    if not json_files:
        print(f"No session files found in {results_dir}/")
        return

    # Get the most recent file
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"Loading latest session: {latest_file}")

    session_data = load_session_data(latest_file)
    plot_session_statistics(session_data, results_dir)

def visualize_specific_session(session_file):
    """Visualize a specific session file."""
    if not os.path.exists(session_file):
        print(f"File not found: {session_file}")
        return

    print(f"Loading session: {session_file}")
    session_data = load_session_data(session_file)

    output_dir = os.path.dirname(session_file) or "interactive_run_results"
    plot_session_statistics(session_data, output_dir)

def list_sessions(results_dir="interactive_run_results"):
    """List all available session files."""
    json_files = glob(os.path.join(results_dir, "session_*.json"))

    if not json_files:
        print(f"No session files found in {results_dir}/")
        return

    print(f"\nAvailable sessions in {results_dir}/:")
    print("="*70)
    for i, file in enumerate(sorted(json_files, key=os.path.getmtime, reverse=True), 1):
        session_data = load_session_data(file)
        print(f"{i}. {os.path.basename(file)}")
        print(f"   Session ID: {session_data['session_id']}")
        print(f"   Total Actions: {session_data['total_actions']}")
        if session_data['history']:
            final_acc = session_data['history'][-1]['accuracy']['overall_accuracy']
            print(f"   Final Accuracy: {final_acc:.2f}%")
        print()

if __name__ == "__main__":
    print("="*70)
    print("K-Means Session Statistics Visualizer")
    print("="*70)

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_sessions()
        else:
            visualize_specific_session(sys.argv[1])
    else:
        print("\nUsage:")
        print("  python visualize_stats.py              # Visualize latest session")
        print("  python visualize_stats.py list         # List all sessions")
        print("  python visualize_stats.py <file.json>  # Visualize specific session")
        print()

        # Default: visualize latest
        print("Visualizing latest session...\n")
        visualize_latest_session()
