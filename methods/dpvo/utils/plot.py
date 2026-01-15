import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_comparison(pred_traj, gt_traj, title="Trajectory Comparison"):
    """
    Create a matplotlib figure comparing predicted and GT trajectories.

    Args:
        pred_traj: (N, 3) predicted trajectory positions
        gt_traj: (N, 3) ground truth trajectory positions
        title: Plot title

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY plane (top-down view)
    axes[0].plot(gt_traj[:, 0], gt_traj[:, 1], 'b--', label='GT', linewidth=2)
    axes[0].plot(pred_traj[:, 0], pred_traj[:, 1], 'r-', label='Pred', linewidth=2)
    axes[0].scatter(gt_traj[0, 0], gt_traj[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    axes[0].scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='purple', s=100, marker='x', zorder=5, label='End')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Plane (Top-down)')
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)

    # XZ plane (side view)
    axes[1].plot(gt_traj[:, 0], gt_traj[:, 2], 'b--', label='GT', linewidth=2)
    axes[1].plot(pred_traj[:, 0], pred_traj[:, 2], 'r-', label='Pred', linewidth=2)
    axes[1].scatter(gt_traj[0, 0], gt_traj[0, 2], c='green', s=100, marker='o', zorder=5)
    axes[1].scatter(gt_traj[-1, 0], gt_traj[-1, 2], c='purple', s=100, marker='x', zorder=5)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ Plane (Side)')
    axes[1].legend()
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)

    # YZ plane (front view)
    axes[2].plot(gt_traj[:, 1], gt_traj[:, 2], 'b--', label='GT', linewidth=2)
    axes[2].plot(pred_traj[:, 1], pred_traj[:, 2], 'r-', label='Pred', linewidth=2)
    axes[2].scatter(gt_traj[0, 1], gt_traj[0, 2], c='green', s=100, marker='o', zorder=5)
    axes[2].scatter(gt_traj[-1, 1], gt_traj[-1, 2], c='purple', s=100, marker='x', zorder=5)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ Plane (Front)')
    axes[2].legend()
    axes[2].axis('equal')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def plot_trajectory_3d(pred_traj, gt_traj, title="3D Trajectory"):
    """Create a 3D trajectory comparison plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'b--', label='GT', linewidth=2)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 'r-', label='Pred', linewidth=2)
    ax.scatter(gt_traj[0, 0], gt_traj[0, 1], gt_traj[0, 2], c='green', s=100, marker='o', label='Start')
    ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], gt_traj[-1, 2], c='purple', s=100, marker='x', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig


def compute_trajectory_error(pred_traj, gt_traj):
    """Compute trajectory error metrics."""
    # Align trajectories using Umeyama alignment
    pred_centered = pred_traj - pred_traj.mean(axis=0)
    gt_centered = gt_traj - gt_traj.mean(axis=0)

    # Compute scale
    scale = np.sqrt((gt_centered ** 2).sum() / (pred_centered ** 2).sum())
    pred_scaled = pred_centered * scale

    # Compute ATE (Absolute Trajectory Error)
    ate = np.sqrt(((pred_scaled - gt_centered) ** 2).sum(axis=1)).mean()

    return ate, scale