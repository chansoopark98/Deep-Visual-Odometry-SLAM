import os
import sys
import yaml
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for TensorBoard
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dpvo.data_readers.factory import dataset_factory
from dpvo.lietorch import SE3
from dpvo.logger import Logger
from dpvo.net import VONet
from utils.utils import kabsch_umeyama


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


@torch.no_grad()
def validate(net, val_loader, config, logger, step, num_samples=8):
    """
    Run validation on a few samples and log trajectory plots to TensorBoard.

    Args:
        net: VONet model
        val_loader: Validation DataLoader
        config: Training configuration
        logger: Logger instance
        step: Current training step
        num_samples: Number of samples to validate
    """
    print(f"\n{'='*60}")
    print(f"Running validation at step {step}...")
    print(f"{'='*60}")

    net.eval()

    M = config['model'].get('M', 1024)
    STEPS = config['model'].get('STEPS', 18)

    all_ate = []
    val_iter = iter(val_loader)

    for sample_idx in range(num_samples):
        try:
            data_blob = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            data_blob = next(val_iter)

        images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
        gt_poses = SE3(poses).inv()

        # Forward pass
        traj = net(images, gt_poses, disps, intrinsics, M=M, STEPS=STEPS, structure_only=False)

        # Get final prediction (last iteration)
        v, x, y, P1, P2, kl = traj[-1]

        # P1: predicted poses, P2: GT poses
        pred_poses = P1.inv()
        gt_poses_final = P2.inv()

        # Extract translation components
        pred_t = pred_poses.matrix()[0, :, :3, 3].cpu().numpy()  # (N, 3)
        gt_t = gt_poses_final.matrix()[0, :, :3, 3].cpu().numpy()  # (N, 3)

        # Normalize to start from origin (first frame = [0,0,0])
        pred_t = pred_t - pred_t[0:1]  # Subtract first frame position
        gt_t = gt_t - gt_t[0:1]        # Subtract first frame position

        # Scale alignment using kabsch_umeyama
        s = kabsch_umeyama(
            torch.from_numpy(gt_t).cuda(),
            torch.from_numpy(pred_t).cuda()
        ).item()
        s = min(s, 10.0)  # Clamp scale
        pred_t_scaled = pred_t * s

        # Compute ATE
        ate, _ = compute_trajectory_error(pred_t_scaled, gt_t)
        all_ate.append(ate)

        print(f"  Sample {sample_idx+1}/{num_samples}: ATE = {ate:.4f}")

        # Plot trajectory for first few samples
        if sample_idx < 4:
            # 2D comparison plot
            fig_2d = plot_trajectory_comparison(
                pred_t_scaled, gt_t,
                title=f"Sample {sample_idx+1} (Step {step}, ATE: {ate:.3f})"
            )
            logger.add_figure(f"val/trajectory_2d_sample{sample_idx+1}", fig_2d, step)
            plt.close(fig_2d)

            # 3D plot
            fig_3d = plot_trajectory_3d(
                pred_t_scaled, gt_t,
                title=f"3D Trajectory Sample {sample_idx+1} (ATE: {ate:.3f})"
            )
            logger.add_figure(f"val/trajectory_3d_sample{sample_idx+1}", fig_3d, step)
            plt.close(fig_3d)

    # Log aggregate metrics
    mean_ate = np.mean(all_ate)
    median_ate = np.median(all_ate)

    val_metrics = {
        "val/ATE_mean": mean_ate,
        "val/ATE_median": median_ate,
        "val/ATE_min": np.min(all_ate),
        "val/ATE_max": np.max(all_ate),
    }

    logger.write_dict(val_metrics, step)

    print(f"\nValidation Summary:")
    print(f"  Mean ATE: {mean_ate:.4f}")
    print(f"  Median ATE: {median_ate:.4f}")
    print(f"{'='*60}\n")

    net.train()

    return val_metrics


def train(config):
    """Main training loop."""

    # Find active datasets
    active_datasets = []
    for name in ['tartan', 'redwood']:
        if config['dataset'].get(name, {}).get('use', False):
            active_datasets.append(name)

    if not active_datasets:
        raise ValueError("No dataset enabled. Set 'use: true' for at least one dataset.")

    # AMP configuration
    use_amp = config['training'].get('amp', False)

    # Print configuration
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"  Dataset: {', '.join(active_datasets)}")
    for ds_name in active_datasets:
        print(f"    - {ds_name}: {config['dataset'][ds_name]['path']}")
    print(f"  Steps: {config['training'].get('steps', 240000)}")
    print(f"  Learning rate: {config['training'].get('lr', 0.00008)}")
    print(f"  Scheduler: {config.get('scheduler', {}).get('type', 'onecycle')}")
    print(f"  AMP: {use_amp}")
    print(f"  N frames: {config['model'].get('n_frames', 15)}")
    print(f"  Batch size: {config['dataloader'].get('batch_size', 1)}")
    print(f"  Pose weight: {config['loss'].get('pose_weight', 10.0)}")
    print(f"  Flow weight: {config['loss'].get('flow_weight', 0.1)}")
    print("=" * 60)

    # Build training dataset
    dataset_configs = []
    for ds_name in active_datasets:
        dataset_configs.append({
            'name': ds_name,
            'datapath': config['dataset'][ds_name]['path'],
            'mode': config['dataset'][ds_name].get('mode', 'train')
        })

    db = dataset_factory(
        dataset_configs,
        n_frames=config['model'].get('n_frames', 15)
    )
    train_loader = DataLoader(
        db,
        batch_size=config['dataloader'].get('batch_size', 1),
        shuffle=True,
        num_workers=config['dataloader'].get('num_workers', 8),
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # Build validation dataset (same datasets but can use different mode)
    val_dataset_configs = []
    for ds_name in active_datasets:
        val_dataset_configs.append({
            'name': ds_name,
            'datapath': config['dataset'][ds_name]['path'],
            'mode': 'train'  # Use train data for validation samples
        })

    val_db = dataset_factory(
        val_dataset_configs,
        n_frames=config['model'].get('n_frames', 15)
    )
    val_loader = DataLoader(
        val_db,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Model
    net = VONet()
    net.train()
    net.cuda()

    ckpt = config['training'].get('ckpt')
    if ckpt is not None:
        state_dict = torch.load(ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict, strict=False)

    # Optimizer & Scheduler
    lr = config['training'].get('lr', 0.00008)
    steps = config['training'].get('steps', 240000)
    weight_decay = config['training'].get('weight_decay', 1e-6)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_type = config.get('scheduler', {}).get('type', 'onecycle')
    if scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lr, steps,
            pct_start=config.get('scheduler', {}).get('pct_start', 0.01),
            cycle_momentum=False, anneal_strategy='linear'
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps,
            eta_min=config.get('scheduler', {}).get('eta_min', 1e-7)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('scheduler', {}).get('step_size', steps // 3),
            gamma=config.get('scheduler', {}).get('gamma', 0.1)
        )
    elif scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # AMP GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    logger = Logger(config['training'].get('name', 'dpvo'), scheduler)

    # Training loop
    total_steps = 0
    flow_weight = config['loss'].get('flow_weight', 0.1)
    pose_weight = config['loss'].get('pose_weight', 10.0)
    clip = config['training'].get('clip', 10.0)
    save_freq = config['training'].get('save_freq', 10000)
    M = config['model'].get('M', 1024)
    STEPS = config['model'].get('STEPS', 18)

    # Validation settings
    val_cfg = config.get('validation', {})
    val_freq = val_cfg.get('freq', 10000)
    val_num_samples = val_cfg.get('num_samples', 8)
    val_enabled = val_cfg.get('enabled', True)

    while True:
        for data_blob in train_loader:
            images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
            optimizer.zero_grad()

            # Fix poses to GT for first 1k steps
            so = total_steps < 1000 and ckpt is None

            poses = SE3(poses).inv()

            with torch.amp.autocast('cuda', enabled=use_amp):
                traj = net(images, poses, disps, intrinsics, M=M, STEPS=STEPS, structure_only=so)

                loss = 0.0
                for i, (v, x, y, P1, P2, kl) in enumerate(traj):
                    e = (x - y).norm(dim=-1)
                    e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values

                    N = P1.shape[1]
                    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
                    ii = ii.reshape(-1).cuda()
                    jj = jj.reshape(-1).cuda()

                    k = ii != jj
                    ii = ii[k]
                    jj = jj[k]

                    P1 = P1.inv()
                    P2 = P2.inv()

                    t1 = P1.matrix()[...,:3,3]
                    t2 = P2.matrix()[...,:3,3]

                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                    P1 = P1.scale(s.view(1, 1))

                    dP = P1[:,ii].inv() * P1[:,jj]
                    dG = P2[:,ii].inv() * P2[:,jj]

                    e1 = (dP * dG.inv()).log()
                    tr = e1[...,0:3].norm(dim=-1)
                    ro = e1[...,3:6].norm(dim=-1)

                    loss += flow_weight * e.mean()
                    if not so and i >= 2:
                        loss += pose_weight * (tr.mean() + ro.mean())

                loss += kl

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_steps += 1

            metrics = {
                "loss": loss.item(),
                "kl": kl.item(),
                "px1": (e < .25).float().mean().item(),
                "ro": ro.float().mean().item(),
                "tr": tr.float().mean().item(),
                "r1": (ro < .001).float().mean().item(),
                "r2": (ro < .01).float().mean().item(),
                "t1": (tr < .001).float().mean().item(),
                "t2": (tr < .01).float().mean().item(),
            }

            logger.push(metrics)

            # Validation
            if val_enabled and total_steps > 0 and total_steps % val_freq == 0:
                torch.cuda.empty_cache()
                validate(net, val_loader, config, logger, total_steps, num_samples=val_num_samples)
                torch.cuda.empty_cache()

            # Periodic memory cleanup to prevent fragmentation
            if total_steps % 1000 == 0:
                torch.cuda.empty_cache()

            if total_steps % save_freq == 0:
                torch.cuda.empty_cache()
                os.makedirs('checkpoints', exist_ok=True)
                PATH = 'checkpoints/%s_%06d.pth' % (config['training'].get('name', 'dpvo'), total_steps)
                torch.save(net.state_dict(), PATH)
                torch.cuda.empty_cache()
                net.train()

            if total_steps >= steps:
                print(f"Training completed at step {total_steps}")
                return


if __name__ == '__main__':
    config_path = 'config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    train(config)
