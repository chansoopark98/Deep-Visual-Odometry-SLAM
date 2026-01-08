import os
import sys
import yaml
from collections import OrderedDict

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


def train(config):
    """Main training loop."""

    # Find active dataset
    dataset_type = None
    for name in ['tartan', 'redwood']:
        if config['dataset'].get(name, {}).get('use', False):
            dataset_type = name
            break

    if dataset_type is None:
        raise ValueError("No dataset enabled. Set 'use: true' for at least one dataset.")

    # Print configuration
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"  Dataset: {dataset_type}")
    print(f"  Datapath: {config['dataset'][dataset_type]['path']}")
    print(f"  Mode: {config['dataset'][dataset_type].get('mode', 'train')}")
    print(f"  Steps: {config['training'].get('steps', 240000)}")
    print(f"  Learning rate: {config['training'].get('lr', 0.00008)}")
    print(f"  Scheduler: {config.get('scheduler', {}).get('type', 'onecycle')}")
    print(f"  N frames: {config['model'].get('n_frames', 15)}")
    print(f"  Batch size: {config['dataloader'].get('batch_size', 1)}")
    print(f"  Pose weight: {config['loss'].get('pose_weight', 10.0)}")
    print(f"  Flow weight: {config['loss'].get('flow_weight', 0.1)}")
    print("=" * 60)

    # Build dataset
    db = dataset_factory(
        [dataset_type],
        datapath=config['dataset'][dataset_type]['path'],
        n_frames=config['model'].get('n_frames', 15),
        mode=config['dataset'][dataset_type].get('mode', 'train')
    )
    train_loader = DataLoader(
        db,
        batch_size=config['dataloader'].get('batch_size', 1),
        shuffle=True,
        num_workers=config['dataloader'].get('num_workers', 4)
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

    logger = Logger(config['training'].get('name', 'dpvo'), scheduler)

    # Training loop
    total_steps = 0
    flow_weight = config['loss'].get('flow_weight', 0.1)
    pose_weight = config['loss'].get('pose_weight', 10.0)
    clip = config['training'].get('clip', 10.0)
    save_freq = config['training'].get('save_freq', 10000)
    M = config['model'].get('M', 1024)
    STEPS = config['model'].get('STEPS', 18)

    while True:
        for data_blob in train_loader:
            images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
            optimizer.zero_grad()

            # Fix poses to GT for first 1k steps
            so = total_steps < 1000 and ckpt is None

            poses = SE3(poses).inv()
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
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
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

            if total_steps % save_freq == 0:
                torch.cuda.empty_cache()
                os.makedirs('checkpoints', exist_ok=True)
                PATH = 'checkpoints/%s_%06d.pth' % (config['training'].get('name', 'dpvo'), total_steps)
                torch.save(net.state_dict(), PATH)
                torch.cuda.empty_cache()
                net.train()


if __name__ == '__main__':
    config_path = 'config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    train(config)
