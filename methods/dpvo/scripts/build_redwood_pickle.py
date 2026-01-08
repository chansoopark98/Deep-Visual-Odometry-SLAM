#!/usr/bin/env python3
"""
Build Redwood dataset pickle file for DPVO training.

This script generates the scene_info pickle file containing:
- Image paths
- Depth paths
- Poses (7-DOF: tx, ty, tz, qx, qy, qz, qw)
- Intrinsics (fx, fy, cx, cy)
- Frame graph (co-visibility based on optical flow)

Usage:
    python scripts/build_redwood_pickle.py --datapath datasets/redwood --mode train
    python scripts/build_redwood_pickle.py --datapath datasets/redwood --mode train --stride 5
"""

import os
import sys
import argparse
import glob
import pickle
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.spatial.transform import Rotation


# Redwood dataset constants
DEPTH_SCALE = 1.0  # Scale factor for depth normalization
DEPTH_MM_TO_M = 1000.0  # mm to meters


def matrix_to_pose(T):
    """
    Convert 4x4 transformation matrix to pose vector.

    Args:
        T: 4x4 transformation matrix

    Returns:
        pose: [tx, ty, tz, qx, qy, qz, qw]
    """
    t = T[:3, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [qx, qy, qz, qw]
    return np.concatenate([t, q])


def load_poses_from_json(json_file):
    """
    Load poses from Redwood JSON file format.

    The JSON contains a PoseGraph with nodes, each node has a 'pose' field
    containing a 4x4 transformation matrix stored as a 16-element array
    in column-major order.
    """
    import json

    with open(json_file, 'r') as f:
        data = json.load(f)

    poses = []
    for node in data['nodes']:
        # Pose is stored as 16 floats in column-major order
        pose_flat = node['pose']
        # Reshape to 4x4 matrix (column-major to row-major)
        matrix = np.array(pose_flat, dtype=np.float32).reshape(4, 4).T

        pose = matrix_to_pose(matrix)
        poses.append(pose)

    return np.array(poses, dtype=np.float32)


def depth_read(depth_file):
    """Read depth image and convert to disparity."""
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / DEPTH_MM_TO_M  # mm to meters

    # Handle invalid values
    depth[depth == 0] = 1.0
    depth[np.isnan(depth)] = 1.0
    depth[np.isinf(depth)] = 1.0

    depth = depth / DEPTH_SCALE
    return depth


def build_frame_graph(poses, depths, intrinsics, f=16, max_flow=256):
    """
    Compute frame co-visibility graph based on optical flow.
    Uses rgbd_utils.compute_distance_matrix_flow for consistency with TartanAir.

    Args:
        poses: list of (7,) pose vectors
        depths: list of depth file paths
        intrinsics: list of (4,) intrinsic vectors
        f: downsample factor
        max_flow: maximum flow threshold

    Returns:
        graph: dict mapping frame_id -> (neighbor_ids, flow_distances)
    """
    from dpvo.data_readers.rgbd_utils import compute_distance_matrix_flow

    def read_disp(fn):
        depth = depth_read(fn)[f//2::f, f//2::f]
        depth[depth < 0.01] = np.mean(depth)
        return 1.0 / depth

    N = len(depths)
    print(f"  Loading {N} disparity maps...")
    disps = np.stack([read_disp(fn) for fn in tqdm(depths, leave=False)], axis=0)

    poses = np.array(poses)
    intrinsics = np.array(intrinsics) / f

    # Estimate time: N^2 pairs, ~2048 per batch
    total_pairs = N * N
    estimated_batches = (total_pairs + 2047) // 2048
    print(f"  Computing flow distance matrix ({N}x{N} = {total_pairs:,} pairs, ~{estimated_batches} batches)...")

    d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

    print(f"  Building graph...")
    graph = {}
    for i in range(d.shape[0]):
        j, = np.where(d[i] < max_flow)
        graph[i] = (j, d[i, j])

    return graph


def build_redwood_pickle(datapath, mode='train', output_path=None, stride=1):
    """
    Build Redwood dataset pickle file.

    Args:
        datapath: path to redwood dataset root
        mode: 'train', 'validation', or 'test'
        output_path: output pickle path (default: cache/Redwood_{mode}.pickle)
        stride: frame subsampling stride (default: 1, use higher for faster build)
    """
    mode_path = os.path.join(datapath, mode)
    if not os.path.isdir(mode_path):
        raise ValueError(f"Mode path does not exist: {mode_path}")

    # Load shared intrinsic
    intrinsic_path = os.path.join(datapath, 'intrinsic.npy')
    if os.path.isfile(intrinsic_path):
        intrinsic_matrix = np.load(intrinsic_path)
        intrinsic = np.array([
            intrinsic_matrix[0, 0],  # fx
            intrinsic_matrix[1, 1],  # fy
            intrinsic_matrix[0, 2],  # cx
            intrinsic_matrix[1, 2],  # cy
        ], dtype=np.float32)
    else:
        # Default PrimeSense intrinsic
        intrinsic = np.array([525.0, 525.0, 319.5, 239.5], dtype=np.float32)
        print(f"Warning: intrinsic.npy not found, using default: {intrinsic}")

    scenes = sorted(glob.glob(os.path.join(mode_path, '*')))
    print(f"Found {len(scenes)} scenes in {mode} split")

    scene_info = {}

    for scene_path in scenes:
        scene_name = os.path.basename(scene_path)
        print(f"\nProcessing scene: {scene_name}")

        # Get image and depth files (use absolute paths)
        images = sorted([os.path.abspath(p) for p in glob.glob(os.path.join(scene_path, 'image', '*.jpg'))])
        depths = sorted([os.path.abspath(p) for p in glob.glob(os.path.join(scene_path, 'depth', '*.png'))])

        if len(images) == 0 or len(depths) == 0:
            print(f"  Skipping: no images or depths found")
            continue

        # Load poses from JSON
        json_file = os.path.join(scene_path, f'{scene_name}.json')
        if not os.path.isfile(json_file):
            print(f"  Skipping: pose file not found ({json_file})")
            continue

        poses = load_poses_from_json(json_file)

        # Align counts
        min_len = min(len(images), len(depths), len(poses))
        print(f"  Found {len(images)} images, {len(depths)} depths, {len(poses)} poses")
        print(f"  Using {min_len} frames")

        images = images[:min_len]
        depths = depths[:min_len]
        poses = poses[:min_len]

        # Apply stride for subsampling
        if stride > 1:
            images = images[::stride]
            depths = depths[::stride]
            poses = poses[::stride]
            print(f"  After stride={stride}: {len(images)} frames")

        # Scale poses
        poses[:, :3] /= DEPTH_SCALE

        # Create intrinsics list
        intrinsics = [intrinsic.copy() for _ in range(len(images))]

        # Build frame graph
        print("  Building frame graph...")
        graph = build_frame_graph(poses, depths, intrinsics)

        scene_key = f"{mode}/{scene_name}"
        scene_info[scene_key] = {
            'images': images,
            'depths': depths,
            'poses': poses,
            'intrinsics': intrinsics,
            'graph': graph
        }

        print(f"  Done: {len(images)} frames, {len(graph)} graph entries")

    # Save pickle
    if output_path is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'dpvo', 'data_readers', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        output_path = os.path.join(cache_dir, f'Redwood_{mode}.pickle')
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(scene_info, f)

    print(f"Done! Total scenes: {len(scene_info)}")
    return scene_info


def main():
    parser = argparse.ArgumentParser(description='Build Redwood dataset pickle')
    parser.add_argument('--datapath', type=str, default='datasets/redwood',
                        help='Path to Redwood dataset root')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--output', type=str, default='../../datasets/redwood/',
                        help='Output pickle path')
    parser.add_argument('--stride', type=int, default=1,
                        help='Frame subsampling stride (higher = faster build)')
    parser.add_argument('--all', action='store_true',
                        help='Build all splits (train, validation, test)')

    args = parser.parse_args()

    if args.all:
        for mode in ['train', 'validation', 'test']:
            print(f"\n{'='*60}")
            print(f"Building {mode} split")
            print('='*60)
            build_redwood_pickle(args.datapath, mode, stride=args.stride)
    else:
        build_redwood_pickle(args.datapath, args.mode, args.output, args.stride)


if __name__ == '__main__':
    main()
