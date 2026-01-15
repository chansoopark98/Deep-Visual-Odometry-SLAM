
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from scipy.spatial.transform import Rotation
from .base import RGBDDataset

class Redwood(RGBDDataset):
    """
    Redwood Indoor RGBD Dataset

    Directory structure:
        datasets/redwood/
        ├── intrinsic.npy           # 3x3 intrinsic matrix
        ├── train/
        │   └── {scene}/
        │       ├── image/          # RGB jpg images (480x640)
        │       ├── depth/          # Depth png (480x640, uint16, mm)
        │       └── {scene}.log     # Pose log (4x4 matrices)
        ├── validation/
        └── test/
    """

    # Depth is stored in mm (uint16), convert to meters then scale
    DEPTH_SCALE = 1.0  # Redwood depth is already in reasonable scale
    DEPTH_MM_TO_M = 1000.0  # mm to meters conversion

    def __init__(self, mode='train', **kwargs):
        """
        Args:
            mode: 'train', 'validation', or 'test'
            datapath: path to redwood dataset root
            n_frames: number of frames per training sample
            crop_size: output image size [H, W]
            fmin, fmax: optical flow thresholds for frame sampling
            aug: whether to apply augmentation
        """
        self.mode = mode
        # Set CACHE_NAME based on mode (each mode has its own pickle)
        self.__class__.CACHE_NAME = f'Redwood_{mode}'
        super(Redwood, self).__init__(name='Redwood', **kwargs)

    def _load_or_build_scene_info(self, _cache_dir):
        """
        Load scene_info from cache in dataset directory.
        Override base class to use datasets/redwood/cache instead of dpvo/data_readers/cache.
        Note: _cache_dir is ignored, using self.root/cache instead.
        """
        import pickle

        cache_name = self.__class__.CACHE_NAME
        # Use cache directory inside dataset root (e.g., datasets/redwood/cache/)
        dataset_cache_dir = osp.join(self.root, 'cache')

        if not osp.isdir(dataset_cache_dir):
            os.makedirs(dataset_cache_dir, exist_ok=True)

        cache_path = osp.join(dataset_cache_dir, f'{cache_name}.pickle')

        if osp.isfile(cache_path):
            print(f"Loading {cache_name} from cache: {cache_path}")
            return pickle.load(open(cache_path, 'rb'))
        else:
            print(f"Building {cache_name} dataset (this may take a while)...")
            print(f"Hint: Use scripts/build_redwood_pickle.py for faster building with stride option")
            scene_info = self._build_dataset()
            pickle.dump(scene_info, open(cache_path, 'wb'))
            print(f"Saved cache to: {cache_path}")
            return scene_info

    @staticmethod
    def is_test_scene(scene):
        """Check if scene is reserved for testing."""
        # Redwood uses folder-based splits, so we filter by mode in _build_dataset
        return False

    def _build_dataset(self):
        """Build scene_info dictionary for Redwood dataset."""
        from tqdm import tqdm
        print(f"Building Redwood dataset (mode={self.mode})")

        scene_info = {}

        # Get scenes for current mode
        mode_path = osp.join(self.root, self.mode)
        if not osp.isdir(mode_path):
            raise ValueError(f"Mode path does not exist: {mode_path}")

        scenes = sorted(glob.glob(osp.join(mode_path, '*')))

        # Load shared intrinsic
        intrinsic_path = osp.join(self.root, 'intrinsic.npy')
        if osp.isfile(intrinsic_path):
            intrinsic_matrix = np.load(intrinsic_path)
            # Convert 3x3 matrix to [fx, fy, cx, cy]
            intrinsic = np.array([
                intrinsic_matrix[0, 0],  # fx
                intrinsic_matrix[1, 1],  # fy
                intrinsic_matrix[0, 2],  # cx
                intrinsic_matrix[1, 2],  # cy
            ])
        else:
            # Default PrimeSense intrinsic
            intrinsic = np.array([525.0, 525.0, 319.5, 239.5])
            print(f"Warning: intrinsic.npy not found, using default: {intrinsic}")

        for scene_path in tqdm(sorted(scenes)):
            scene_name = osp.basename(scene_path)

            # Get image and depth files (use absolute paths)
            images = sorted([osp.abspath(p) for p in glob.glob(osp.join(scene_path, 'image', '*.jpg'))])
            depths = sorted([osp.abspath(p) for p in glob.glob(osp.join(scene_path, 'depth', '*.png'))])

            if len(images) == 0 or len(depths) == 0:
                print(f"Skipping {scene_name}: no images or depths found")
                continue

            if len(images) != len(depths):
                # Use minimum length
                min_len = min(len(images), len(depths))
                print(f"Warning: {scene_name} has mismatched images ({len(images)}) and depths ({len(depths)}), using {min_len}")
                images = images[:min_len]
                depths = depths[:min_len]

            # Load poses from JSON file
            json_file = osp.join(scene_path, f'{scene_name}.json')
            if not osp.isfile(json_file):
                print(f"Skipping {scene_name}: pose file not found ({json_file})")
                continue

            poses = self._load_poses_from_json(json_file)

            if len(poses) < len(images):
                print(f"Warning: {scene_name} has fewer poses ({len(poses)}) than images ({len(images)})")
                images = images[:len(poses)]
                depths = depths[:len(poses)]
            elif len(poses) > len(images):
                poses = poses[:len(images)]

            # Scale poses for depth normalization
            poses[:, :3] /= Redwood.DEPTH_SCALE

            # Create intrinsics list
            intrinsics = [intrinsic.copy() for _ in range(len(images))]

            # Build frame graph for co-visibility based sampling
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene_key = f"{self.mode}/{scene_name}"
            scene_info[scene_key] = {
                'images': images,
                'depths': depths,
                'poses': poses,
                'intrinsics': intrinsics,
                'graph': graph
            }

            print(f"  {scene_name}: {len(images)} frames")

        return scene_info

    def _load_poses_from_json(self, json_file):
        """
        Load poses from Redwood JSON file format.

        The JSON contains a PoseGraph with nodes, each node has a 'pose' field
        containing a 4x4 transformation matrix stored as a 16-element array
        in column-major order.

        Returns:
            poses: np.array of shape (N, 7) with [tx, ty, tz, qx, qy, qz, qw]
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

            # Convert 4x4 matrix to [tx, ty, tz, qx, qy, qz, qw]
            pose = self._matrix_to_pose(matrix)
            poses.append(pose)

        return np.array(poses, dtype=np.float32)

    @staticmethod
    def _matrix_to_pose(T):
        """
        Convert 4x4 transformation matrix to pose vector.

        Args:
            T: 4x4 transformation matrix

        Returns:
            pose: [tx, ty, tz, qx, qy, qz, qw]
        """
        t = T[:3, 3]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # returns [qx, qy, qz, qw]
        return np.concatenate([t, q])

    @staticmethod
    def calib_read():
        """Return default intrinsic parameters [fx, fy, cx, cy]."""
        return np.array([525.0, 525.0, 319.5, 239.5])

    @staticmethod
    def image_read(image_file):
        """Read RGB image."""
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        """
        Read depth image.

        Redwood depth is stored as uint16 PNG in millimeters.
        Convert to meters and handle invalid values.
        """
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / Redwood.DEPTH_MM_TO_M  # mm to meters

        # Handle invalid depth values
        depth[depth == 0] = 1.0  # Invalid depth (no reading)
        depth[np.isnan(depth)] = 1.0
        depth[np.isinf(depth)] = 1.0

        depth = depth / Redwood.DEPTH_SCALE
        return depth