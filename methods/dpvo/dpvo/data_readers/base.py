import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from .augmentation import RGBDAugmentor
from .rgbd_utils import *

class RGBDDataset(data.Dataset):
    # Subclasses should define these
    DEPTH_SCALE = 1.0
    CACHE_NAME = None  # e.g., 'TartanAir', 'Redwood'

    def __init__(self, name, datapath, n_frames=4, crop_size=[480,640], fmin=10.0, fmax=75.0, aug=True, sample=True):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name

        self.aug = aug
        self.sample = sample

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples

        if self.aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path = osp.dirname(osp.abspath(__file__))
        cache_dir = osp.join(cur_path, 'cache')
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        # Load or build scene_info
        self.scene_info = self._load_or_build_scene_info(cache_dir)

        self._build_dataset_index()

    def _load_or_build_scene_info(self, cache_dir):
        """Load scene_info from cache or build it. Subclasses can override."""
        cache_name = self.__class__.CACHE_NAME or self.name
        cache_path = osp.join(cache_dir, f'{cache_name}.pickle')

        if osp.isfile(cache_path):
            print(f"Loading {cache_name} from cache: {cache_path}")
            return pickle.load(open(cache_path, 'rb'))
        else:
            print(f"Building {cache_name} dataset (this may take a while)...")
            scene_info = self._build_dataset()
            pickle.dump(scene_info, open(cache_path, 'wb'))
            print(f"Saved cache to: {cache_path}")
            return scene_info

    def _build_dataset(self):
        """Build scene_info dictionary. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_dataset()")
                
    def _build_dataset_index(self):
        self.dataset_index = []
        for scene in self.scene_info:
            if not self.__class__.is_test_scene(scene):
                graph = self.scene_info[scene]['graph']
                for i in graph:
                    if i < len(graph) - 65:
                        self.dataset_index.append((scene, i))
            else:
                print("Reserving {} for validation".format(scene))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index):
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        # stride = np.random.choice([1,2,3])

        d = np.random.uniform(self.fmin, self.fmax)
        s = 1

        inds = [ ix ]

        while len(inds) < self.n_frames:
            # get other frames within flow threshold

            if self.sample:
                k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
                frames = frame_graph[ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.random.choice(frames[frames > ix])

                elif ix + 1 < len(images_list):
                    ix = ix + 1

                elif np.count_nonzero(frames):
                    ix = np.random.choice(frames)

            else:
                i = frame_graph[ix][0].copy()
                g = frame_graph[ix][1].copy()

                g[g > d] = -1
                if s > 0:
                    g[i <= ix] = -1
                else:
                    g[i >= ix] = -1

                if len(g) > 0 and np.max(g) > 0:
                    ix = i[np.argmax(g)]
                else:
                    if ix + s >= len(images_list) or ix + s < 0:
                        s *= -1

                    ix = ix + s
            
            inds += [ ix ]


        images, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.aug:
            images, poses, disps, intrinsics = \
                self.aug(images, poses, disps, intrinsics)

        # normalize depth
        s = .7 * torch.quantile(disps, .98)
        disps = disps / s
        poses[...,:3] *= s

        return images, poses, disps, intrinsics 

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self
