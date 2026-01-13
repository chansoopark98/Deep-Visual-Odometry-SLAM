#!/usr/bin/env python3
"""
Benchmark script for comparing original vs optimized data loader pipelines.

Usage:
    python benchmark_dataloader.py --dataset tartan --datapath datasets/TartanAir --num_samples 100
"""

import os
import sys
import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2


# =============================================================================
# Original Pipeline Components (from base.py, augmentation.py)
# =============================================================================

class OriginalAugmentor:
    """Original augmentation (uses PIL conversion)"""

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = T.Compose([
            T.ToPILImage(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2/3.14),
            T.RandomGrayscale(p=0.1),
            T.RandomInvert(p=0.1),
            T.ToTensor()
        ])
        self.max_scale = 0.5

    def spatial_transform(self, images, depths, poses, intrinsics):
        ht, wd = images.shape[2:]

        scale = 1
        if np.random.rand() < 0.8:
            scale = 2 ** np.random.uniform(0.0, self.max_scale)

        intrinsics = scale * intrinsics
        ht1, wd1 = int(scale * ht), int(scale * wd)

        depths = depths.unsqueeze(dim=1)
        images = F.interpolate(images, (ht1, wd1), mode='bicubic', align_corners=False)
        depths = F.interpolate(depths, (ht1, wd1), recompute_scale_factor=False)

        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths.squeeze(dim=1)

        return images, poses, depths, intrinsics

    def color_transform(self, images):
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, poses, depths, intrinsics):
        if np.random.rand() < 0.5:
            images = self.color_transform(images)
        return self.spatial_transform(images, depths, poses, intrinsics)


def original_load_sample(image_paths, depth_paths, poses_list, intrinsics_list):
    """Original sequential loading"""
    images, depths, poses, intrinsics = [], [], [], []

    for i in range(len(image_paths)):
        # cv2.imread - synchronous I/O
        img = cv2.imread(image_paths[i])
        images.append(img)

        # np.load - synchronous I/O
        depth = np.load(depth_paths[i])
        depths.append(depth)

        poses.append(poses_list[i])
        intrinsics.append(intrinsics_list[i])

    # Multiple type conversions
    images = np.stack(images).astype(np.float32)
    depths = np.stack(depths).astype(np.float32)
    poses = np.stack(poses).astype(np.float32)
    intrinsics = np.stack(intrinsics).astype(np.float32)

    images = torch.from_numpy(images).float()
    images = images.permute(0, 3, 1, 2)
    disps = torch.from_numpy(1.0 / depths)
    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)

    return images, poses, disps, intrinsics


# =============================================================================
# Optimized Pipeline Components
# =============================================================================

class OptimizedAugmentor:
    """Optimized augmentation using native torch operations (no PIL)"""

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.max_scale = 0.5

        # Pre-compute jitter parameters
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.2 / 3.14

    def color_transform(self, images):
        """Native torch color augmentation without PIL conversion"""
        # images: [N, C, H, W], range [0, 255]
        images = images / 255.0

        # Apply color jitter using torch operations
        if np.random.rand() < 0.5:
            # Brightness
            brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            images = images * brightness_factor

        if np.random.rand() < 0.5:
            # Contrast
            contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            mean = images.mean(dim=(2, 3), keepdim=True)
            images = (images - mean) * contrast_factor + mean

        if np.random.rand() < 0.5:
            # Saturation
            saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            gray = 0.2989 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            images = gray + saturation_factor * (images - gray)

        # Grayscale (10% chance)
        if np.random.rand() < 0.1:
            gray = 0.2989 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            images = gray.expand_as(images)

        # Invert (10% chance)
        if np.random.rand() < 0.1:
            images = 1.0 - images

        return (images * 255.0).clamp(0, 255)

    def spatial_transform(self, images, depths, poses, intrinsics):
        """Optimized spatial transform"""
        ht, wd = images.shape[2:]

        scale = 1.0
        if np.random.rand() < 0.8:
            scale = 2 ** np.random.uniform(0.0, self.max_scale)

        intrinsics = scale * intrinsics
        ht1, wd1 = int(scale * ht), int(scale * wd)

        # Use faster interpolation mode for depth
        depths = depths.unsqueeze(1)
        images = F.interpolate(images, (ht1, wd1), mode='bilinear', align_corners=False)
        depths = F.interpolate(depths, (ht1, wd1), mode='nearest')

        y0 = (ht1 - self.crop_size[0]) // 2
        x0 = (wd1 - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0], device=intrinsics.device)
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]].squeeze(1)

        return images, poses, depths, intrinsics

    def __call__(self, images, poses, depths, intrinsics):
        if np.random.rand() < 0.5:
            images = self.color_transform(images)
        return self.spatial_transform(images, depths, poses, intrinsics)


def load_image_optimized(path):
    """Optimized image loading"""
    # cv2.IMREAD_COLOR is faster than default
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def load_depth_optimized(path, depth_scale=5.0):
    """Optimized depth loading with mmap"""
    # Use mmap_mode for memory efficiency on large files
    depth = np.load(path, mmap_mode='r') / depth_scale
    # Convert to float32 immediately
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.where(np.isnan(depth) | np.isinf(depth), 1.0, depth)
    return depth


def optimized_load_sample_parallel(image_paths, depth_paths, poses_list, intrinsics_list,
                                   depth_scale=5.0, num_workers=4):
    """Optimized parallel loading using ThreadPoolExecutor"""
    n = len(image_paths)

    # Pre-allocate numpy arrays
    images = []
    depths = []

    # Parallel I/O for images and depths
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all image loading tasks
        image_futures = [executor.submit(load_image_optimized, p) for p in image_paths]
        depth_futures = [executor.submit(load_depth_optimized, p, depth_scale) for p in depth_paths]

        # Collect results in order
        images = [f.result() for f in image_futures]
        depths = [f.result() for f in depth_futures]

    # Stack and convert in one step
    images = np.stack(images, dtype=np.float32)
    depths = np.stack(depths, dtype=np.float32)
    poses = np.stack(poses_list, dtype=np.float32)
    intrinsics = np.stack(intrinsics_list, dtype=np.float32)

    # Convert to torch tensors efficiently
    images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    disps = torch.from_numpy(1.0 / depths)
    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)

    return images, poses, disps, intrinsics


def optimized_load_sample_sequential(image_paths, depth_paths, poses_list, intrinsics_list,
                                     depth_scale=5.0):
    """Optimized sequential loading (for comparison)"""
    n = len(image_paths)

    # Pre-allocate
    first_img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    h, w, c = first_img.shape

    images = np.empty((n, h, w, c), dtype=np.float32)
    images[0] = first_img

    first_depth = np.load(depth_paths[0], mmap_mode='r') / depth_scale
    dh, dw = first_depth.shape
    depths = np.empty((n, dh, dw), dtype=np.float32)
    depths[0] = np.where(np.isnan(first_depth) | np.isinf(first_depth), 1.0, first_depth)

    for i in range(1, n):
        images[i] = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        depth = np.load(depth_paths[i], mmap_mode='r') / depth_scale
        depths[i] = np.where(np.isnan(depth) | np.isinf(depth), 1.0, depth)

    poses = np.stack(poses_list, dtype=np.float32)
    intrinsics = np.stack(intrinsics_list, dtype=np.float32)

    images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    disps = torch.from_numpy(1.0 / depths)
    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)

    return images, poses, disps, intrinsics


# =============================================================================
# Benchmark Functions
# =============================================================================

def get_sample_data(datapath, dataset_type='tartan', n_frames=15):
    """Get sample file paths for benchmarking"""
    import glob

    if dataset_type == 'tartan':
        scenes = glob.glob(os.path.join(datapath, '*/*/*/*'))
        if not scenes:
            raise ValueError(f"No scenes found in {datapath}")

        scene = sorted(scenes)[0]
        image_paths = sorted(glob.glob(os.path.join(scene, 'image_left/*.png')))[:n_frames]
        depth_paths = sorted(glob.glob(os.path.join(scene, 'depth_left/*.npy')))[:n_frames]

        poses = np.loadtxt(os.path.join(scene, 'pose_left.txt'), delimiter=' ')[:n_frames]
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
        poses[:, :3] /= 5.0  # DEPTH_SCALE

        intrinsics = [np.array([320.0, 320.0, 320.0, 240.0])] * n_frames
        depth_scale = 5.0

    elif dataset_type == 'redwood':
        scenes = glob.glob(os.path.join(datapath, 'train/*'))
        if not scenes:
            raise ValueError(f"No scenes found in {datapath}/train")

        scene = sorted(scenes)[0]
        image_paths = sorted(glob.glob(os.path.join(scene, 'image/*.jpg')))[:n_frames]
        depth_paths = sorted(glob.glob(os.path.join(scene, 'depth/*.png')))[:n_frames]

        # Placeholder poses/intrinsics for benchmark
        poses = [np.eye(4)[:, :7].flatten()[:7]] * n_frames
        intrinsics = [np.array([525.0, 525.0, 319.5, 239.5])] * n_frames
        depth_scale = 1.0
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    return image_paths, depth_paths, poses, intrinsics, depth_scale


def benchmark_loading(image_paths, depth_paths, poses, intrinsics, depth_scale,
                      num_iterations=100, warmup=10):
    """Benchmark different loading methods"""
    results = {}

    print(f"\nBenchmarking with {len(image_paths)} frames, {num_iterations} iterations")
    print("=" * 60)

    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        _ = original_load_sample(image_paths, depth_paths, poses, intrinsics)

    # 1. Original sequential loading
    print("\n[1] Original Sequential Loading...")
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        images, poses_t, disps, intr = original_load_sample(
            image_paths, depth_paths, poses, intrinsics
        )
        times.append(time.perf_counter() - start)

    results['original_sequential'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
    }
    print(f"   Mean: {results['original_sequential']['mean']:.2f} ms")
    print(f"   Std:  {results['original_sequential']['std']:.2f} ms")

    # 2. Optimized sequential loading
    print("\n[2] Optimized Sequential Loading...")
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        images, poses_t, disps, intr = optimized_load_sample_sequential(
            image_paths, depth_paths, poses, intrinsics, depth_scale
        )
        times.append(time.perf_counter() - start)

    results['optimized_sequential'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
    }
    print(f"   Mean: {results['optimized_sequential']['mean']:.2f} ms")
    print(f"   Std:  {results['optimized_sequential']['std']:.2f} ms")

    # 3. Optimized parallel loading
    for num_workers in [2, 4, 8]:
        print(f"\n[3] Optimized Parallel Loading (workers={num_workers})...")
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            images, poses_t, disps, intr = optimized_load_sample_parallel(
                image_paths, depth_paths, poses, intrinsics, depth_scale, num_workers
            )
            times.append(time.perf_counter() - start)

        key = f'optimized_parallel_{num_workers}'
        results[key] = {
            'mean': np.mean(times) * 1000,
            'std': np.std(times) * 1000,
            'min': np.min(times) * 1000,
            'max': np.max(times) * 1000,
        }
        print(f"   Mean: {results[key]['mean']:.2f} ms")
        print(f"   Std:  {results[key]['std']:.2f} ms")

    return results


def benchmark_augmentation(images, poses, disps, intrinsics,
                           num_iterations=100, warmup=10):
    """Benchmark augmentation methods"""
    results = {}
    crop_size = [480, 640]

    print(f"\nBenchmarking Augmentation ({num_iterations} iterations)")
    print("=" * 60)

    # 1. Original augmentation
    print("\n[1] Original Augmentation (with PIL)...")
    aug_orig = OriginalAugmentor(crop_size)

    # Warmup
    for _ in range(warmup):
        _ = aug_orig(images.clone(), poses.clone(), disps.clone(), intrinsics.clone())

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = aug_orig(images.clone(), poses.clone(), disps.clone(), intrinsics.clone())
        times.append(time.perf_counter() - start)

    results['original_augmentation'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
    }
    print(f"   Mean: {results['original_augmentation']['mean']:.2f} ms")
    print(f"   Std:  {results['original_augmentation']['std']:.2f} ms")

    # 2. Optimized augmentation
    print("\n[2] Optimized Augmentation (native torch)...")
    aug_opt = OptimizedAugmentor(crop_size)

    # Warmup
    for _ in range(warmup):
        _ = aug_opt(images.clone(), poses.clone(), disps.clone(), intrinsics.clone())

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = aug_opt(images.clone(), poses.clone(), disps.clone(), intrinsics.clone())
        times.append(time.perf_counter() - start)

    results['optimized_augmentation'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
    }
    print(f"   Mean: {results['optimized_augmentation']['mean']:.2f} ms")
    print(f"   Std:  {results['optimized_augmentation']['std']:.2f} ms")

    return results


def print_summary(loading_results, aug_results):
    """Print benchmark summary"""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print("\n[Loading Times (ms)]")
    print("-" * 40)
    baseline = loading_results['original_sequential']['mean']
    for name, result in loading_results.items():
        speedup = baseline / result['mean']
        print(f"  {name:30s}: {result['mean']:7.2f} ms (x{speedup:.2f})")

    print("\n[Augmentation Times (ms)]")
    print("-" * 40)
    baseline = aug_results['original_augmentation']['mean']
    for name, result in aug_results.items():
        speedup = baseline / result['mean']
        print(f"  {name:30s}: {result['mean']:7.2f} ms (x{speedup:.2f})")

    # Best loading method
    best_loading = min(loading_results.items(), key=lambda x: x[1]['mean'])
    best_aug = min(aug_results.items(), key=lambda x: x[1]['mean'])

    print("\n[Recommendations]")
    print("-" * 40)
    print(f"  Best loading method: {best_loading[0]}")
    print(f"  Best augmentation:   {best_aug[0]}")

    total_orig = loading_results['original_sequential']['mean'] + aug_results['original_augmentation']['mean']
    total_opt = best_loading[1]['mean'] + best_aug[1]['mean']
    print(f"\n  Total original:  {total_orig:.2f} ms/sample")
    print(f"  Total optimized: {total_opt:.2f} ms/sample")
    print(f"  Overall speedup: x{total_orig/total_opt:.2f}")


def benchmark_dataloader_throughput(datapath, dataset_type='tartan', n_frames=15,
                                    num_samples=50, batch_size=1):
    """Benchmark actual DataLoader throughput with different num_workers"""
    from torch.utils.data import DataLoader, Dataset

    class SimpleDataset(Dataset):
        """Minimal dataset for benchmarking"""
        def __init__(self, image_paths, depth_paths, poses, intrinsics, depth_scale,
                     use_parallel=False, num_threads=4):
            self.image_paths = image_paths
            self.depth_paths = depth_paths
            self.poses = poses
            self.intrinsics = intrinsics
            self.depth_scale = depth_scale
            self.use_parallel = use_parallel
            self.num_threads = num_threads

        def __len__(self):
            return 100  # Simulate larger dataset

        def __getitem__(self, idx):
            idx = idx % len(self.image_paths)
            if self.use_parallel:
                return optimized_load_sample_parallel(
                    self.image_paths, self.depth_paths,
                    self.poses, self.intrinsics,
                    self.depth_scale, self.num_threads
                )
            else:
                return optimized_load_sample_sequential(
                    self.image_paths, self.depth_paths,
                    self.poses, self.intrinsics,
                    self.depth_scale
                )

    # Get sample data
    image_paths, depth_paths, poses, intrinsics, depth_scale = get_sample_data(
        datapath, dataset_type, n_frames
    )

    results = {}

    print(f"\nBenchmarking DataLoader Throughput ({num_samples} samples)")
    print("=" * 60)

    configs = [
        # (num_workers, use_parallel_in_getitem, threads_per_getitem, description)
        (0, False, 0, "Sequential (num_workers=0)"),
        (0, True, 4, "Parallel in __getitem__ (num_workers=0, 4 threads)"),
        (2, False, 0, "DataLoader parallel (num_workers=2)"),
        (4, False, 0, "DataLoader parallel (num_workers=4)"),
        (8, False, 0, "DataLoader parallel (num_workers=8)"),
        (4, True, 2, "Hybrid (num_workers=4, 2 threads each)"),
    ]

    for num_workers, use_parallel, num_threads, desc in configs:
        print(f"\n[{desc}]")

        dataset = SimpleDataset(
            image_paths, depth_paths, poses, intrinsics, depth_scale,
            use_parallel=use_parallel, num_threads=num_threads
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # Warmup
        for i, batch in enumerate(loader):
            if i >= 5:
                break

        # Benchmark
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
        elapsed = time.perf_counter() - start

        samples_per_sec = num_samples / elapsed
        ms_per_sample = elapsed / num_samples * 1000

        results[desc] = {
            'samples_per_sec': samples_per_sec,
            'ms_per_sample': ms_per_sample,
        }
        print(f"   {samples_per_sec:.2f} samples/sec ({ms_per_sample:.2f} ms/sample)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark data loader pipeline')
    parser.add_argument('--dataset', type=str, default='tartan', choices=['tartan', 'redwood'])
    parser.add_argument('--datapath', type=str, default='datasets/TartanAir')
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'loading', 'augmentation', 'dataloader'])
    args = parser.parse_args()

    print("Data Loader Pipeline Benchmark")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Path: {args.datapath}")
    print(f"Frames: {args.n_frames}")

    # Get sample data
    image_paths, depth_paths, poses, intrinsics, depth_scale = get_sample_data(
        args.datapath, args.dataset, args.n_frames
    )

    print(f"\nSample data:")
    print(f"  Images: {len(image_paths)}")
    print(f"  First image: {image_paths[0]}")

    if args.mode in ['all', 'loading']:
        # Benchmark loading
        loading_results = benchmark_loading(
            image_paths, depth_paths, poses, intrinsics, depth_scale,
            args.num_iterations, args.warmup
        )
    else:
        loading_results = None

    if args.mode in ['all', 'augmentation']:
        # Load sample for augmentation benchmark
        images, poses_t, disps, intrinsics_t = original_load_sample(
            image_paths, depth_paths, poses, intrinsics
        )

        # Benchmark augmentation
        aug_results = benchmark_augmentation(
            images, poses_t, disps, intrinsics_t,
            args.num_iterations, args.warmup
        )
    else:
        aug_results = None

    if args.mode in ['all', 'dataloader']:
        # Benchmark DataLoader throughput
        dataloader_results = benchmark_dataloader_throughput(
            args.datapath, args.dataset, args.n_frames
        )
    else:
        dataloader_results = None

    # Print summary
    if loading_results and aug_results:
        print_summary(loading_results, aug_results)

    if dataloader_results:
        print("\n" + "=" * 60)
        print("DATALOADER THROUGHPUT SUMMARY")
        print("=" * 60)
        baseline = list(dataloader_results.values())[0]['samples_per_sec']
        for name, result in dataloader_results.items():
            speedup = result['samples_per_sec'] / baseline
            print(f"  {name:45s}: {result['samples_per_sec']:6.2f} samples/sec (x{speedup:.2f})")


if __name__ == '__main__':
    main()
