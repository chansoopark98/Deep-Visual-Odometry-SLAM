#!/usr/bin/env python3
"""
Verify that native torch augmentation produces equivalent results to PIL-based augmentation.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np


class PILAugmentor:
    """Original PIL-based augmentation for comparison"""

    def __init__(self):
        self.augcolor = T.Compose([
            T.ToPILImage(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2/3.14),
            T.RandomGrayscale(p=0.1),
            T.RandomInvert(p=0.1),
            T.ToTensor()
        ])

    def color_transform(self, images):
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd * num)
        images = 255 * self.augcolor(images[[2, 1, 0]] / 255.0)
        return images[[2, 1, 0]].reshape(ch, ht, wd, num).permute(3, 0, 1, 2).contiguous()


class NativeAugmentor:
    """Native torch augmentation (new implementation)"""

    def __init__(self):
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.2 / 3.14

    def color_transform(self, images, fn_idx, brightness_factor, contrast_factor,
                        saturation_factor, hue_factor, do_grayscale, do_invert):
        """Apply with fixed parameters for reproducibility"""
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd * num)
        images = images[[2, 1, 0]] / 255.0

        for fn_id in fn_idx:
            if fn_id == 0:
                images = TF.adjust_brightness(images, brightness_factor)
            elif fn_id == 1:
                images = TF.adjust_contrast(images, contrast_factor)
            elif fn_id == 2:
                images = TF.adjust_saturation(images, saturation_factor)
            elif fn_id == 3:
                images = TF.adjust_hue(images, hue_factor)

        if do_grayscale:
            images = TF.rgb_to_grayscale(images, num_output_channels=3)

        if do_invert:
            images = TF.invert(images)

        images = images[[2, 1, 0]] * 255.0
        return images.reshape(ch, ht, wd, num).permute(3, 0, 1, 2).contiguous()


def test_individual_transforms():
    """Test each transform individually for equivalence"""
    print("=" * 60)
    print("Testing Individual Transforms")
    print("=" * 60)

    # Create test image [1, 3, 64, 64]
    torch.manual_seed(42)
    test_img = torch.rand(1, 3, 64, 64) * 255.0

    # Test brightness
    print("\n[Brightness]")
    img_rgb = test_img[0][[2, 1, 0]] / 255.0  # BGR to RGB, normalize
    factor = 1.2

    # PIL way
    pil_img = T.ToPILImage()(img_rgb)
    pil_result = T.functional.adjust_brightness(pil_img, factor)
    pil_tensor = T.ToTensor()(pil_result)

    # Native way
    native_result = TF.adjust_brightness(img_rgb, factor)

    diff = (pil_tensor - native_result).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Equivalent: {diff.max().item() < 0.01}")

    # Test contrast
    print("\n[Contrast]")
    factor = 0.8

    pil_img = T.ToPILImage()(img_rgb)
    pil_result = T.functional.adjust_contrast(pil_img, factor)
    pil_tensor = T.ToTensor()(pil_result)

    native_result = TF.adjust_contrast(img_rgb, factor)

    diff = (pil_tensor - native_result).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Equivalent: {diff.max().item() < 0.01}")

    # Test saturation
    print("\n[Saturation]")
    factor = 1.3

    pil_img = T.ToPILImage()(img_rgb)
    pil_result = T.functional.adjust_saturation(pil_img, factor)
    pil_tensor = T.ToTensor()(pil_result)

    native_result = TF.adjust_saturation(img_rgb, factor)

    diff = (pil_tensor - native_result).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Equivalent: {diff.max().item() < 0.01}")

    # Test hue
    print("\n[Hue]")
    factor = 0.05

    pil_img = T.ToPILImage()(img_rgb)
    pil_result = T.functional.adjust_hue(pil_img, factor)
    pil_tensor = T.ToTensor()(pil_result)

    native_result = TF.adjust_hue(img_rgb, factor)

    diff = (pil_tensor - native_result).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Equivalent: {diff.max().item() < 0.01}")

    # Test grayscale
    print("\n[Grayscale]")
    pil_img = T.ToPILImage()(img_rgb)
    pil_result = T.Grayscale(num_output_channels=3)(pil_img)
    pil_tensor = T.ToTensor()(pil_result)

    native_result = TF.rgb_to_grayscale(img_rgb, num_output_channels=3)

    diff = (pil_tensor - native_result).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Equivalent: {diff.max().item() < 0.01}")

    # Test invert
    print("\n[Invert]")
    pil_img = T.ToPILImage()(img_rgb)
    pil_result = T.functional.invert(pil_img)
    pil_tensor = T.ToTensor()(pil_result)

    native_result = TF.invert(img_rgb)

    diff = (pil_tensor - native_result).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Equivalent: {diff.max().item() < 0.01}")


def test_value_range():
    """Test that output values are in valid range"""
    print("\n" + "=" * 60)
    print("Testing Value Range")
    print("=" * 60)

    from dpvo.data_readers.augmentation import RGBDAugmentor

    aug = RGBDAugmentor(crop_size=[480, 640])

    # Create test images
    torch.manual_seed(42)
    images = torch.rand(15, 3, 480, 640) * 255.0

    for i in range(10):
        result = aug.color_transform(images.clone())
        min_val = result.min().item()
        max_val = result.max().item()
        print(f"  Run {i+1}: min={min_val:.2f}, max={max_val:.2f}, valid={0 <= min_val and max_val <= 255}")


def test_output_shape():
    """Test that output shapes are preserved"""
    print("\n" + "=" * 60)
    print("Testing Output Shape")
    print("=" * 60)

    from dpvo.data_readers.augmentation import RGBDAugmentor

    aug = RGBDAugmentor(crop_size=[480, 640])

    # Test various input shapes
    for n_frames in [4, 8, 15]:
        images = torch.rand(n_frames, 3, 480, 640) * 255.0
        result = aug.color_transform(images.clone())

        print(f"  Input: {images.shape} -> Output: {result.shape}, Match: {images.shape == result.shape}")


def test_statistical_distribution():
    """Test that statistical properties are similar"""
    print("\n" + "=" * 60)
    print("Testing Statistical Distribution")
    print("=" * 60)

    from dpvo.data_readers.augmentation import RGBDAugmentor

    aug = RGBDAugmentor(crop_size=[480, 640])

    # Run many augmentations and check distribution
    torch.manual_seed(42)
    np.random.seed(42)

    images = torch.rand(15, 3, 480, 640) * 255.0

    mean_diffs = []
    std_diffs = []

    for _ in range(100):
        result = aug.color_transform(images.clone())
        mean_diffs.append((result.mean() - images.mean()).item())
        std_diffs.append((result.std() - images.std()).item())

    print(f"  Mean change - avg: {np.mean(mean_diffs):.2f}, std: {np.std(mean_diffs):.2f}")
    print(f"  Std change  - avg: {np.mean(std_diffs):.2f}, std: {np.std(std_diffs):.2f}")


if __name__ == '__main__':
    test_individual_transforms()
    test_output_shape()
    test_value_range()
    test_statistical_distribution()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
