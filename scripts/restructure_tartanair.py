#!/usr/bin/env python3
"""
Restructure TartanAir dataset to match the expected format for DPVO training.

Current structure (downloaded):
    TartanAir/
    ├── abandonedfactory_Easy_image_left/abandonedfactory/Easy/P009/image_left/*.png
    ├── abandonedfactory_Easy_depth_left/abandonedfactory/Easy/P009/depth_left/*.npy

Expected structure (for DPVO loader):
    TartanAir/
    └── abandonedfactory/abandonedfactory/Easy/P009/
        ├── image_left/*.png
        ├── depth_left/*.npy
        └── pose_left.txt

This script moves actual files to restructure the dataset.
"""

import os
import glob
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def restructure_tartanair(dataset_root, dry_run=False):
    """Restructure TartanAir dataset by moving files."""

    dataset_root = Path(dataset_root).resolve()

    # Find all image folders
    image_folders = sorted(glob.glob(str(dataset_root / '*_image_left')))

    print(f"Found {len(image_folders)} image folders")

    moved_items = 0
    errors = []

    for img_folder in tqdm(image_folders, desc="Processing folders"):
        img_folder = Path(img_folder)
        folder_name = img_folder.name

        # Extract scene info: abandonedfactory_Easy_image_left -> abandonedfactory_Easy
        base_name = folder_name.replace('_image_left', '')
        depth_folder = dataset_root / f'{base_name}_depth_left'

        if not depth_folder.exists():
            errors.append(f"Depth folder not found: {depth_folder}")
            continue

        # Find all sequences in image folder
        # Pattern: abandonedfactory_Easy_image_left/abandonedfactory/Easy/P009/
        sequences = glob.glob(str(img_folder / '*/*/*'))

        for seq_path in sequences:
            seq_path = Path(seq_path)

            # Get relative path: abandonedfactory/Easy/P009
            rel_path = seq_path.relative_to(img_folder)

            # Corresponding depth sequence
            depth_seq_path = depth_folder / rel_path

            if not depth_seq_path.exists():
                errors.append(f"Depth sequence not found: {depth_seq_path}")
                continue

            # Target directory: TartanAir/abandonedfactory/abandonedfactory/Easy/P009/
            parts = str(rel_path).split('/')
            if len(parts) >= 3:
                scene = parts[0]
                difficulty = parts[1]
                trajectory = parts[2]

                target_dir = dataset_root / scene / scene / difficulty / trajectory

                # Source directories
                image_src = seq_path / 'image_left'
                depth_src = depth_seq_path / 'depth_left'
                pose_src = depth_seq_path / 'pose_left.txt'

                # Target directories
                image_dst = target_dir / 'image_left'
                depth_dst = target_dir / 'depth_left'
                pose_dst = target_dir / 'pose_left.txt'

                if dry_run:
                    if image_src.exists():
                        print(f"Would move: {image_src} -> {image_dst}")
                    if depth_src.exists():
                        print(f"Would move: {depth_src} -> {depth_dst}")
                    if pose_src.exists():
                        print(f"Would move: {pose_src} -> {pose_dst}")
                else:
                    # Create target directory
                    target_dir.mkdir(parents=True, exist_ok=True)

                    # Move image_left directory
                    if image_src.exists() and not image_dst.exists():
                        try:
                            shutil.move(str(image_src), str(image_dst))
                            moved_items += 1
                        except Exception as e:
                            errors.append(f"Failed to move {image_src}: {e}")

                    # Move depth_left directory
                    if depth_src.exists() and not depth_dst.exists():
                        try:
                            shutil.move(str(depth_src), str(depth_dst))
                            moved_items += 1
                        except Exception as e:
                            errors.append(f"Failed to move {depth_src}: {e}")

                    # Move pose_left.txt file
                    if pose_src.exists() and not pose_dst.exists():
                        try:
                            shutil.move(str(pose_src), str(pose_dst))
                            moved_items += 1
                        except Exception as e:
                            errors.append(f"Failed to move {pose_src}: {e}")

    print(f"\nMoved {moved_items} items")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return moved_items, errors


def cleanup_empty_folders(dataset_root, dry_run=False):
    """Remove empty folders after restructuring."""

    dataset_root = Path(dataset_root).resolve()

    # Find original download folders
    folders_to_check = sorted(glob.glob(str(dataset_root / '*_image_left')))
    folders_to_check += sorted(glob.glob(str(dataset_root / '*_depth_left')))

    removed = 0

    for folder in folders_to_check:
        folder = Path(folder)

        # Check if folder is empty (recursively)
        has_files = False
        for root, dirs, files in os.walk(folder):
            if files:
                has_files = True
                break

        if not has_files:
            if dry_run:
                print(f"Would remove empty folder: {folder}")
            else:
                try:
                    shutil.rmtree(folder)
                    removed += 1
                    print(f"Removed empty folder: {folder.name}")
                except Exception as e:
                    print(f"Failed to remove {folder}: {e}")

    print(f"\nRemoved {removed} empty folders")
    return removed


def verify_structure(dataset_root):
    """Verify the restructured dataset."""

    dataset_root = Path(dataset_root).resolve()

    # Expected pattern: scene/scene/difficulty/trajectory
    scenes = glob.glob(str(dataset_root / '*/*/*/*'))

    valid = 0
    invalid = []

    for scene in scenes:
        scene = Path(scene)

        # Skip the original downloaded folders
        if '_image_left' in str(scene) or '_depth_left' in str(scene):
            continue

        has_images = (scene / 'image_left').exists()
        has_depths = (scene / 'depth_left').exists()
        has_poses = (scene / 'pose_left.txt').exists()

        if has_images and has_depths and has_poses:
            # Count files
            num_images = len(list((scene / 'image_left').glob('*.png')))
            num_depths = len(list((scene / 'depth_left').glob('*.npy')))

            if num_images > 0 and num_depths > 0 and num_images == num_depths:
                valid += 1
            else:
                invalid.append(f"{scene}: images={num_images}, depths={num_depths}")
        else:
            missing = []
            if not has_images: missing.append('image_left')
            if not has_depths: missing.append('depth_left')
            if not has_poses: missing.append('pose_left.txt')
            invalid.append(f"{scene}: missing {missing}")

    print(f"\nVerification:")
    print(f"  Valid scenes: {valid}")
    print(f"  Invalid scenes: {len(invalid)}")

    if invalid:
        print("\nInvalid scenes (first 5):")
        for inv in invalid[:5]:
            print(f"  - {inv}")

    return valid, invalid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restructure TartanAir dataset')
    parser.add_argument('--root', type=str, default='datasets/TartanAir',
                        help='Path to TartanAir dataset')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify the structure')
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove empty folders after restructuring')

    args = parser.parse_args()

    if args.verify:
        verify_structure(args.root)
    else:
        print(f"Restructuring TartanAir dataset at: {args.root}")
        print(f"Dry run: {args.dry_run}\n")

        moved, errors = restructure_tartanair(args.root, dry_run=args.dry_run)

        if not args.dry_run:
            verify_structure(args.root)

            if args.cleanup:
                print("\nCleaning up empty folders...")
                cleanup_empty_folders(args.root, dry_run=args.dry_run)
