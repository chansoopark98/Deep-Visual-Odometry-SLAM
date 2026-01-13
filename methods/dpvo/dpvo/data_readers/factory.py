
import pickle
import os
import os.path as osp

# RGBD-Datasets
from .tartan import TartanAir
from .redwood import Redwood


DATASET_MAP = {
    'tartan': TartanAir,
    'redwood': Redwood,
}


def dataset_factory(dataset_configs, **common_kwargs):
    """
    Create a combined dataset from one or more sources.

    Args:
        dataset_configs: list of dataset configurations, each containing:
            - 'name': dataset name ('tartan' or 'redwood')
            - 'datapath': path to dataset
            - 'mode': (optional) 'train', 'validation', 'test'
        **common_kwargs: common arguments passed to all dataset constructors
            - n_frames: number of frames per sample
            - crop_size: output image size [H, W]
            - fmin, fmax: optical flow thresholds
            - aug: whether to apply augmentation

    Returns:
        ConcatDataset combining all requested datasets

    Example:
        # Single dataset
        db = dataset_factory([
            {'name': 'tartan', 'datapath': 'datasets/TartanAir'}
        ], n_frames=15)

        # Multiple datasets
        db = dataset_factory([
            {'name': 'tartan', 'datapath': 'datasets/TartanAir'},
            {'name': 'redwood', 'datapath': 'datasets/redwood', 'mode': 'train'},
        ], n_frames=15)
    """
    from torch.utils.data import ConcatDataset

    db_list = []
    for cfg in dataset_configs:
        # Copy config to avoid modifying original
        cfg = cfg.copy()
        name = cfg.pop('name')

        if name not in DATASET_MAP:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_MAP.keys())}")

        # Merge common kwargs with dataset-specific kwargs
        kwargs = {**common_kwargs, **cfg}
        db = DATASET_MAP[name](**kwargs)
        print("Dataset {} has {} images".format(name, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)
