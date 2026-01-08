
import pickle
import os
import os.path as osp

# RGBD-Datasets
from .tartan import TartanAir
from .redwood import Redwood


def dataset_factory(dataset_list, **kwargs):
    """
    Create a combined dataset from multiple sources.

    Args:
        dataset_list: list of dataset names, e.g., ['tartan'], ['redwood'], ['tartan', 'redwood']
        **kwargs: arguments passed to dataset constructors
            - datapath: root path to dataset
            - n_frames: number of frames per sample
            - crop_size: output image size [H, W]
            - fmin, fmax: optical flow thresholds
            - aug: whether to apply augmentation
            - mode: 'train', 'validation', 'test' (for Redwood)

    Returns:
        ConcatDataset combining all requested datasets

    Example:
        # Single dataset
        db = dataset_factory(['tartan'], datapath='datasets/TartanAir', n_frames=15)

        # Multiple datasets (requires separate datapaths)
        db = dataset_factory(['redwood'], datapath='datasets/redwood', n_frames=15, mode='train')

        # Combined training
        tartan_db = dataset_factory(['tartan'], datapath='datasets/TartanAir', n_frames=15)
        redwood_db = dataset_factory(['redwood'], datapath='datasets/redwood', n_frames=15, mode='train')
        combined_db = ConcatDataset([tartan_db, redwood_db])
    """
    from torch.utils.data import ConcatDataset

    dataset_map = {
        'tartan': TartanAir,
        'redwood': Redwood,
    }

    db_list = []
    for key in dataset_list:
        if key not in dataset_map:
            raise ValueError(f"Unknown dataset: {key}. Available: {list(dataset_map.keys())}")

        db = dataset_map[key](**kwargs)
        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)
