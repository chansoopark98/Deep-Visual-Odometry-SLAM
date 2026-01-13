
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


def dataset_factory(dataset_list, **kwargs):
    """
    Create a combined dataset from multiple sources (single datapath).

    Args:
        dataset_list: list of dataset names, e.g., ['tartan'], ['redwood']
        **kwargs: arguments passed to dataset constructors

    Returns:
        ConcatDataset combining all requested datasets
    """
    from torch.utils.data import ConcatDataset

    db_list = []
    for key in dataset_list:
        if key not in DATASET_MAP:
            raise ValueError(f"Unknown dataset: {key}. Available: {list(DATASET_MAP.keys())}")

        db = DATASET_MAP[key](**kwargs)
        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)


def dataset_factory_multi(dataset_configs, common_kwargs=None):
    """
    Create a combined dataset from multiple sources with separate configs.

    Args:
        dataset_configs: list of dicts, each containing:
            - 'name': dataset name ('tartan' or 'redwood')
            - 'datapath': path to dataset
            - 'mode': (optional) 'train', 'validation', 'test'
        common_kwargs: dict of common arguments (n_frames, crop_size, etc.)

    Returns:
        ConcatDataset combining all requested datasets

    Example:
        dataset_configs = [
            {'name': 'tartan', 'datapath': 'datasets/TartanAir'},
            {'name': 'redwood', 'datapath': 'datasets/redwood', 'mode': 'train'},
        ]
        db = dataset_factory_multi(dataset_configs, {'n_frames': 15})
    """
    from torch.utils.data import ConcatDataset

    if common_kwargs is None:
        common_kwargs = {}

    db_list = []
    for cfg in dataset_configs:
        name = cfg.pop('name')
        if name not in DATASET_MAP:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_MAP.keys())}")

        # Merge common kwargs with dataset-specific kwargs
        kwargs = {**common_kwargs, **cfg}
        db = DATASET_MAP[name](**kwargs)
        print("Dataset {} has {} images".format(name, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)
