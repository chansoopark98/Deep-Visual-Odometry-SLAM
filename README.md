# Deep-Visual-Odometry-SLAM

Deep Learning based Visual Odometry and SLAM Open Project

## Features

- **DPVO (Deep Patch Visual Odometry)** based visual odometry
- **Multi-dataset support**: TartanAir, Redwood (custom datasets)
- **FP16/AMP Training**: Automatic Mixed Precision for faster training with lower memory usage
- **RTX 50 Series Support**: Optimized CUDA kernels for Blackwell architecture (sm_120)
- **Loop Closure**: Optional DBoW2-based loop closure for large-scale SLAM

## Requirements

- **GPU**: NVIDIA RTX 20/30/40/50 series
- **CUDA**: 12.8+ (for RTX 50 series)
- **Python**: 3.12
- **PyTorch**: 2.9.1+

See [INSTALL.md](methods/dpvo/INSTALL.md) for detailed installation instructions.

## Quick Start

### 1. Installation

```bash
# Create conda environment
conda create -n visual_slam python=3.12
conda activate visual_slam

# Install PyTorch
pip install torch torchvision

# Install dependencies
pip install tensorboard numba tqdm einops pypose kornia numpy plyfile evo opencv-python yacs

# Build CUDA extensions
export TORCH_CUDA_ARCH_LIST="12.0"  # Adjust for your GPU
cd methods/dpvo
pip install --no-build-isolation .
```

### 2. Dataset Preparation

**TartanAir:**
```bash
# Download TartanAir dataset and structure as:
# datasets/TartanAir/{scene}/Easy/{P00X}/
```

**Redwood (Custom):**
```bash
# Build pose pickle for Redwood dataset
python methods/dpvo/scripts/build_redwood_pickle.py \
    --datapath datasets/redwood \
    --mode train \
    --output datasets/redwood/cache/Redwood_train.pickle
```

### 3. Training

```bash
# Train on TartanAir
python methods/dpvo/train.py --config methods/dpvo/config/tartan_train.yaml

# Fine-tune on Redwood
python methods/dpvo/train.py --config methods/dpvo/config/redwood_train.yaml
```

### 4. Demo / Evaluation

```bash
# Run demo with visualization
python methods/dpvo/demo.py \
    --imagedir=<path_to_images> \
    --calib=<path_to_calibration> \
    --stride=1 --viz

# Evaluate on TartanAir
python methods/dpvo/evaluate_tartan.py \
    --weights=checkpoints/dpvo.pth \
    --datapath=datasets/TartanAir
```

## Training Configuration

Training is configured via YAML files. Key options:

```yaml
training:
  name: experiment_name
  steps: 240000
  lr: 0.0008
  amp: true  # Enable FP16 training

dataloader:
  batch_size: 1
  num_workers: 8  # Parallel data loading
```

### AMP (Automatic Mixed Precision)

FP16 training is enabled by setting `amp: true` in the config. This provides:
- ~30% faster training
- ~40% less GPU memory usage
- Maintained numerical accuracy (< 1% relative error)

The CUDA kernels (`cuda_corr`, `cuda_ba`, `lietorch_backends`) have been modified to support FP16 operations with proper numerical stability.

## Project Structure

```
Deep-Visual-Odometry-SLAM/
├── methods/
│   └── dpvo/                    # Main DPVO implementation
│       ├── dpvo/                # Core modules
│       │   ├── altcorr/         # CUDA correlation kernels (FP16 supported)
│       │   ├── fastba/          # CUDA bundle adjustment
│       │   ├── lietorch/        # Lie group operations
│       │   └── data_readers/    # Dataset loaders
│       ├── config/              # Training configurations
│       ├── train.py             # Training script
│       └── INSTALL.md           # Installation guide
├── modules/
│   ├── eigen-3.4.0/             # Eigen3 library
│   ├── DBoW2/                   # Loop closure vocabulary
│   ├── DPRetrieval/             # Place recognition
│   ├── DPViewer/                # 3D visualization
│   └── Pangolin/                # OpenGL viewer
├── datasets/                    # Dataset storage
└── checkpoints/                 # Model checkpoints
```

## Supported GPU Architectures

| GPU Series | Architecture | Compute Capability | NVCC Flag |
|------------|--------------|-------------------|-----------|
| RTX 50xx | Blackwell | 12.0 | sm_120 |
| RTX 40xx | Ada Lovelace | 8.9 | sm_89 |
| RTX 30xx | Ampere | 8.6 | sm_86 |
| RTX 20xx | Turing | 7.5 | sm_75 |

## Acknowledgements

This project is based on:
- [DPVO](https://github.com/princeton-vl/DPVO) - Deep Patch Visual Odometry
- [TartanAir](https://theairlab.org/tartanair-dataset/) - Challenging Visual SLAM Dataset
- [lietorch](https://github.com/princeton-vl/lietorch) - Lie Groups for PyTorch

## License

MIT License
