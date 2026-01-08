import torch
import torch.nn as nn
import torch_scatter
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

# DPVO imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dpvo.blocks import SoftAgg


# ============================================
# Configuration - DPVO actual tensor shapes
# ============================================
@dataclass
class DPVOConfig:
    """DPVO에서 사용하는 실제 텐서 크기"""
    DIM: int = 384                    # Feature dimension
    BATCH_SIZE: int = 1               # Batch size (always 1 in DPVO)
    PATCHES_PER_FRAME: int = 80       # M: patches per frame (inference)
    PATCHES_PER_FRAME_TRAIN: int = 1024  # M: patches per frame (training)
    NUM_FRAMES: int = 8               # Initial frames
    NUM_FRAMES_MAX: int = 15          # Max frames in training

    @property
    def num_edges_inference(self) -> int:
        """Approximate number of edges during inference"""
        return self.NUM_FRAMES * self.PATCHES_PER_FRAME * 2  # ~1280

    @property
    def num_edges_training(self) -> int:
        """Approximate number of edges during training"""
        return self.NUM_FRAMES * self.PATCHES_PER_FRAME_TRAIN  # ~8192


# ============================================
# Accurate CUDA Timing Utilities
# ============================================
class CUDATimer:
    """정확한 CUDA 타이밍 측정을 위한 클래스"""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    @contextmanager
    def measure(self):
        """Context manager for timing"""
        torch.cuda.synchronize()
        self.start_event.record()
        yield
        self.end_event.record()
        torch.cuda.synchronize()

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds"""
        return self.start_event.elapsed_time(self.end_event)


def benchmark_function(func, args, warmup: int = 50, iterations: int = 200) -> Dict[str, float]:
    """
    정확한 벤치마크 수행

    Args:
        func: 벤치마크할 함수
        args: 함수 인자 (tuple)
        warmup: 워밍업 반복 횟수
        iterations: 측정 반복 횟수

    Returns:
        dict with mean, std, min, max times in ms
    """
    timer = CUDATimer()

    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(iterations):
        with timer.measure():
            _ = func(*args)
        times.append(timer.elapsed_ms())

    times = np.array(times)
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'median': float(np.median(times)),
    }


# ============================================
# FLOPs Calculation
# ============================================
def calculate_scatter_sum_flops(src_shape: Tuple[int, ...], num_groups: int) -> int:
    """
    scatter_sum의 FLOPs 계산

    scatter_sum은 기본적으로 addition 연산
    각 원소당 1 FLOP (addition)
    """
    total_elements = np.prod(src_shape)
    return int(total_elements)  # 각 원소를 한 번씩 더함


def calculate_scatter_softmax_flops(src_shape: Tuple[int, ...], num_groups: int) -> int:
    """
    scatter_softmax의 FLOPs 계산

    Steps:
    1. scatter_max: N comparisons
    2. subtraction: N ops
    3. exp: N ops (약 20 FLOPs per exp)
    4. scatter_sum: N additions
    5. division: N ops

    총: ~23N FLOPs
    """
    N = np.prod(src_shape)
    exp_flops = 20  # exp 연산의 대략적 FLOPs
    return int(N * (1 + 1 + exp_flops + 1 + 1))


def calculate_softagg_flops(batch_size: int, num_elements: int, dim: int, num_groups: int) -> int:
    """
    SoftAgg 모듈의 FLOPs 계산

    Components:
    1. Linear f: 2 * num_elements * dim * dim (matmul + bias)
    2. Linear g: 2 * num_elements * dim * dim
    3. Linear h: 2 * num_groups * dim * dim
    4. scatter_softmax: see above
    5. element-wise multiply: num_elements * dim
    6. scatter_sum: num_elements * dim
    7. gather (indexing): negligible
    """
    linear_flops = 2 * num_elements * dim * dim  # per linear layer
    scatter_softmax_flops = calculate_scatter_softmax_flops((batch_size, num_elements, dim), num_groups)
    multiply_flops = num_elements * dim
    scatter_sum_flops = calculate_scatter_sum_flops((batch_size, num_elements, dim), num_groups)

    total = 3 * linear_flops + scatter_softmax_flops + multiply_flops + scatter_sum_flops
    return int(total)


# ============================================
# Native PyTorch scatter functions
# ============================================
def scatter_sum_native(src: torch.Tensor, index: torch.Tensor, dim: int = 1,
                       dim_size: Optional[int] = None) -> torch.Tensor:
    """
    torch_scatter.scatter_sum의 순수 PyTorch 대체

    Args:
        src: source tensor (B, N, D) or (B, N)
        index: index tensor - will be expanded to match src shape
        dim: dimension to scatter along
        dim_size: size of output dimension (optional)
    """
    # Expand index to match src dimensions if needed
    if index.dim() < src.dim():
        expand_shape = [1] * src.dim()
        expand_shape[dim] = -1
        index = index.view(*expand_shape)
        index = index.expand_as(src)

    if dim_size is None:
        dim_size = index.max().item() + 1

    # Create output tensor
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)

    # Use scatter_add_
    return out.scatter_add_(dim, index, src)


def scatter_softmax_native(src: torch.Tensor, index: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    torch_scatter.scatter_softmax의 순수 PyTorch 대체

    Numerically stable implementation using max subtraction
    """
    # Expand index to match src dimensions if needed
    if index.dim() < src.dim():
        expand_shape = [1] * src.dim()
        expand_shape[dim] = -1
        index = index.view(*expand_shape)
        index = index.expand_as(src)

    # Get number of groups
    num_groups = index.max().item() + 1

    # Step 1: Find max per group for numerical stability
    shape = list(src.shape)
    shape[dim] = num_groups

    max_vals = torch.full(shape, float('-inf'), dtype=src.dtype, device=src.device)
    max_vals.scatter_reduce_(dim, index, src, reduce='amax', include_self=False)

    # Step 2: Subtract max
    src_shifted = src - max_vals.gather(dim, index)

    # Step 3: Exp
    exp_src = torch.exp(src_shifted)

    # Step 4: Sum per group
    sum_exp = torch.zeros(shape, dtype=src.dtype, device=src.device)
    sum_exp.scatter_add_(dim, index, exp_src)

    # Step 5: Normalize
    return exp_src / sum_exp.gather(dim, index)


# ============================================
# SoftAgg without torch_scatter
# ============================================
class SoftAggNative(nn.Module):
    """
    SoftAgg implementation without torch_scatter dependency
    Uses native PyTorch scatter operations (requires PyTorch 2.0+)
    """
    def __init__(self, dim: int = 512, expand: bool = True):
        super(SoftAggNative, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, self.dim)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
        _, jx = torch.unique(ix, return_inverse=True)

        w = scatter_softmax_native(self.g(x), jx, dim=1)
        y = scatter_sum_native(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:, jx]
        return self.h(y)


# ============================================
# Comparison and Reporting
# ============================================
def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, name: str) -> Dict[str, Any]:
    """두 텐서의 상세 비교"""
    diff = (t1 - t2).abs()
    return {
        'name': name,
        'shape': tuple(t1.shape),
        'dtype': str(t1.dtype),
        'max_diff': diff.max().item(),
        'mean_diff': diff.mean().item(),
        'std_diff': diff.std().item(),
        'allclose_1e-5': torch.allclose(t1, t2, atol=1e-5),
        'allclose_1e-6': torch.allclose(t1, t2, atol=1e-6),
        't1_stats': {
            'min': t1.min().item(),
            'max': t1.max().item(),
            'mean': t1.mean().item(),
        },
        't2_stats': {
            'min': t2.min().item(),
            'max': t2.max().item(),
            'mean': t2.mean().item(),
        }
    }


def print_comparison(comp: Dict[str, Any]):
    """비교 결과 출력"""
    print(f"\n  [{comp['name']}]")
    print(f"    Shape: {comp['shape']}, dtype: {comp['dtype']}")
    print(f"    Max diff: {comp['max_diff']:.2e}, Mean diff: {comp['mean_diff']:.2e}")
    print(f"    torch_scatter - min: {comp['t1_stats']['min']:.4f}, max: {comp['t1_stats']['max']:.4f}, mean: {comp['t1_stats']['mean']:.4f}")
    print(f"    Native        - min: {comp['t2_stats']['min']:.4f}, max: {comp['t2_stats']['max']:.4f}, mean: {comp['t2_stats']['mean']:.4f}")
    print(f"    Match (atol=1e-5): {'✓' if comp['allclose_1e-5'] else '✗'}, (atol=1e-6): {'✓' if comp['allclose_1e-6'] else '✗'}")


def print_benchmark(name: str, ts_result: Dict, native_result: Dict, flops: int):
    """벤치마크 결과 출력"""
    speedup = ts_result['mean'] / native_result['mean']
    ts_gflops = flops / (ts_result['mean'] * 1e6)  # GFLOPS
    native_gflops = flops / (native_result['mean'] * 1e6)

    print(f"\n  [{name}]")
    print(f"    torch_scatter: {ts_result['mean']:.4f} ± {ts_result['std']:.4f} ms (min: {ts_result['min']:.4f}, max: {ts_result['max']:.4f})")
    print(f"    Native:        {native_result['mean']:.4f} ± {native_result['std']:.4f} ms (min: {native_result['min']:.4f}, max: {native_result['max']:.4f})")
    print(f"    Speedup: {speedup:.2f}x {'(Native faster)' if speedup > 1 else '(torch_scatter faster)'}")
    print(f"    FLOPs: {flops:,}")
    print(f"    Throughput: torch_scatter={ts_gflops:.2f} GFLOPS, Native={native_gflops:.2f} GFLOPS")


# ============================================
# Main Benchmark
# ============================================
def run_benchmark():
    """전체 벤치마크 실행"""
    print("=" * 70)
    print(" torch_scatter vs Native PyTorch Comparison Benchmark")
    print(" DPVO Tensor Shapes")
    print("=" * 70)

    # Configuration
    cfg = DPVOConfig()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f"\n[System Info]")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

    # ============================================
    # Test 1: scatter_sum (Training scenario)
    # ============================================
    print("\n" + "=" * 70)
    print(" Test 1: scatter_sum (Training scenario)")
    print("=" * 70)

    num_edges = cfg.num_edges_training
    num_groups = cfg.NUM_FRAMES_MAX
    print(f"\n[Configuration]")
    print(f"  Batch size: {cfg.BATCH_SIZE}")
    print(f"  Num edges: {num_edges}")
    print(f"  Feature dim: {cfg.DIM}")
    print(f"  Num groups: {num_groups}")

    src_3d = torch.randn(cfg.BATCH_SIZE, num_edges, cfg.DIM, device='cuda')
    index_1d = torch.randint(0, num_groups, (num_edges,), device='cuda')
    index_3d = index_1d.unsqueeze(0).unsqueeze(-1).expand(cfg.BATCH_SIZE, -1, cfg.DIM)

    # Correctness
    result_ts = torch_scatter.scatter_sum(src_3d, index_3d, dim=1)
    result_native = scatter_sum_native(src_3d, index_1d, dim=1)

    print("\n[Correctness]")
    print_comparison(compare_tensors(result_ts, result_native, "scatter_sum"))

    # Performance
    print("\n[Performance]")
    flops = calculate_scatter_sum_flops(src_3d.shape, num_groups)
    ts_bench = benchmark_function(lambda: torch_scatter.scatter_sum(src_3d, index_3d, dim=1), ())
    native_bench = benchmark_function(lambda: scatter_sum_native(src_3d, index_1d, dim=1), ())
    print_benchmark("scatter_sum", ts_bench, native_bench, flops)

    # ============================================
    # Test 2: scatter_softmax (Training scenario)
    # ============================================
    print("\n" + "=" * 70)
    print(" Test 2: scatter_softmax (Training scenario)")
    print("=" * 70)

    src_2d = torch.randn(cfg.BATCH_SIZE, num_edges, device='cuda')
    index_2d = index_1d.unsqueeze(0)

    # Correctness
    result_ts = torch_scatter.scatter_softmax(src_2d, index_2d, dim=1)
    result_native = scatter_softmax_native(src_2d, index_1d, dim=1)

    print("\n[Correctness]")
    print_comparison(compare_tensors(result_ts, result_native, "scatter_softmax"))

    # Performance
    print("\n[Performance]")
    flops = calculate_scatter_softmax_flops(src_2d.shape, num_groups)
    ts_bench = benchmark_function(lambda: torch_scatter.scatter_softmax(src_2d, index_2d, dim=1), ())
    native_bench = benchmark_function(lambda: scatter_softmax_native(src_2d, index_1d, dim=1), ())
    print_benchmark("scatter_softmax", ts_bench, native_bench, flops)

    # ============================================
    # Test 3: SoftAgg Module (Training scenario)
    # ============================================
    print("\n" + "=" * 70)
    print(" Test 3: SoftAgg Module (Training scenario)")
    print("=" * 70)

    soft_agg_original = SoftAgg(dim=cfg.DIM).cuda().eval()
    soft_agg_native = SoftAggNative(dim=cfg.DIM).cuda().eval()
    soft_agg_native.load_state_dict(soft_agg_original.state_dict())

    src_module = torch.randn(cfg.BATCH_SIZE, num_edges, cfg.DIM, device='cuda')

    # Correctness
    with torch.no_grad():
        result_ts = soft_agg_original(src_module, index_1d)
        result_native = soft_agg_native(src_module, index_1d)

    print("\n[Correctness]")
    print_comparison(compare_tensors(result_ts, result_native, "SoftAgg"))

    # Performance
    print("\n[Performance]")
    flops = calculate_softagg_flops(cfg.BATCH_SIZE, num_edges, cfg.DIM, num_groups)
    with torch.no_grad():
        ts_bench = benchmark_function(lambda: soft_agg_original(src_module, index_1d), ())
        native_bench = benchmark_function(lambda: soft_agg_native(src_module, index_1d), ())
    print_benchmark("SoftAgg", ts_bench, native_bench, flops)

    # ============================================
    # Test 4: Inference scenario (smaller tensors)
    # ============================================
    print("\n" + "=" * 70)
    print(" Test 4: Inference scenario (smaller tensors)")
    print("=" * 70)

    num_edges_inf = cfg.num_edges_inference
    num_groups_inf = cfg.NUM_FRAMES
    print(f"\n[Configuration]")
    print(f"  Num edges: {num_edges_inf}")
    print(f"  Num groups: {num_groups_inf}")

    src_inf = torch.randn(cfg.BATCH_SIZE, num_edges_inf, cfg.DIM, device='cuda')
    index_inf = torch.randint(0, num_groups_inf, (num_edges_inf,), device='cuda')
    index_inf_3d = index_inf.unsqueeze(0).unsqueeze(-1).expand(cfg.BATCH_SIZE, -1, cfg.DIM)

    # scatter_sum inference
    result_ts = torch_scatter.scatter_sum(src_inf, index_inf_3d, dim=1)
    result_native = scatter_sum_native(src_inf, index_inf, dim=1)

    print("\n[scatter_sum - Inference]")
    print_comparison(compare_tensors(result_ts, result_native, "scatter_sum_inference"))

    flops = calculate_scatter_sum_flops(src_inf.shape, num_groups_inf)
    ts_bench = benchmark_function(lambda: torch_scatter.scatter_sum(src_inf, index_inf_3d, dim=1), ())
    native_bench = benchmark_function(lambda: scatter_sum_native(src_inf, index_inf, dim=1), ())
    print_benchmark("scatter_sum_inference", ts_bench, native_bench, flops)


if __name__ == '__main__':
    run_benchmark()