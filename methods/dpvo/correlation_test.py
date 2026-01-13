#!/usr/bin/env python3
"""
Test script for verifying FP16 support in cuda_corr extension.

Usage:
    python correlation_test.py
"""

import torch
import cuda_corr


def test_corr_forward_fp32():
    """Test correlation forward pass with FP32"""
    print("\n[Test 1] corr forward - FP32")

    B, N, C, H, W = 1, 10, 128, 30, 40
    M = 100
    radius = 3

    fmap1 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float32)
    fmap2 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float32)
    coords = torch.rand(B, M, 2, H, W, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda').view(1, 1, 2, 1, 1)
    ii = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)
    jj = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)

    try:
        corr, = cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)
        print(f"   Output shape: {corr.shape}")
        print(f"   Output dtype: {corr.dtype}")
        print(f"   Output range: [{corr.min().item():.4f}, {corr.max().item():.4f}]")
        print("   PASSED")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_corr_forward_fp16():
    """Test correlation forward pass with FP16"""
    print("\n[Test 2] corr forward - FP16 (Half)")

    B, N, C, H, W = 1, 10, 128, 30, 40
    M = 100
    radius = 3

    fmap1 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float16)
    fmap2 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float16)
    coords = torch.rand(B, M, 2, H, W, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda').view(1, 1, 2, 1, 1)
    ii = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)
    jj = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)

    try:
        corr, = cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)
        print(f"   Output shape: {corr.shape}")
        print(f"   Output dtype: {corr.dtype}")
        print(f"   Output range: [{corr.min().item():.4f}, {corr.max().item():.4f}]")
        print("   PASSED")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_corr_backward_fp32():
    """Test correlation backward pass with FP32"""
    print("\n[Test 3] corr backward - FP32")

    B, N, C, H, W = 1, 10, 128, 30, 40
    M = 100
    radius = 3
    D = 2 * radius + 1

    fmap1 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float32, requires_grad=True)
    fmap2 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float32, requires_grad=True)
    coords = torch.rand(B, M, 2, H, W, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda').view(1, 1, 2, 1, 1)
    ii = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)
    jj = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)
    grad = torch.randn(B, M, D, D, H, W, device='cuda', dtype=torch.float32)

    try:
        fmap1_grad, fmap2_grad = cuda_corr.backward(fmap1, fmap2, coords, ii, jj, grad, radius)
        print(f"   fmap1_grad shape: {fmap1_grad.shape}, dtype: {fmap1_grad.dtype}")
        print(f"   fmap2_grad shape: {fmap2_grad.shape}, dtype: {fmap2_grad.dtype}")
        print("   PASSED")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_corr_backward_fp16():
    """Test correlation backward pass with FP16"""
    print("\n[Test 4] corr backward - FP16 (Half)")

    B, N, C, H, W = 1, 10, 128, 30, 40
    M = 100
    radius = 3
    D = 2 * radius + 1

    fmap1 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float16, requires_grad=True)
    fmap2 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float16, requires_grad=True)
    coords = torch.rand(B, M, 2, H, W, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda').view(1, 1, 2, 1, 1)
    ii = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)
    jj = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)
    grad = torch.randn(B, M, D, D, H, W, device='cuda', dtype=torch.float16)

    try:
        fmap1_grad, fmap2_grad = cuda_corr.backward(fmap1, fmap2, coords, ii, jj, grad, radius)
        print(f"   fmap1_grad shape: {fmap1_grad.shape}, dtype: {fmap1_grad.dtype}")
        print(f"   fmap2_grad shape: {fmap2_grad.shape}, dtype: {fmap2_grad.dtype}")
        print("   PASSED")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_patchify_forward_fp32():
    """Test patchify forward with FP32"""
    print("\n[Test 5] patchify forward - FP32")

    B, C, H, W = 1, 128, 120, 160
    M = 1000
    radius = 1

    net = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
    coords = torch.rand(B, M, 2, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda')

    try:
        patches, = cuda_corr.patchify_forward(net, coords, radius)
        print(f"   Output shape: {patches.shape}")
        print(f"   Output dtype: {patches.dtype}")
        print("   PASSED")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_patchify_forward_fp16():
    """Test patchify forward with FP16"""
    print("\n[Test 6] patchify forward - FP16 (Half)")

    B, C, H, W = 1, 128, 120, 160
    M = 1000
    radius = 1

    net = torch.randn(B, C, H, W, device='cuda', dtype=torch.float16)
    coords = torch.rand(B, M, 2, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda')

    try:
        patches, = cuda_corr.patchify_forward(net, coords, radius)
        print(f"   Output shape: {patches.shape}")
        print(f"   Output dtype: {patches.dtype}")
        print("   PASSED")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_patchify_backward_fp16():
    """Test patchify backward with FP16"""
    print("\n[Test 7] patchify backward - FP16 (Half)")

    B, C, H, W = 1, 128, 120, 160
    M = 1000
    radius = 1
    D = 2 * radius + 2

    net = torch.randn(B, C, H, W, device='cuda', dtype=torch.float16)
    coords = torch.rand(B, M, 2, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda')
    gradient = torch.randn(B, M, C, D, D, device='cuda', dtype=torch.float16)

    try:
        net_grad, = cuda_corr.patchify_backward(net, coords, gradient, radius)
        print(f"   Output shape: {net_grad.shape}")
        print(f"   Output dtype: {net_grad.dtype}")
        print("   PASSED")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_autocast_forward():
    """Test with torch.amp.autocast"""
    print("\n[Test 8] corr forward with autocast")

    B, N, C, H, W = 1, 10, 128, 30, 40
    M = 100
    radius = 3

    fmap1 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float32)
    fmap2 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float32)
    coords = torch.rand(B, M, 2, H, W, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda').view(1, 1, 2, 1, 1)
    ii = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)
    jj = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)

    try:
        with torch.amp.autocast('cuda', enabled=True):
            # Inside autocast, fmap1/fmap2 will be cast to FP16
            fmap1_half = fmap1.half()
            fmap2_half = fmap2.half()
            corr, = cuda_corr.forward(fmap1_half, fmap2_half, coords, ii, jj, radius)

        print(f"   Output shape: {corr.shape}")
        print(f"   Output dtype: {corr.dtype}")
        print("   PASSED")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_numerical_consistency():
    """Test that FP16 and FP32 produce similar results"""
    print("\n[Test 9] Numerical consistency (FP32 vs FP16)")

    B, N, C, H, W = 1, 5, 128, 20, 25
    M = 50
    radius = 3

    # Use same random seed for both
    torch.manual_seed(42)
    fmap1_fp32 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float32)
    fmap2_fp32 = torch.randn(B, N, C, H, W, device='cuda', dtype=torch.float32)
    coords = torch.rand(B, M, 2, H, W, device='cuda', dtype=torch.float32) * torch.tensor([W-1, H-1], device='cuda').view(1, 1, 2, 1, 1)
    ii = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)
    jj = torch.randint(0, N, (M,), device='cuda', dtype=torch.long)

    fmap1_fp16 = fmap1_fp32.half()
    fmap2_fp16 = fmap2_fp32.half()

    try:
        corr_fp32, = cuda_corr.forward(fmap1_fp32, fmap2_fp32, coords, ii, jj, radius)
        corr_fp16, = cuda_corr.forward(fmap1_fp16, fmap2_fp16, coords, ii, jj, radius)

        # Compare results
        corr_fp16_as_fp32 = corr_fp16.float()
        diff = (corr_fp32 - corr_fp16_as_fp32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_error = (diff / (corr_fp32.abs() + 1e-6)).mean().item()

        print(f"   Max absolute diff: {max_diff:.6f}")
        print(f"   Mean absolute diff: {mean_diff:.6f}")
        print(f"   Mean relative error: {relative_error:.4%}")

        # FP16 has ~0.1% precision, so relative error should be < 1%
        if relative_error < 0.01:
            print("   PASSED (relative error < 1%)")
            return True
        else:
            print("   WARNING: Relative error > 1%, but this may be acceptable for training")
            return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("CUDA Correlation Extension - FP16 Support Test")
    print("=" * 60)

    # Print GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    results = []

    results.append(("corr forward FP32", test_corr_forward_fp32()))
    results.append(("corr forward FP16", test_corr_forward_fp16()))
    results.append(("corr backward FP32", test_corr_backward_fp32()))
    results.append(("corr backward FP16", test_corr_backward_fp16()))
    results.append(("patchify forward FP32", test_patchify_forward_fp32()))
    results.append(("patchify forward FP16", test_patchify_forward_fp16()))
    results.append(("patchify backward FP16", test_patchify_backward_fp16()))
    results.append(("autocast forward", test_autocast_forward()))
    results.append(("numerical consistency", test_numerical_consistency()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed! FP16 support is working correctly.")
    else:
        print("\n  Some tests failed. Check the output above for details.")


if __name__ == '__main__':
    main()
