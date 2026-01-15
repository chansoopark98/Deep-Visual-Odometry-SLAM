import cv2
import torch
import numpy as np

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


@torch.amp.autocast('cuda', enabled=False)
def kabsch_umeyama(A, B):
    """
    Compute scale factor for trajectory alignment (used in training loss).

    Args:
        A: GT trajectory (N, 3)
        B: Predicted trajectory (N, 3)

    Returns:
        c: Scale factor
    """
    # SVD requires FP32 - disable autocast and convert inputs
    A = A.float()
    B = B.float()

    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def align_trajectory_umeyama(pred_traj, gt_traj):
    """
    Full Umeyama alignment: align predicted trajectory to GT using scale, rotation, and translation.

    This computes the optimal similarity transformation (Sim3) that minimizes:
        sum || gt_i - (s * R @ pred_i + t) ||^2

    Args:
        pred_traj: Predicted trajectory (N, 3) numpy array
        gt_traj: Ground truth trajectory (N, 3) numpy array

    Returns:
        aligned_traj: Aligned predicted trajectory (N, 3)
        scale: Scale factor
        rotation: Rotation matrix (3, 3)
        translation: Translation vector (3,)
    """

    # Ensure numpy arrays
    pred_traj = np.array(pred_traj)
    gt_traj = np.array(gt_traj)

    n = pred_traj.shape[0]

    # Compute centroids
    mu_pred = np.mean(pred_traj, axis=0)
    mu_gt = np.mean(gt_traj, axis=0)

    # Center the trajectories
    pred_centered = pred_traj - mu_pred
    gt_centered = gt_traj - mu_gt

    # Compute variances
    var_pred = np.sum(pred_centered ** 2) / n

    # Compute cross-covariance matrix
    H = (gt_centered.T @ pred_centered) / n

    # SVD
    U, D, Vt = np.linalg.svd(H)

    # Compute rotation (handle reflection case)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    # Compute scale
    scale = np.trace(np.diag(D) @ S) / var_pred

    # Compute translation
    t = mu_gt - scale * R @ mu_pred

    # Apply transformation
    aligned_traj = scale * (pred_traj @ R.T) + t

    return aligned_traj, scale, R, t