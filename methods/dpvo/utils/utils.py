import cv2
import torch

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