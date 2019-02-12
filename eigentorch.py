"""
This module implements custom PyTorch Autograd functions which can be used to facilitate weight updating for
Semi-Positive Definite Matrices on a Steifel Manifold via Riemannian Geometry, as opposed to Euclidean Space.
The functions are then used to define an SPDNet layer.

This is described fully in:

Huang, Z., & Van Gool, L. (2016). A Riemannian Network for SPD Matrix Learning, 2036–2042.
https://doi.org/10.1109/CVPR.2014.132
"""
import torch
nn = torch.nn


class BiMap(torch.autograd.Function):
    """
    Defines Bilinear Layer whose weights are updated with an orthogonality constraint. Can be used to generate
    CNN-like SPD filters which convert an input SPD matrix into another learned SPD matrix
    """
    @staticmethod
    def forward(ctx, Xkm1, W):
        out = torch.mm(torch.mm(W, Xkm1), W.t())
        ctx.save_for_backward(Xkm1, W)
        print("BiMap Forward")
        return out

    @staticmethod
    def backward(ctx, grad_output):
        "test edit    "
        Xkm1, W = ctx.saved_tensors
        grad_Xk = torch.mm(torch.mm(W.t(), grad_output), W)
        grad_Wk = 2 * grad_output.mm(W.mm(Xkm1))
        print('BiMap Backward')
        return grad_Xk, grad_Wk


if __name__ == "__main__":
    myfunc = BiMap.apply
    Xdat = torch.rand(3, 3)
    Xdat = Xdat @ Xdat.t()
    Wdat = torch.rand(3, 3)
    Wdat = Wdat @ Wdat.t()

    X2 = Xdat.clone().detach().requires_grad_(True)
    W2 = Wdat.clone().detach().requires_grad_(True)
    output2 = torch.mm(torch.mm(W2, X2), W2.t())
    loss2 = (torch.norm(output2 - torch.ones_like(output2)))
    loss2.backward()

    X = Xdat.clone().detach().requires_grad_(True)
    W = Wdat.clone().detach().requires_grad_(True)
    output = myfunc(X, W)
    loss = (torch.norm(output - torch.ones_like(output)))
    loss.backward()


