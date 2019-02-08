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
    def forward(ctx, Xkm1, W, Wt):
        ctx.save_for_backward(Wt)  # Does Wt really need to be saved or is it calculated out?
        out = torch.mm(torch.mm(W, Xkm1), torch.transpose(W, 0, 1))
        ctx.save_for_backward(out)
        return torch.mm(torch.mm(W, Xkm1), torch.transpose(W, 0, 1))

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass
