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
from torch.optim.optimizer import Optimizer, required


class BiMap(torch.autograd.Function):
    """
    Defines Bilinear Layer whose weights are updated with an orthogonality constraint. Can be used to generate
    CNN-like SPD filters which convert an input SPD matrix into another learned SPD matrix
    """
    @staticmethod
    def forward(ctx, X, W):
        out = torch.mm(torch.mm(W, X), W.t())
        ctx.save_for_backward(X, W)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes input gradients for BiMap Bilinear transformation.  The gradient for the transformations weights
        represent the Riemannian gradient with respect to the tangent of the Stiefel manifold
        NOTE: Results are only valid if inputs to forward are symmetric matrices!

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
        """
        X, W = ctx.saved_tensors
        grad_X = torch.mm(torch.mm(W.t(), grad_output), W)
        grad_W = 2 * grad_output.mm(W.mm(X))
        grad_W_r = grad_W - grad_W.mm(W.t().mm(W))
        return grad_X, grad_W_r


class StiefelOpt(Optimizer):
    """
    Implements Parameter optimization with respect to a gradient on the Steifel manifold.
    Expects that the gradient associated with a given Parameter is the Riemannian gradient
    tangent to the manifold (i.e. gradient generated from BiMap function)
    """
    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(StiefelOpt, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(StiefelOpt, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('placeholder', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Perform Retraction onto Steifel manifold
                d_p = p.grad.data
                p.data = torch.qr(p.data - group['lr']*d_p)[0]

        return loss


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


