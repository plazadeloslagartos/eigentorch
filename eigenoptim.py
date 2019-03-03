"""
This module implements a custom optimizer for optimizing parameters of size (m x n) in Euclidean space, when the
received from the next layer conform to the geometry of a Steifel manifold.  This ensures that the weights obey
the constrait of semi-orthogonality.


This is described fully in:

Huang, Z., & Van Gool, L. (2016). A Riemannian Network for SPD Matrix Learning, 2036–2042.
https://doi.org/10.1109/CVPR.2014.132
"""

from torch.optim.optimizer import Optimizer, required


class StiefelOpt(Optimizer):
    """
    Implements Parameter optimization with respect to a gradient on the Steifel manifold.
    Expects that the gradient associated with a given Parameter is the Riemannian gradient
    tangent to the manifold (i.e. gradient generated from BiMap function)

     Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
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
                p.data = torch.qr((p.data - group['lr']*d_p).t())[0].t()

        return loss
