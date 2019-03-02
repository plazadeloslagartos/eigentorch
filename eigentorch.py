"""
This module implements custom PyTorch Autograd functions which can be used to facilitate weight updating for
Semi-Positive Definite Matrices on a Steifel Manifold via Riemannian Geometry, as opposed to Euclidean Space.
The functions are then used to define an SPDNet layer.

This is described fully in:

Huang, Z., & Van Gool, L. (2016). A Riemannian Network for SPD Matrix Learning, 2036–2042.
https://doi.org/10.1109/CVPR.2014.132

Ionescu, C.; Vantzos, O.; and Sminchisescu, C. 2015. Matrix back- propagation for deep networks with structured layers.
In ICCV. Jarrett,
https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.pdf
"""
import torch
nn = torch.nn
from torch.optim.optimizer import Optimizer, required


def sym_op(X):
    return 0.5 * (X + X.t())


def diag_op(X):
    X[torch.eye(X.shape[0]) == 0] = 0
    return X


class BiMap(torch.autograd.Function):
    """
    Defines Bilinear Layer whose weights are updated with an orthogonality constraint. Can be used to generate
    CNN-like SPD filters which convert an input SPD matrix into another learned SPD matrix
    """
    @staticmethod
    def forward(ctx, X, W):
        """
        Performs forward pass of Bilinear transformation of an SPD matrix with dimension reduction Weights
        :param ctx: context object
        :param X: Tensor, input SPD matrix (d x d)
        :param W: Tensor, Orthogonal Weights (orthogonality is necessary for Steifel manifold gradient search) (m x d, m < d)
        :return: Tensor
        """
        out = torch.mm(torch.mm(W, X), W.t())
        ctx.save_for_backward(X, W)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes input gradients for BiMap Bilinear transformation.  The gradient for the transformations weights
        represent the Riemannian gradient with respect to the tangent of the Stiefel manifold
        NOTE: Results are only valid if input (X) to forward is a symmetric matrix!
        """
        X, W = ctx.saved_tensors
        grad_X = torch.mm(torch.mm(W.t(), grad_output), W)
        grad_W = 2 * grad_output.mm(W.mm(X))
        grad_W_r = grad_W - grad_W.mm(W.t().mm(W))
        return grad_X, grad_W_r


class EigDecomp(torch.autograd.Function):
    """
    Implements the ReEig non-linearity which regularizes SPD matrices but placing a lower bound on eigen-values.
    """

    @staticmethod
    def forward(ctx, X):
        """Performs forward pass of eigen-value based thresholding
        :param ctx: context object
        :param X: Tensor, input SPD matrix (d x d)
        :param eps: minimum allowed eigen-value
        :return: Thresholded SPD matrix
        """
        S, U = torch.eig(X, eigenvectors=True)
        S = S[:, 0]
        S = S.diag()
        #S_th = torch.max(eps * torch.eye(S.shape[0]), S)
        ctx.save_for_backward(U, S)
        return S, U

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Determines gradients of Loss with respect to input tensor via the eigen-value tresholding
        :param ctx: context object
        :param grad_outputs: gradients received from next layer
        :return: gradient with respect to input, None (to account for eps)
        """
        #grad_X = grad_outputs[0]
        grad_S, grad_U = grad_outputs
        U, S = ctx.saved_tensors
        rank = S.shape[0]
        #grad_U = 2 * sym_op(grad_X).mm(U.mm(S_th))
        #Q = torch.eye(S.shape[0])
        #Q[S <= eps[0]] = 0
        #grad_S = Q.mm(U.t().mm(sym_op(grad_X).mm(U)))
        s_vals = S.diag()
        P = 1/(s_vals.view(-1, 1).repeat((1, rank)) - s_vals.repeat(rank, 1))
        P[torch.eye(rank) == 1] = 0

        dU = 2 * U.mm((P.t() * sym_op(U.t().mm(grad_U))).mm(U.t()))
        dS = U.mm(diag_op(grad_S).mm(U.t()))

        return dU + dS


class EigenLog(torch.autograd.Function):
    """
    Implements the LogEig non-linearity which induces a Log-Euclidean Riemannian metric on an input SPD matrix.
    """
    @staticmethod
    def forward(ctx, S, U):
        """
        Implementes the forward pass of the LogEig non-linearity which allows for Euclidean geometry to be used for
        separation of features in the SPD matrix space.
        :param ctx: context object
        :param X: input SPD matrix (d x d)
        :return: log-euclidean spd matrix
        """
        S_log = torch.log(S.diag()).diag()
        ctx.save_for_backward(S, S_log, U)
        return U.mm(S_log.mm(U.t()))

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Determines gradients of Loss with respect to input tensor via the Log-Euclidean transformation
        :param ctx: context object
        :param grad_outputs: gradients received from next layer
        :return: gradient with respect to input
        """
        grad_X = grad_outputs[0]
        S, S_log, U = ctx.saved_tensors
        grad_S = S.inverse().mm(U.t().mm(sym_op(grad_X).mm(U)))
        grad_U = 2 * sym_op(grad_X).mm(U.mm(S_log))

        return grad_S, grad_U


def ReEig(X, eps):
    S, U = EigDecomp.apply(X)
    S_th = torch.max(eps * torch.eye(S.shape[0]), S)
    return U.mm(S_th.mm(U.t()))


def LogEig(X):
    S, U = EigDecomp.apply(X)
    out = EigenLog.apply(S, U)
    return out


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.optim import SGD
    func1 = BiMap.apply
    func2 = ReEig
    func3 = LogEig
    func4 = lambda x, w: torch.mm(torch.mm(w, x), w.t())
    Xdat = torch.rand(5, 5)
    Xdat = Xdat @ Xdat.t()
    Wdat = torch.rand(5, 5)
    Wdat = Wdat @ Wdat.t()
    expect = torch.rand(3, 3)
    expect = expect @ expect.t()
    e, v = torch.eig(Wdat, eigenvectors=True)
    Wdat = v[:3]

    X2 = Xdat.clone().detach().requires_grad_(True)
    W2 = Wdat.clone().detach().requires_grad_(True)
    # output2 = torch.mm(torch.mm(W2, X2), W2.t())
    # loss2 = (torch.norm(output2 - torch.ones_like(output2)))
    # loss2.backward()
    lvec = []
    lvec2 = []
    X = Xdat.clone().detach().requires_grad_(True)
    W = Wdat.clone().detach().requires_grad_(True)
    #output = myfunc(X)
    #loss = (output.norm() - W.norm())**2
    #loss.backward()

    optimizer = StiefelOpt([W], lr=0.001)
    optimizer2 = SGD([W2], lr=0.001)
    for idx in range(1000):
        output = func1(X, W)
        output = func2(output, 1e-4)
        output = func3(output)
        output2 = func4(X2, W2)
        #e = torch.eig(output)[0]
        #print("min eval: {:.4f}".format(e[:, 0].min()))
        #output = func3(output)
        loss = (torch.mean(output - expect))
        loss2 = (torch.norm(output2 - expect))
        lvec.append(loss.item())
        lvec2.append(loss2.item())
        loss.backward()
        loss2.backward()
        optimizer.step()
        optimizer2.step()
    pass


