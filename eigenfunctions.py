"""
This module implements custom PyTorch Autograd functions which can be used to facilitate weight updating for
Semi-Positive Definite Matrices on a Stiefel Manifold via Riemannian Geometry, as opposed to Euclidean Space.
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
    X = EigenLog.apply(S, U)
    return X
