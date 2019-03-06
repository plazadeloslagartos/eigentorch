"""
This module implements SPDNet, a neural network which makes use of Riemannian Geometry for processing of features
formed by symmetric-positive definite matrices.

This is described fully in:

Huang, Z., & Van Gool, L. (2016). A Riemannian Network for SPD Matrix Learning, 2036–2042.
https://doi.org/10.1109/CVPR.2014.132
"""
import torch
from torch import nn
from collections import OrderedDict
import eigenfunctions as eF


class StiefelParameter(nn.Parameter):
    def __init__(self, *args, **kwargs):
        super(StiefelParameter, self).__init__()


class SPDNet(nn.Module):
    def __init__(self, dim_in, dim_out, num_filters=1, eig_thresh=1e-4, log_euclid=True):
        """
        SPDnet Layer
        :param dim_in: (int) rank of input SPD feature
        :param dim_out: (int) rank of output SPD features (must be less than dim_in)
        :param num_filters: (int) number of SPD filters (features to return)
        :param eig_thresh: (float) minimum eigenvalue threshold for ReEig, default: 1e-4
        :param log_euclid: (bool) whether or not to perform LogEig operation, default: True
        """
        super(SPDNet, self).__init__()
        assert dim_out < dim_in
        self.weights_list = []
        self.eig_thresh = 1e-4
        self.log_euclid = log_euclid
        for idx in range(num_filters):
            W_dat = torch.rand(dim_in, dim_in)
            W_dat = W_dat.t().mm(W_dat)
            junk, W_init = torch.eig(W_dat, eigenvectors=True)
            m_name = "W{:d}".format(idx)
            self.weights_list.append(StiefelParameter(W_init[:dim_out]))
            setattr(self, m_name, self.weights_list[-1])

    def forward(self, X):
        """
        Forward pass of spdnet
        :param X: Tensor [n, n] Symeetric positive definite matrix
        :return: Tensor [b, nf, n , n]:
                            b is batch size
                            nf is number of filters
                            n is rank of input X
        """
        batch_output = []
        if len(X.shape) == 2:
            X = torch.stack([X])
        for feat in X:
            feat_output = []
            for W in self.weights_list:
                X_spd = eF.BiMap.apply(feat, W)
                X_spd = eF.ReEig(X_spd, self.eig_thresh)
                if self.log_euclid:
                    feat_output.append(eF.LogEig(X_spd))
            batch_output.append(torch.stack(feat_output))
        return torch.stack(batch_output)
