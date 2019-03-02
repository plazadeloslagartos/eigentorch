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


class SPDNet(nn.Module):
    def __init__(self, dim_in, dim_out, num_filters=1):
        super(SPDNet, self).__init__()
        assert dim_out <= dim_in
        weight_dict = OrderedDict()
        for idx in range(num_filters):
            W_dat = torch.rand(dim_in, dim_in)
            W_dat = W_dat.t().mm(W_dat)
            junk, W_init = torch.eig(W_dat, eigenvectors=True)
            m_name = "W{:d}".format(idx)
            weight_dict.update({m_name: nn.Parameter(W_init[:dim_out])})
        self.weights = nn.Sequential(weight_dict)
