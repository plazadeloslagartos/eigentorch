from torch.utils import data as data
from sklearn.model_selection import StratifiedShuffleSplit
from time import time
import numpy as np
import logging
import torch
from data_utilities import AggregateDataSet
from spdnn import SPDNet, StiefelParameter
from torch.optim import SGD
log = logging.getLogger(__name__)
nn = torch.nn


class P300SpdModel(nn.Module):
    def __init__(self, feature_size=6, output_size=3, num_filters=2, batch_size=1):
        super(P300SpdModel, self).__init__()
        self.batch_size = batch_size
        self.spd = SPDNet(feature_size, output_size, num_filters)
        self.fc = nn.Linear(in_features=(output_size**2) * num_filters, out_features=2)

    def forward(self, input):
        output = self.sped(input)
        output = self.fc(output)
        return torch.sigmoid(output)


if __name__ == "__main__":
    metaset = AggregateDataSet('../Neurable/subvox-exploratory/p300_epoch_data', downsample=12, num_dsets=None, num_epochs_avg=None, avg_all=False)
    num_cv = 3
    cv_split = 0.7
    num_test = int((1.0 - cv_split) * len(metaset))
    full_indices = np.arange(len(metaset))
    split_idx = np.int(len(metaset) * cv_split)
    results = []
    dl_tmp = iter(data.DataLoader(metaset, batch_size=len(metaset)))
    labels = dl_tmp.next()['label']
    cv = StratifiedShuffleSplit(n_splits=num_cv, test_size=num_test)

    batch_s = 1
    cv_idx = 0
    num_spd_filters = 2
    # for train_idxs, test_idxs in cv.split(full_indices, labels):
    #     cv_idx += 1
    #     # Get train/test sets
    #     train_set = data.SubsetRandomSampler(train_idxs)
    #     test_set = data.SubsetRandomSampler(test_idxs)
    #
    #     dl_train = data.DataLoader(metaset, batch_size=batch_s, sampler=train_set, drop_last=True)
    #     dl_test = data.DataLoader(metaset, batch_size=batch_s, sampler=test_set, drop_last=True)
    #
    #     # Setup model objects
    #     model = P300SpdModel()
    #     optimizer = SGD(model.parameters(), lr=0.01, momentum=0.5)
    #     #loss_function = nn.NLLLoss(weight=torch.Tensor([.2, .8]))
    #     loss_function = nn.CrossEntropyLoss()
