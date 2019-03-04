from torch.utils import data as data
from sklearn.model_selection import StratifiedShuffleSplit
from time import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from data_utilities import AggregateDataSet
from spdnn import SPDNet, StiefelParameter
from eigenoptim import StiefelOpt
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
        output = self.spd(input)
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
    for train_idxs, test_idxs in cv.split(full_indices, labels):
        cv_idx += 1
        # Get train/test sets
        train_set = data.SubsetRandomSampler(train_idxs)
        test_set = data.SubsetRandomSampler(test_idxs)

        dl_train = data.DataLoader(metaset, batch_size=batch_s, sampler=train_set, drop_last=True)
        dl_test = data.DataLoader(metaset, batch_size=batch_s, sampler=test_set, drop_last=True)

        # Setup model objects
        model = P300SpdModel()
        rm_params = []
        eu_params = []
        for param in model.parameters():
            if isinstance(param, StiefelParameter):
                rm_params.append(param)
            else:
                eu_params.append(param)
        optimizer_rm = StiefelOpt(rm_params, lr=0.001)
        optimizer_eu = SGD(eu_params, lr=0.001)
        #loss_function = nn.NLLLoss(weight=torch.Tensor([.2, .8]))
        loss_function = nn.CrossEntropyLoss()

        # loop through batches and train
        loss_arr = []
        agg_loss_arr = []

        num_epochs = 1
        a = time()
        agg_loss_all = []
        for epoch in range(num_epochs):
            agg_loss = 0
            p300s = []
            for idx, ba in enumerate(dl_train):
                p_count = (ba['label'] == 1.0).sum()
                np_count = (ba['label'] == 2.0).sum()
                if idx % 20 == 0:
                    log.info('Epoch: {:d}, Batch: {:d}, P300: {:d}, NP300 {:d}'.format(epoch, idx, p_count, np_count))
                optimizer_rm.zero_grad()
                optimizer_eu.zero_grad()
                scores = model(ba['features'])
                target = ba['label'].long() - 1
                loss = loss_function(scores, target)
                loss.backward()
                optimizer_rm.step()
                optimizer_eu.step()
                loss_val = loss.item()
                loss_arr.append(loss_val)
                agg_loss += loss_val
            agg_loss_arr.append(agg_loss)
        elapsed = time() - a
        log.info("Elapsed Time: {:.2f}".format(elapsed))

        predicted = []
        labels = []
        scores_arr = []
        with torch.no_grad():
            for idx, ba in enumerate(dl_test):
                scores = model(ba['features'])
                scores_arr.extend(scores)
                predicted.extend(torch.argmax(scores, 1).numpy())
                labels.extend(ba['label'].long().numpy())
            predicted = np.hstack(predicted)
            labels = np.hstack(labels) - 1
            scores_arr = np.vstack(scores_arr)

        overall = (predicted == labels).sum() / len(labels)
        p300 = ((predicted == 0) & (labels == 0)).sum() / (labels == 0).sum()
        np300 = ((predicted == 1) & (labels == 1)).sum() / (labels == 1).sum()
        results.append([overall, p300, np300])
        agg_loss_all.append(agg_loss_arr)
    plt.figure()
    plt.plot(np.array(agg_loss_all).T)
    plt.title('Aggregate Loss at each epoch}')

    print("scores \n{:s}".format(np.array(results).__repr__()))
    print("avg \n{:s}".format(np.array(results).mean(axis=0).__repr__()))
    print("std \n{:s}".format(np.array(results).std(axis=0).__repr__()))