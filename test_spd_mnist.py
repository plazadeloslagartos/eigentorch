from torch.utils import data as data
from sklearn.model_selection import StratifiedShuffleSplit
from time import time
import numpy as np
import torchvision.datasets as dset
import matplotlib.pyplot as plt
from os.path import join
import logging
import torch
import torchvision.transforms as transforms
from spdnn import SPDNet, StiefelParameter
from eigenoptim import StiefelOpt
from torch.optim import SGD
log = logging.getLogger(__name__)
nn = torch.nn


class SPDNetModel(nn.Module):
    def __init__(self, spd_input_size=6, spd_output_size=3, fc_output_size=10, num_filters=2, batch_size=1):
        super(SPDNetModel, self).__init__()
        self.batch_size = batch_size
        self.spd = SPDNet(spd_input_size, spd_output_size, num_filters)
        fc_input_size = (spd_output_size**2) * num_filters
        self.bn = nn.BatchNorm1d(fc_input_size)
        self.fc = nn.Linear(in_features=fc_input_size, out_features=fc_output_size)

    def forward(self, input):
        output = self.spd(input)
        if self.batch_size > 1:
            output = self.fc(self.bn(output.view(self.batch_size, -1)))
        else:
            output = self.fc(output.view(-1))
        return torch.sigmoid(output)


if __name__ == "__main__":
    data_dir = join('.', 'MNIST')
    results = []
    num_epochs = 5
    batch_s = 10
    num_spd_filters = 10
    n_pwr = 0.05
    # if not exist, download mnist dataset
    tforms = transforms.Compose(
        (transforms.ToTensor(),
         lambda x: torch.tensor([np.cov(x[0, 4:-4, 4:-4]) + np.random.rand(20, 20) * n_pwr])))
    train_set = dset.MNIST(root=data_dir, train=True, transform=tforms, download=True)
    test_set = dset.MNIST(root=data_dir, train=False, transform=tforms, download=True)

    # Get train/test sets
    dl_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_s, shuffle=True)
    dl_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_s, shuffle=False)

    # Setup model objects
    model = SPDNetModel(batch_size=batch_s, num_filters=num_spd_filters)
    rm_params = []
    eu_params = []
    for param in model.parameters():
        if isinstance(param, StiefelParameter):
            rm_params.append(param)
        else:
            eu_params.append(param)
    optimizer_rm = StiefelOpt(rm_params, lr=0.01)
    optimizer_eu = SGD(eu_params, lr=0.01)
    #loss_function = nn.NLLLoss(weight=torch.Tensor([.2, .8]))
    loss_function = nn.CrossEntropyLoss()

    # loop through batches and train
    loss_arr = []
    agg_loss_arr = []

    a = time()
    for epoch in range(num_epochs):
        agg_loss = 0
        p300s = []
        for idx, ba in enumerate(dl_train):
            p_count = (ba['label'] == 1.0).sum()
            np_count = (ba['label'] == 2.0).sum()
            if idx % 20 == 0:
                log.info('Epoch: {:d}, Batch: {:d}, P300: {:d}, NP300 {:d}'.format(epoch, idx, p_count, np_count))
                print('CV: {:d}, Epoch: {:d}, Batch: {:d}, P300: {:d}, NP300 {:d}'.format(cv_idx, epoch, idx, p_count, np_count))
            optimizer_rm.zero_grad()
            optimizer_eu.zero_grad()
            scores = model(ba['features'])
            target = ba['label'].long() - 1
            loss = loss_function(scores.view(-1, 2), target)
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
    with torch.no_grad():
        for idx, ba in enumerate(dl_test):
            scores = model(ba['features'])
            if batch_s > 1:
                predicted.extend(torch.argmax(scores, 1).numpy())
            else:
                predicted.append(torch.argmax(scores).numpy())
            labels.extend(ba['label'].long().numpy())
        predicted = np.hstack(predicted)
        labels = np.hstack(labels) - 1

    overall = (predicted == labels).sum() / len(labels)
    #results.append([overall, p300, np300])

    plt.figure()
    plt.plot(np.array(agg_loss_arr).T)
    plt.title('Aggregate Loss at each epoch}')

    print("scores \n{:s}".format(np.array(results).__repr__()))
    print("avg \n{:s}".format(np.array(results).mean(axis=0).__repr__()))
    print("std \n{:s}".format(np.array(results).std(axis=0).__repr__()))
