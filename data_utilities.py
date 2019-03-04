import h5py as h5
import numpy as np
from sklearn.preprocessing import scale
from os import listdir
from os.path import join
from torch.utils import data
import logging

bucket = 'brains.neurable.com'


class HDF5DataSetP300Basic(data.Dataset):
    def __init__(self, hdf5_file, downsample=None):
        self.downsample = downsample
        self.hdf5_data = h5.File(hdf5_file, 'r')
        self.dataset_name = list(self.hdf5_data.keys())[0]
        self.size = len(self.hdf5_data[self.dataset_name]['meta_data'])

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        features = self.hdf5_data['metaset']['eeg_data']
        labels = self.hdf5_data['metaset']['meta_data']
        if self.downsample is not None:
            ds = self.downsample
        else:
            ds = 1
        sample = {'features': np.float32(features[item][::ds]), 'label': np.float32(labels[item][0])}
        return sample


class HDF5DataSetP300Avg(data.Dataset):
    def __init__(self, hdf5_file, scale=True, downsample=None):
        self.downsample = downsample
        self.scale = scale
        self.hdf5_data = h5.File(hdf5_file, 'r')
        self.dataset_name = list(self.hdf5_data.keys())[0]
        labels = np.array([val[2] for val in self.hdf5_data[self.dataset_name]['meta_data']])
        data = self.hdf5_data[self.dataset_name]['eeg_data']

        if (labels == 0.0).any():
            self.item0 = None, 0.0
            self.item1 = None, 0.0
        else:
            self.item0 = data[np.where(labels == 1.0)[0].tolist()], 1.0
            self.item1 = data[np.where(labels == 2.0)[0].tolist()], 2.0

    def __len__(self):
        return 2

    def __getitem__(self, item):
        if self.item0[1] == 0.0:
            return {'features': None, 'label': 0.0}
        if self.downsample is not None:
            ds = self.downsample
        else:
            ds = 1
        if self.scale:
            features0 = np.stack([scale(val, axis=0) for val in self.item0[0]])
            features1 = np.stack([scale(val, axis=0) for val in self.item1[0]])
        else:
            features0 = self.item0[0]
            features1 = self.item1[0]

        if item == 0:
            return {'features': np.cov(np.float32(np.mean(features0, axis=0)[::ds]).T), 'label': np.float32(self.item0[1])}
        elif item == 1:
            return {'features': np.cov(np.float32(np.mean(features1, axis=0)[::ds]).T), 'label': np.float32(self.item1[1])}


class HDF5DataSetP300(data.Dataset):
    def __init__(self, hdf5_file, num_seq=10, num_epochs_avg=None, num_groups=5, transform=None, scale=True, downsample=None):
        self.hdf5_data = h5.File(hdf5_file, 'r')
        self.scale = scale
        self.num_seq = 10
        self.downsample = downsample
        self.num_seq = num_seq
        self.num_epochs_avg = num_epochs_avg
        self.trial_size = num_seq * num_groups
        self.num_groups = num_groups
        self.dataset_name = list(self.hdf5_data.keys())[0]
        self.transform = transform
        self.group_ids = np.array([val[1] for val in self.hdf5_data[self.dataset_name]['meta_data']])
        self.labels = np.array([val[2] for val in self.hdf5_data[self.dataset_name]['meta_data']])

    def __len__(self):
        return len(self.labels)//self.num_seq

    def __getitem__(self, idx):
        start_idx = ((idx * self.num_seq) // self.trial_size) * self.trial_size
        end_idx = start_idx + self.trial_size
        group_id_idx = (idx * self.num_seq) % self.trial_size
        frame_idxs = range(start_idx, end_idx)
        group_id_frame = self.group_ids[frame_idxs]
        group_id_idxs_sorted = np.argsort(group_id_frame)
        item_group_id = group_id_frame[group_id_idxs_sorted[group_id_idx]]
        item_label = self.labels[start_idx + group_id_idxs_sorted[group_id_idx]]
        data_idxs = (np.where(group_id_frame == item_group_id)[0] + start_idx).tolist()
        epoch_subset = self.hdf5_data[self.dataset_name]['eeg_data'][data_idxs]
        if self.scale:
            data_epochs = np.stack([scale(chan, axis=0) for chan in epoch_subset])
        else:
            data_epochs = epoch_subset
        if self.transform is not None:
            data_epochs = self.transform(data_epochs)
        if self.num_epochs_avg is None:
            n_idx = len(data_epochs)
        else:
            n_idx = self.num_epochs_avg
        if self.downsample is not None:
            ds = self.downsample
        else:
            ds = 1
        sample = {'features': np.cov(np.float32(np.mean(data_epochs[:n_idx], axis=0)[::ds]).T), 'label': np.float32(item_label),
                  'group_id': np.float32(item_group_id)}
        return sample


class AggregateDataSet:
    def __init__(self, datasets, num_dsets=None, num_seq=10, num_epochs_avg=None,
                 avg_all=False, transform=None, scale=True, downsample=None):
        log = logging.getLogger(__name__)
        self.datasets = []
        dataset_list = [join(datasets, val) for val in listdir(datasets) if val.find('hdf5') > -1]
        if num_dsets is not None:
            dataset_list = dataset_list[:num_dsets]
        self.num_seq = num_seq
        self.num_epochs_avg = num_epochs_avg
        self.dataset_sizes = []
        for dset in dataset_list:
            log.info("loading dataset: {:s}".format(dset))
            if avg_all:
                new_set = HDF5DataSetP300Avg(dset, scale=scale, downsample=downsample)
            else:
                new_set = HDF5DataSetP300(dset, num_epochs_avg=num_epochs_avg, num_seq=num_seq,
                                          transform=transform, scale=scale, downsample=downsample)
            labels = np.array([new_set[idx]['label'] for idx in range(len(new_set))])
            if (labels == 0.0).any():
                log.info("Skipping {:s}, found bad label data".format(dset))
                continue
            self.datasets.append(new_set)
            self.dataset_sizes.append(len(new_set))
        self.transform = transform

    def __len__(self):
        return sum(self.dataset_sizes)

    def __getitem__(self, item):
        item_idx = item
        running_sum = np.cumsum(self.dataset_sizes)
        start_idxs = np.hstack([[0], running_sum[:-1]])
        dset_idx = None
        dset_idx = np.where(start_idxs <= item_idx)[0][-1]
        dset = self.datasets[dset_idx]
        epoch_idx = (item_idx - start_idxs[dset_idx])
        return dset[epoch_idx]


if __name__ == "__main__":
    #retrieve_hdf5_files_from_s3('ml_extracted_data', 'p300_epoch_data')
    # data_path = 'p300_epoch_data/epoch_data_1218072501a_estudiotraining.hdf5'
    myset = HDF5DataSetP300Basic('p300_avg.hdf5')
    # blah = myset[45]
    #metaset = AggregateDataSet('./p300_epoch_data', num_dsets=10)
    #metaset[100]
    # dl = data.DataLoader(metaset, batch_size=100)
    # for batch in dl:
    #     print(len(batch))
