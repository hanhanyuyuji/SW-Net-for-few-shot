# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import pickle

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import Tensor
import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, batch_size):
        fileObject = open('mini_meta_TCGA_60.pickle', 'rb')
        self.data_episode1 = pickle.load(fileObject)

        #self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.sub_dataloader = []
        for key, val in self.data_episode1.items():

            X = val['xs']
            y = val['xs_class']
            train_set = CustomDataset(X, y)
            sub_data_loader_params = dict(batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0,  # use main thread only or may receive multiple batches
                                          pin_memory=False)
            self.sub_dataloader.append(torch.utils.data.dataloader(train_set,**sub_data_loader_params))

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.data_episode1.keys())


class CustomDataset(Dataset):
    def __init__(self, X,y):
        self.labels = y
        self.img_dir =X


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image=self.img_dir[idx]
        label=self.labels[idx]

        return image, label

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = 2
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
