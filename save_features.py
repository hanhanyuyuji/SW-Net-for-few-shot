import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file

from TCGAdata.meta_dataloader import TCGA

import numpy as np
from TCGAdata.data import gene_graphs
import collections
import sklearn.metrics
import sklearn.model_selection
import random
from collections import OrderedDict
import pandas as pd
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

import pickle
from sklearn.feature_selection import VarianceThreshold


def save_features(model, data_loader, outfile ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    print(max_count)
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):

        print('{:d}/{:d}'.format(i, len(data_loader)))
        #x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
            print([max_count]+list( feats.size()[1:]))
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    params = parse_args('save_features')
    #assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'



    split = params.split

    tasks = TCGA.TCGAMeta(download=False,
                          min_samples_per_class=10)


    def findTask():
        for task in tasks:
            if task.id == ('Expression_Subtype', 'LUNG'):
                print(task.id[1])
                print(task._samples.shape)
                print(np.asarray(task._labels).shape)
                return task


    task = findTask()



    params.model = 'MLP10'



    def load_sets(task, valid=False):

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(task._samples,
                                                                                    task._labels,
                                                                                    stratify=task._labels,
                                                                                    train_size=150,
                                                                                    test_size=100,
                                                                                    shuffle=True,
                                                                                    random_state=0
                                                                                    )

        train_set = TensorDataset(Tensor(X_train), Tensor(y_train))
        test_set = TensorDataset(Tensor(X_test), Tensor(y_test))

        if valid:
            X_test, X_valid, y_test, y_valid = sklearn.model_selection.train_test_split(X_test,
                                                                                        y_test,
                                                                                        stratify=y_test,
                                                                                        train_size=50,
                                                                                        test_size=50,
                                                                                        shuffle=True,
                                                                                        random_state=0
                                                                                        )
            valid_set = TensorDataset(Tensor(X_valid), Tensor(y_valid))
            return train_set, valid_set, test_set

        return train_set, test_set


    fileObject = open('mini_meta_TCGA_60.pickle', 'rb')
    data_episode1 = pickle.load(fileObject)

    final_Support = np.array([])
    for key, val in data_episode1.items():
        X = val['xq']
        y = val['xq_class']
        sel = VarianceThreshold(threshold=4.983)
        X_new = sel.fit_transform(X)
        support_X = sel.get_support(indices=True)
        final_Support = np.union1d(final_Support, support_X)
    print(final_Support.shape)

    all_datasets = []
    for key, val in data_episode1.items():
        X = val['xq']
        y = val['xq_class']
        X_new = X[:, final_Support]
        train_set = TensorDataset(X_new, y)
        # print(train_set.shape)
        all_datasets.append(train_set)

    final_dataset = torch.utils.data.ConcatDataset(all_datasets)
    print((type(final_dataset)))



    train_set, test_set = load_sets(task)
    data_loader_params = dict(batch_size=32, shuffle=True, pin_memory=True)
    data_loader = torch.utils.data.DataLoader(final_dataset, **data_loader_params)
    val_loader = torch.utils.data.DataLoader(final_dataset, **data_loader_params)

    #checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    checkpoint_dir = '%s/checkpoints/%s' % (configs.save_dir, task.id[1])
    params.save_iter=9
    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
#    elif params.method in ['baseline', 'baseline++'] :
#        modelfile   = get_resume_file(checkpoint_dir) #comment in 2019/08/03 updates as the validation of baseline/baseline++ is added
    else:
        modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5") 


    model = model_dict[params.model]()

    #model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    print(state_keys)

    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, val_loader, outfile)
