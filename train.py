import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file

import sys

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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
#%load_ext autoreload
#%autoreload 2




def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    #np.random.seed(10)
    params = parse_args('train')
    optimization = 'Adam'

    if params.stop_epoch == -1:
        params.stop_epoch = 10

    params.n_shot = 10

    fileObject = open('mini_meta_TCGA_60.pickle', 'rb')
    data_episode1 = pickle.load(fileObject)

    # for key, val in data_episode1.items():
    #     #if 'venous_invasion-COADREAD' not in key:
    #         X = val['xs']
    #         y = val['xs_class']
    #         sel = VarianceThreshold(threshold=5)
    #         X_new = sel.fit_transform(X)
    #         support_X = sel.get_support(indices=True)
    #         final_Support = np.union1d(final_Support, support_X)
    # print(final_Support.shape)

    all_datasets = []
    for key, val in data_episode1.items():
        X = val['xs']
        y = val['xs_class']
        train_set = TensorDataset(X, y)
        #print(train_set.shape)
        all_datasets.append(train_set)

    final_dataset = torch.utils.data.ConcatDataset(all_datasets)
    print((type(final_dataset)))



#    params.model='MLP10'

    # def load_sets(task, valid=False):
    #
    #     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(task._samples,
    #                                                                                 task._labels,
    #                                                                                 stratify=task._labels,
    #                                                                                 train_size=150,
    #                                                                                 test_size=100,
    #                                                                                 shuffle=True,
    #                                                                                 random_state=0
    #                                                                                 )
    #
    #     train_set = TensorDataset(Tensor(X_train), Tensor(y_train))
    #     test_set = TensorDataset(Tensor(X_test), Tensor(y_test))
    #
    #     if valid:
    #         X_test, X_valid, y_test, y_valid = sklearn.model_selection.train_test_split(X_test,
    #                                                                                     y_test,
    #                                                                                     stratify=y_test,
    #                                                                                     train_size=50,
    #                                                                                     test_size=50,
    #                                                                                     shuffle=True,
    #                                                                                     random_state=0
    #                                                                                     )
    #         valid_set = TensorDataset(Tensor(X_valid), Tensor(y_valid))
    #         return train_set, valid_set, test_set
    #

    #    return train_set, test_set




    train_few_shot_params = dict(n_way=2, n_support=10)
    base_loader=DataLoader(final_dataset,batch_size=10)
    # base_datamgr = SetDataManager(n_query=10, **train_few_shot_params)
    # base_loader = base_datamgr.get_data_loader()
    # print(base_loader)

    # test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    # val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
    # val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

    # if params.method == 'protonet':
    model = ProtoNet(model_dict[params.model], **train_few_shot_params)  # 这里的model指Conv4等

        # if params.method in ['baseline', 'baseline++'] :
        # base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
        # base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        # val_datamgr     = SimpleDataManager(image_size, batch_size = 64)
        # val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
        #
        # if params.dataset == 'omniglot':
        #     assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        # if params.dataset == 'cross_char':
        #     assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'
        #
        # if params.method == 'baseline':
        #     model           = BaselineTrain( model_dict[params.model], params.num_classes)
        # elif params.method == 'baseline++':
    #print(num_classes)


#   params.checkpoint_dir = '%s/checkpoints/%s' %(configs.save_dir, task.id[1])

    # if not os.path.isdir(params.checkpoint_dir):
    #     os.makedirs(params.checkpoint_dir)
    #
    # start_epoch = params.start_epoch
    # stop_epoch = params.stop_epoch
    #
    # model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)

