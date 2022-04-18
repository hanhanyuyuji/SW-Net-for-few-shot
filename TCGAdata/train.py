import sys

import meta_dataloader.TCGA

import numpy as np
import data.gene_graphs
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
# %load_ext autoreload
# %autoreload 2

tasks = meta_dataloader.TCGA.TCGAMeta(download=False,
                                      min_samples_per_class=10)
print(len(tasks.task_ids))
clinical_variable_list = []
for taskid in sorted(tasks.task_ids):
    clinical_variable_list.append(taskid[0])
print(np.unique(clinical_variable_list))

task = meta_dataloader.TCGA.TCGATask(('Expression_Subtype','LUNG'))
print(task.id)
print(task._samples.shape)
print(np.asarray(task._labels).shape)
print(collections.Counter(task._labels))