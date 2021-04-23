# -*- coding: utf-8 -*-

"""
------------------------------------------
@author: 兴微
@software: PyCharm
@file: data.py
@time: 2021/4/22 14:07
------------------------------------------
@description：
生成数据集
------------------------------------------
"""

import torch.nn.functional as F
import torch
from torch.utils.data import  DataLoader
from config import (
    train_rate,
    val_rate,
    test_rate,
    batch_size,
    ds,
    DEVICE,
)


def getDataLoader():
    ds_len = len(ds)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [int(train_rate * ds_len),
                                                                   int(val_rate * ds_len),
                                                                   int(test_rate * ds_len)])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_dl, val_dl, test_dl


def name_to_tensor(name):
    name_index = [[ds.all_letters.find(char)] for char in name]
    name_tensor = torch.tensor(name_index, dtype=torch.int64)
    name_tensor = F.one_hot(name_tensor, num_classes=ds.n_letters)
    return name_tensor.to(DEVICE)


def label_to_tensor(label):
    return torch.tensor([ds.labels.index(label)], dtype=torch.long, device=DEVICE)
