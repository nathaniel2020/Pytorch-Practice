# -*- coding: utf-8 -*-

"""
------------------------------------------
@author: 兴微
@software: PyCharm
@file: config.py
@time: 2021/4/22 14:00
------------------------------------------
@description：

------------------------------------------
"""
import torch
from Dataset import NameDataset


DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 路径
filePath = '../../data/names'
savePath = './modelSave/net.pt'

# 数据集
ds = NameDataset(filePath)

# 模型参数
INPUT_SIZE = ds.n_letters # 字母总数
HIDDEN_SIZE = 256
OUTPUT_SIZE = len(ds.labels) # 类别数
EPOCHS = 1000
lr = 1e-3
val_period = 1 # 每20个epoch打印一次


# 训练集，验证集，测试集比例
train_rate = 0.8
val_rate = 0.1
test_rate = 0.1

# 数据集batch
batch_size = 1



