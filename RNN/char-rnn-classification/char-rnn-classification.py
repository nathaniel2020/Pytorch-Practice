# -*- coding: utf-8 -*-

"""
------------------------------------------
@author: 兴微
@software: PyCharm
@file: char-rnn-classification.py
@time: 2021/4/14 14:07
------------------------------------------
@description：

------------------------------------------
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
import unicodedata
import string
from pathlib import Path

class NameDataset(Dataset):
    def __init__(self, filePath):
        # 姓氏中所有的字符
        # string.ascii_letters是大小写各26字母
        self.all_letters = string.ascii_letters + " .,;'"
        # 字符的种类数
        self.n_letters = len(self.all_letters)
        self.data, self.labels = self.prepare(filePath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def prepare(self, filePath):
        '''
        处理数据
        :param filePath: 数据路径
        :return: list->tuple (name, label)
        '''
        labels = []
        data = []

        data_path = Path(filePath)
        files = list(data_path.glob('*.txt'))

        def read_names(file):
            names = open(file).read().strip().split('\n')
            return [self.unicode_to_ascii(name) for name in names]

        for file in files:
            names = read_names(file)
            file_name = file.name[:-4]
            labels.append(file_name)
            data.extend([(name, file_name) for name in names])
        random.shuffle(data)
        return data[:1000], labels

    # 将Unicode码转换成标准的ASCII码
    def unicode_to_ascii(self, s):
        '''

        :param s: unicode_to_ascill编码的字符串
        :param all_letters: 大小写各26字母 + " .,;'"
        :return:
        '''
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def name_to_tensor(self, name):
        name_index = [[self.all_letters.find(char)] for char in name]
        name_tensor = torch.tensor(name_index, dtype=torch.int64)
        name_tensor = F.one_hot(name_tensor, num_classes=self.n_letters)
        return name_tensor.to(DEVICE)

    def label_to_tensor(self, label):
        return torch.tensor([self.labels.index(label)], dtype=torch.long, device=DEVICE)

class RNN(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        # self.softMax = nn.Softmax()

    # def forward(self, input, hidden):
    #     # 每次只有一个单词，故而batch_size=1
    #     # input: tensor (batch_size=1, input_size)
    #     # hidden: tensor (batch_size=1, hidden_size)
    #     combined = torch.cat((input, hidden), dim=1) # tensor (batch_size=1, input_size + hidden_size)
    #     output = self.i2o(combined) # (batch_size=1, output_size)
    #     hidden = self.i2h(combined) # (batch_size=1, hidden_size)
    #     return output, hidden

    def forward(self, input, hidden):
        # 每次只有一个单词，故而batch_size=1
        # input: tensor (time_step, batch_size=1, input_size)
        # hidden: tensor (batch_size=1, hidden_size)
        for step in range(input.size()[0]):
            combined = torch.cat((input[step].float(), hidden), dim=1) # tensor (batch_size=1, input_size + hidden_size)
            output = self.i2o(combined) # (batch_size=1, output_size)
            hidden = self.i2h(combined) # (batch_size=1, hidden_size)
        return output, hidden

    def init_hidden(self):
        '''
        初始化隐藏层参数
        :return:
        '''
        return torch.zeros((1, self.hidden_size), device=DEVICE) # (batch_size=1, hidden_size)

filePath = '../../data/names'
ds = NameDataset(filePath)
# 训练集，验证集，测试集比例
train_rate = 0.8
val_rate = 0.1
test_rate = 0.1
ds_len = len(ds)
train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [int(train_rate * ds_len),
                                                                     int(val_rate * ds_len),
                                                                     int(test_rate * ds_len)])
batch_size = 1
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

# ------------------- 定义参数 ---------------------- #

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = ds.n_letters # 字母总数
HIDDEN_SIZE = 256
OUTPUT_SIZE = len(ds.labels) # 类别数
EPOCHS = 1000

lr = 1e-3
val_period = 20 # 每20个epoch打印一次
savePath = './modelSave/net.pt'

rnn = RNN(INPUT_SIZE,
          HIDDEN_SIZE,
          OUTPUT_SIZE).to(DEVICE)

criterion = nn.CrossEntropyLoss() # 交叉熵
optim = torch.optim.Adam(rnn.parameters(), lr=lr)


# ------------------- 计算准确度，预测 ---------------------- #
def cal_accuracy(model, val_dl):
    '''
    计算误差
    :param model: 模型
    :param data_gen: 数据生成器
    :return:
    '''
    model.eval()
    acc = 0
    num = 0
    for X, Y in val_dl:
        # X, Y: tuple, len=1
        X = X[0]
        Y = Y[0]
        name_tensor = ds.name_to_tensor(X)
        hidden = model.init_hidden()
        output, _ = model(name_tensor, hidden)
        label = output.argmax(dim=1).item()
        if ds.labels[label] == Y:
            acc += 1
        num += 1
    model.train()
    return acc / num


def predict(model, name, n_predictons=3):
    '''
    给定姓名进行预测
    :param model:
    :param name: str 姓名
    :param n_predictons: int top-K个类别
    :return: list->tuple [(name, value)]
    '''
    hidden = model.init_hidden()
    name_tensor = ds.name_to_tensor(name)
    output, _ = model(name_tensor, hidden)

    # topv: 下标 tensor (1, n_predictons)
    # topi: 值 tensor (1, n_predictons)
    topv, topi = output.data.topk(n_predictons, 1, True) # 取概率最大的前几个

    prediction = [(ds.labels[topi[0][index].item()], topv[0][index].item()) for index in range(n_predictons)]
    return prediction

# ------------------- 训练 ---------------------- #
loss_all = []
bast_acc = 0 # 用以保存最佳的模型
rnn.train()
for epoch in tqdm(range(EPOCHS)):
    start = time.time()
    l_avg = 0
    for X, Y in train_dl:
        # X, Y: tuple, len=1
        X = X[0]
        Y = Y[0]

        name_tensor = ds.name_to_tensor(X)
        hidden = rnn.init_hidden()
        output, hidden = rnn(name_tensor, hidden)
        y = ds.label_to_tensor(Y)
        loss = criterion(output, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        l_avg += loss.item()
    l_avg /= len(train_dl)
    loss_all.append(l_avg)

    if (epoch + 1) % val_period == 0:
        acc = cal_accuracy(rnn, val_dl)
        if acc > bast_acc:
            torch.save(rnn.state_dict(), savePath)
            best_acc = acc
        name, label = iter(test_dl).next() # 在测试集中选一个名字，进行预测
        predictions = predict(rnn, name[0])[0]
        print('epoch %d, acc %f, time %.2f sec, %s(%s) -> %s(%f)'
              % (epoch, acc, time.time() - start, name, label[0], predictions[0], predictions[1]))


# ------------------- 测试集 测试 ---------------------- #
rnn = RNN(INPUT_SIZE,
          HIDDEN_SIZE,
          OUTPUT_SIZE).to(DEVICE)
rnn.load_state_dict(torch.load(savePath))
acc = cal_accuracy(rnn, test_dl)
print('test-acc: %f'%(acc, ))