# -*- coding: utf-8 -*-

"""
------------------------------------------
@author: 兴微
@software: PyCharm
@file: train.py
@time: 2021/4/22 14:12
------------------------------------------
@description：

------------------------------------------
"""
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from Model import RNN
from data import getDataLoader, name_to_tensor, label_to_tensor
from eval import cal_accuracy, predict
from torch.utils.tensorboard import SummaryWriter
from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    EPOCHS,
    lr,
    val_period,
    DEVICE,
    ds,
    savePath,
)

# ------------------- 获取数据 ---------------------- #
train_dl, val_dl, test_dl = getDataLoader()


# ------------------- 初始化模型 ---------------------- #
rnn = RNN(INPUT_SIZE,
          HIDDEN_SIZE,
          OUTPUT_SIZE).to(DEVICE)

criterion = nn.CrossEntropyLoss() # 交叉熵
optim = torch.optim.Adam(rnn.parameters(), lr=lr)

writer = SummaryWriter('../../log') # 可视化
# ------------------- 训练 ---------------------- #
bast_acc = 0 # 用以保存最佳的模型
rnn.train()
for epoch in tqdm(range(EPOCHS)):
    start = time.time()
    l_avg = 0
    for X, Y in train_dl:
        # X, Y: tuple, len=1
        X = X[0]
        Y = Y[0]

        name_tensor = name_to_tensor(X)
        hidden = rnn.init_hidden()
        output, hidden = rnn(name_tensor, hidden)
        y = label_to_tensor(Y)
        loss = criterion(output, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        l_avg += loss.item()
    l_avg /= len(train_dl)
    writer.add_scalar('loss', l_avg, epoch)

    if (epoch + 1) % val_period == 0:
        acc = cal_accuracy(rnn, val_dl)
        if acc > bast_acc:
            torch.save(rnn.state_dict(), savePath)
            best_acc = acc
        name, label = iter(test_dl).next() # 在测试集中选一个名字，进行预测
        predictions = predict(rnn, name[0])[0]
        print('epoch %d, acc %f, time %.2f sec, %s(%s) -> %s(%f)'
              % (epoch, acc, time.time() - start, name[0], label[0], predictions[0], predictions[1]))

# ------------------- 测试 ---------------------- #
rnn.load_state_dict(torch.load(savePath))
acc = cal_accuracy(rnn, test_dl)
print('test-acc: %f'%(acc, ))