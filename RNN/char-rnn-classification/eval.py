# -*- coding: utf-8 -*-

"""
------------------------------------------
@author: 兴微
@software: PyCharm
@file: eval.py
@time: 2021/4/22 14:23
------------------------------------------
@description：

------------------------------------------
"""
from config import ds
from data import name_to_tensor

def cal_accuracy(model, dl):
    '''
    计算误差
    :param model: 模型
    :param data_gen: 数据生成器
    :return:
    '''
    model.eval()
    acc = 0
    num = 0
    for X, Y in dl:
        # X, Y: tuple, len=1
        X = X[0]
        Y = Y[0]
        name_tensor = name_to_tensor(X)
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
    name_tensor = name_to_tensor(name)
    output, _ = model(name_tensor, hidden)

    # topv: 下标 tensor (1, n_predictons)
    # topi: 值 tensor (1, n_predictons)
    topv, topi = output.data.topk(n_predictons, 1, True) # 取概率最大的前几个

    prediction = [(ds.labels[topi[0][index].item()], topv[0][index].item()) for index in range(n_predictons)]
    return prediction
