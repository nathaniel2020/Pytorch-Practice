# -*- coding: utf-8 -*-

"""
------------------------------------------
@author: 兴微
@software: PyCharm
@file: Model.py
@time: 2021/4/22 13:59
------------------------------------------
@description：

------------------------------------------
"""

import torch.nn as nn
import torch
from config import DEVICE

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

        self.softmax = nn.Softmax(dim=1)

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