# -*- coding: utf-8 -*-

"""
------------------------------------------
@author: 兴微
@software: PyCharm
@file: Dataset.py
@time: 2021/4/22 13:57
------------------------------------------
@description：

------------------------------------------
"""
from torch.utils.data import Dataset
import string
from pathlib import Path
import unicodedata
import random

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