# -*- coding: utf-8 -*-

"""
------------------------------------------
@author: 兴微
@software: PyCharm
@file: word2vec.py
@time: 2021/4/16 17:40
------------------------------------------
@description：

------------------------------------------
"""
#%% md

# 实现word2vec

#%% md

## Skip-gram

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from tqdm import tqdm
from collections import Counter
import numpy as np
import random

import scipy
from sklearn.metrics.pairwise import cosine_similarity


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
#%%

C = 3 # context window
K = 15 # 15 number of negative samples
epochs = 2
MAX_VOCAB_SIZE = 10000 # 10000
EMBEDDING_SIZE = 100
batch_size = 16
lr = 0.6
#%%

file_path = '../data/word2vec/text8.{type}.txt'

type = 'train'
with open(file_path.format(type=type)) as f:
    text = f.read().lower().split(' ')
text = text[ : 100000]
vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1)) # dict 单词-次数
vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))

word_to_idx = {word: index for index, word in enumerate(vocab_dict.keys())}
idx_to_word = {index: word for index, word in enumerate(vocab_dict.keys())}

word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32) # 词的次数
word_freqs = (word_counts / np.sum(word_counts)) ** (3. / 4.)

#%%

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, word_freqs):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        super(WordEmbeddingDataset, self).__init__() # #通过父类初始化模型，然后重写两个方法
        self.text_encoded = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text] # 把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded) # nn.Embedding需要传入LongTensor类型
        self.word_to_idx = word_to_idx
        self.word_freqs = torch.Tensor(word_freqs)


    def __len__(self):
        return len(self.text_encoded) # 返回所有单词的总数，即item的总数

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_words = self.text_encoded[idx] # 取得中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1)) # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices] # tensor(list)

        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量

        # while 循环是为了保证 neg_words中不能包含背景词
        while len(set(pos_indices) & set(neg_words.numpy().tolist())) > 0:
            neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_words, pos_words, neg_words

dataset = WordEmbeddingDataset(text, word_to_idx, word_freqs)
dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2) # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding, -input_embedding) # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2) # batch_size, (window * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1) # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().cpu().numpy()

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(DEVICE, len(dataloader))
for e in range(1):
    i = 0
    for input_labels, pos_labels, neg_labels in tqdm(dataloader, desc='epoch-%d'%(e, )):
        input_labels = input_labels.long().to(DEVICE)
        pos_labels = pos_labels.long().to(DEVICE)
        neg_labels = neg_labels.long().to(DEVICE)

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()

        optimizer.step()

        i += 1
        if i % 1000 == 0:
            print('epoch', e, 'iteration', i, loss.item())

embedding_weights = model.input_embedding()
torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))


