import random

import torch
from torch import nn
import os  # ✍️加载文件夹用的
import numpy as np
import math

"""
1. 循环语料文件，路径拼接，对路径下的每个语料开始训练
2. 训练
（1）设置参数
（2）加载词表
    \n 换行符需要在词表中标识
    首个pad，末个unk
（3）加载语料
（4）构建模型和优化器
（5）检查GPU是否可用，复制模型到GPU
（6）构建数据
    在语料中随机截取窗格
    窗格内前n-1字为x，第n字为y
3. 计算ppl
（1）测试句子从左开始截取窗口大小
循环
（2）窗格内n-1个字是x，第n个字是y
（3）使用模型根据输入的x得到概率分布，获取y对应的概率值
（4）概率值进行log10下求和
退出循环
（5）使用公式计算ppl
"""


class TorchRNNModel(nn.Module):
    """
    1. embedding 字表大小，字符纬度，
    2. rnn hidden_size, input_size, batch_first
    3. linear input_size, 字表大小
    4. activation
    5. crossEntropy
    """

    def __init__(self, vocab, hidden_size, input_size):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(vocab),  # ❓为何构建词表时留空了还需要+1
                                      embedding_dim=input_size,
                                      padding_idx=0)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.1)  # ✍️0.1的概率进行丢弃
        self.linear = nn.Linear(hidden_size, len(vocab))
        self.activation = torch.softmax
        # self.loss = nn.CrossEntropyLoss
        self.loss = nn.CrossEntropyLoss()

    """
    1. rnn取output的最后一纬
    2. rnn之后进行dropout
    """

    def forward(self, x, y=None):
        try:
            x = self.embedding(x)
        except Exception as e:
            print("程序运行出错，异常信息：", e)
          # (batch_size, sen_len, input_size)
        x, _ = self.rnn(x)  # (batch_size, sen_len, hidden_size)
        # x = x[: -1:]  # ❌
        x = x[:, -1, :]
        x = self.dropout(x)
        y_pre = self.linear(x)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return self.activation(y_pre, -1)  # 对最后一维做softmax


def train_all(dir_path):
    for path in os.listdir(dir_path):
        corpus_path = os.path.join(dir_path, path)
        train(corpus_path)


def build_vocab():
    vocab = {"<pad>": 0, "\n": 1}
    with open("vocab.txt", encoding="utf-8") as f:
        for index, line in enumerate(f):
            # char = line.strip()  # ❌
            char = line[:-1]  # ‼️
            vocab[char] = index + 2
    vocab["<unk>"] = len(vocab)
    return vocab

# def build_vocab(vocab_path):
#     vocab = {}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]        #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#         vocab["\n"] = 1
#     return vocab


def build_data(vocab, corpus, window_size):
    left = random.randint(0, len(corpus) - 1 - window_size)
    input = corpus[left: left + window_size]
    output = corpus[left + window_size]  # ⚠️1个值
    x = [vocab.get(char, vocab.get("<unk>")) for char in input]
    # y = [vocab.get(char, vocab.get("<unk>")) for char in output]  # ❌无需这样
    y = vocab.get(output, vocab.get("<unk>"))  # ‼️
    return x, y


def build_dataset(vocab, corpus, train_sample_size, window_size):
    x_dataset = []
    y_dataset = []
    for i in range(train_sample_size):
        x, y = build_data(vocab, corpus, window_size)
        x_dataset.append(x)
        y_dataset.append(y)
    return torch.LongTensor(x_dataset), torch.LongTensor(y_dataset)


def load_corpus(corpus_path):
    return open(corpus_path, encoding="utf-8").read()


def train(corpus_path, save_weight=True):
    epoch_num = 20
    train_sample_size = 10000
    batch_size = 20
    vocab = build_vocab()
    corpus = load_corpus(corpus_path)
    window_size = 18
    input_size = 256
    hidde_size = 128
    learning_rate = 0.01
    watch_loss = []

    model = TorchRNNModel(vocab, hidde_size, input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_dataset, y_dataset = build_dataset(vocab, corpus, train_sample_size, window_size)

    model.train()
    for epoch in range(epoch_num):
        if epoch == 19:
            print()
        for index in range(train_sample_size // batch_size):
            x = x_dataset[index * batch_size: (index + 1) * batch_size]
            y = y_dataset[index * batch_size: (index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            # watch_loss.append(loss.items())  # ❌
            watch_loss.append(loss.item())

        print(f"第{epoch + 1}轮，平均loss值为：{np.mean(watch_loss)}")
    # ✍️=============保存模型权重=============
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

def cal_perplexity(model, sentence, window_size):
    vocab = build_vocab()
    pros = 0
    model.eval()  # ‼️
    with torch.no_grad:
        for i in range(1, len(sentence) - 1 - window_size):
            input = sentence[:i]
            output = sentence[i]
            x = [vocab.get(char, vocab.get("<unk>")) for char in input]
            x = torch.LongTensor(x)
            y_target_index = vocab.get(output, vocab.get("<unk>"))
            if torch.cuda.is_available():
                x = x.cuda()
            y_pre = model(x)
            # ‼️获取p值
            p = y_pre[y_target_index]
            pros += math.log(p, 10)

    # return 2 ** (-1/len(vocab) * pros)
    return 2 ** (-1/len(sentence) * pros)



if __name__ == '__main__':
    train_all("corpus")