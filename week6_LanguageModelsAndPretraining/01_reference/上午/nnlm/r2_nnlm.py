#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

"""
基于pytorch的rnn语言模型
"""

""" 笔记📒
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
"""

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        self.layer = nn.RNN(input_dim, input_dim, num_layers=2, batch_first=True)
        self.classify = nn.Linear(input_dim, len(vocab) + 1)
        self.dropout = nn.Dropout(0.1)  # ✍️这里为何考虑加上dropout❓
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #output shape:(batch_size, sen_len, input_dim)
        x, _ = self.layer(x)      #output shape:(batch_size, sen_len, input_dim)
        x = x[:, -1, :]        #output shape:(batch_size, input_dim)
        x = self.dropout(x)
        y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred, y) #[1*vocab_size] []
        else:
            return torch.softmax(y_pred, dim=-1)

#读取语料获得字符集
#输出一份
def build_vocab_from_corpus(path):
    vocab = set()
    with open(path, encoding="utf8") as f:
        for index, char in enumerate(f.read()):
            vocab.add(char)
    vocab.add("<UNK>") #增加一个unk token用来处理未登录词
    writer = open("../../../02_practice/vocab.txt", "w", encoding="utf8")
    for char in sorted(vocab):
        writer.write(char + "\n")
    return vocab

#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]        #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
        vocab["\n"] = 1
    return vocab

#加载语料
def load_corpus(path):
    return open(path, encoding="utf8").read()

# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
# ✍️自监督任务：根据输入语料，下一个字即为真实值
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
        # ✍️在文本长度和窗口大小允许范围内 确定起始位置
    end = start + window_size  # ✍️确定终点位置
    window = corpus[start:end]  # ✍️输入 x （包左不包右）
    target = corpus[end]  # ✍️输出 y （右角标即为下一个字）
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]   # 将字转换成序号
    y = vocab[target]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 计算文本ppl
""" 笔记📒
我的理解 ❌一句话的成句概率，需要输入一句话，从左到右平移窗格，依次计算概率分布后，（进行log10下的加和，利用公式计算）
1. 在测试句子上随机截取窗格大小的文本
2. 文本的前n-1个字是输入x，第n个字是输出y
3. 使用模型计算输入x下的概率分布中，y字的概率
4. 使用公式计算ppl
"""
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)  # ✍️要么是0，要么是文本内窗格顶到最右边的左边界
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 300       #每次训练样本个数  # 128
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 128        #每个字的维度
    window_size = 6       #样本文本长度
    vocab = build_vocab("../../../02_practice/vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim)    #建立模型
    if torch.cuda.is_available():  # ✍️检查系统中是否有可用的 NVIDIA GPU 和 CUDA 环境
        model = model.cuda()  # ✍️将 PyTorch 模型的所有参数和数据从 CPU 内存复制到 GPU 显存
            # ✍️之后模型的前向传播和反向传播都会在 GPU 上执行；速度提升：10-100 倍（取决于模型大小和数据量）
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()  # ✍️准备数据（也要迁移到 GPU）
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            watch_loss.append(loss.item())
            loss.backward()      #计算梯度
            optim.step()         #更新权重
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

    # ✍️=============保存模型权重=============
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

# 训练corpus文件夹下的所有语料，根据文件名将训练后的模型放到莫得了文件夹
def train_all():
    for path in os.listdir("corpus"):
        corpus_path = os.path.join("corpus", path)  # ❓
         # ✍️ # 自动适配系统的路径拼接
            # Linux/Mac → corpus/titles.txt
            # Windows   → corpus\titles.txt
        train(corpus_path)


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    # train("corpus.txt", True)
    train_all()