#coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
"""


class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim, padding_idx=0)  #shape=(vocab_size, dim)
        # ✍️len(vocab) + 1 为什么加1❓构建词表时忘了加pad标识
        self.rnn_layer = nn.RNN(input_size=input_dim,
                                hidden_size=hidden_size,
                                batch_first=True,  # ✍️ batch_first在传入张量时，第一纬代表batch_size；这样设置就能跟embedding层连上
                                num_layers=num_rnn_layers,
                                )
        # ✍️num_layers 多层rnn，上一层的输出作为下一层的输入
        self.classify = nn.Linear(hidden_size, 2)  # w = hidden_size * 2
        # ✍️每个字分成2类 0/1 ❓为何不直接输出一个概率分布？其实是可以的，2分类比较特殊而已
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
        # ✍️ignore_index=-100 之前会做y的padding补0，但是做这个label标签就不能补0了，这里定义成一个特定的数，后续会默认不计算loss
        # ✍️这里的两个view操作是为了能够满足输入交叉熵的形状

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # input shape: (batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)
        # ✍️不适合在embedding前就展平，否则传入rnn时，每句都是下一句的上文，hidden会无限延续
        x, _ = self.rnn_layer(x)  # output shape:(batch_size, sen_len, hidden_size)
        # ✍️这里需要的就是每个字对应的输出
        y_pred = self.classify(x)  # output shape:(batch_size, sen_len, 2) -> y_pred.view(-1, 2) (batch_size*sen_len, 2)
        # ✍️.view(-1, 2) 前面的-1表示自己推算    展平 连成同一个样本
        # ✍️对上一层每一个字做2分类
        if y is not None:
            # cross entropy  为了配合交叉熵的需要，y_pred和y要展平
            # y_pred : n, class_num    [[0.1,0.9], [0.7,0.3]...[0.4, 0.6]]
            # y      : n               [ 1,         0,  ...   , 1       ]

            #y:batch_size, sen_len  = 2 * 5
            #[[0,0,1,0,1],[0,1,1, -100, -100]]  y
            #[0,0,1,0,1,  0,1,1,-100.-100]    y.view(-1) shape= n = batch_size*sen_len
            # ✍️y值进行同样的处理，.view(-1)将整个y连成一个样本

            return self.loss_func(y_pred.view(-1, 2), y.view(-1))  # ✍️❓y.view(-1)是不是不用加？可以试验一下
        else:
            return y_pred  # 这里加不加归一化，因为这里只看较大值，softmax不会改变数的相对关系


"""
Dataloader可以认可的类
"""
class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        """ ✍️
        构建数据集的类‼️   传给DataLoader用的
        :param corpus_path: 语料文件的路径
        :param vocab: 字表（转sequence用，给embedding层做输入用的，以形成传入模型的x
        :param max_length: 字符串的最大长度（切割语料文件用的，输入模型的x 一个batch内的向量需要等长
        """
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()  # ✍️ 自动加载数据，将其加载进data属性里

    def load(self):
        """ ✍️
        数据加载
        1. 根据字表将句子转成整数序列
        2. 根据jieba分词的结果将句子打标（每个字是否是一个词的边界）
        3. 根据最大长度将“整数序列”和“label”补0
        4. 将二者转成Tensor后加入data
        :return: [整数序列, 对应的label（说明每个整数/字符是否是词边界）]
        """
        self.data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)  # ✍️每一行 用jieba做分词
                label = sequence_to_label(line)  # ✍️得到label序列
                sequence, label = self.padding(sequence, label)  # ✍️做padding
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                # 使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10000:
                    break

    # 将文本截断或补齐到固定长度 ✍️因为输入的语料不是等长的了（之前都是手动构造的等长的
    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]  # ‼️
        sequence += [0] * (self.max_length - len(sequence))
            # ✍️sequence之后补0
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
            # ✍️label补一个-100的index，前面已经在交叉熵函数中传入这个值了，就不会计算loss了
        return sequence, label

    # ✍️定义的这个类要有两个方法
    def __len__(self):  # 有多少数据
        return len(self.data)

    def __getitem__(self, item):  # 如何获取里面的其中一个样本
        return self.data[item]


# 文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence


# 基于结巴生成分级结果的标注
"""✍️
1. 通过jieba进行分词
2. 初始化全0的标签列表
3. 循环分词，分词的最后一位对应的角标位置的标签为1---即词的边界
"""
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label


# 加载字表
""" ✍️
这是给embedding层用的【字-角标】词表
1. 字表加入<pad>标识 【⚠️ 不加也没事❓为啥，之前干啥用了？之前好像是数据构造的时候就按照所需长度做的】
2. 加载文件，使文件中的字词对应到序号
3. 加入<unk>标识
"""
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab


# 建立数据集
"""
jieba分词后形成的训练数据
1. 实现Dataset类
    将语料文件使用jieba分词，并生成对应的label（每个字是不是一个词的边界）
    实现“获取长度”、“通过角标获取对应数据”的方法
2. 获取DataLoader处理后的x, y数据集
"""
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length)  # diy __len__ __getitem__
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)  #torch
    # ✍️ DataLoader pytorch专有的类，入参dataset要求是一个类，需要有两个自定义好的方法；；背后还有多线程的加载
    # ✍️batch_size封装了分割样本数据为一个batch的方法
    # ✍️shuffle 模型训练时，需要把顺序打乱（比如短样本/同类样本在前），保证随机性，避免对模型有不好的影响（前部分一个规律，后面另一个规律，然而是没必要的规律）
    # 特定情况不需要，比如课程训练，就是由简单到难的数据喂进来
    return data_loader


def main():
    epoch_num = 10  #训练轮数  ✅5
    batch_size = 20  #每次训练样本个数
    char_dim = 50  #每个字的维度
    hidden_size = 100  #隐含层维度
    num_rnn_layers = 1  #rnn层数
    max_length = 20  #样本最大长度
    learning_rate = 1e-3  #学习率
    vocab_path = "chars.txt"  #字表文件路径
    corpus_path = "../corpus.txt"  #语料文件路径
    vocab = build_vocab(vocab_path)  #建立字表
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  #建立数据集
    # ✍️构建数据集使用新方法【DataLoader】- 传入“语料文件路径”❓，“字表”
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)  #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    #训练开始
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            optim.zero_grad()  # 梯度归零
            loss = model.forward(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return


# 最终预测
def predict(model_path, vocab_path, input_strings):
    # 配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 1  # rnn层数
    vocab = build_vocab(vocab_path)  # 建立字表
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        # 逐条预测
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  # 预测出的01序列
            # 在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ｜ ")
                else:
                    print(input_string[index], end="")
            print()


""" ✍️
模型训练好之后就可以用其来分词了
效果不是特别好的原因：
1. 训练数据少
2. 模型小
3. 使用的是jieba的语料（偷懒了）
"""
if __name__ == "__main__":
    # main()
    input_strings = ["同时，国内有望出台，新汽车刺激方案",
                     "沪胶后市有望延续强势！",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]

    predict("model.pth", "chars.txt", input_strings)
