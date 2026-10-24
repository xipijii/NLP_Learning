# 来自 main.py 文件的代码
# -*- coding: utf-8 -*-

import torch
import random
import os
import numpy as np
import logging
from config import Config
from c4_model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
# [DEBUG, INFO, WARNING, ERROR, CRITICAL]

# ❓
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ✍️level=logging.INFO 设置日志的最低显示级为INFO；asctime-时间戳；name-日志器名称；levelname-日志级别；message-日志消息
logger = logging.getLogger(__name__)
    # ✍️创建日志器对象 __name__ 是当前模块的名称

"""
模型训练主程序
"""

seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
    #
torch.cuda.manual_seed_all(seed)


"""
创建保存模型的目录
加载训练数据
加载模型
标识是否使用GPU
加载优化器
加载效果测试类
训练
    获取批次数据（使用GPU时，数据
    反向传播、梯度更新、梯度归零
    记录loss、打印日志（记录0%，50%和100%的loss值；记录批次的loss平均值）
    评估准确率
"""
def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)  # ✍️所有权重的初始化都受种子控制
        # ✍️先加载数据再定义模型，为了获取分类类别数量
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]  # ❓
                    # ✍️：将数据从 CPU 内存转移到 GPU 显存，以加速训练

            # ✍️梯度归零、计算loss、反向传播、梯度更新
            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:  # ✍️控制日志输出频率，避免打印太多信息
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索 ‼️
    for model in ["gated_cnn", 'bert', 'lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        print("最后一轮准确率：", main(Config), "当前配置：", Config)


# 来自 loader.py 文件的代码
# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""
""" ✍️DataGenerator
__init__
    1. 定义label与index的对应关系
    2. 配置分类的大小（分类层输出纬度）（配置文件）
    3. 定义词表
        Bert的词表
        其余模型的词表
    4. 加载数据【使数据返回（x=input_ids, y=label）】

load()
    1. 读取数据
    2. tag转换为对应的index
    3. title转换为对应的input_ids（Bert用自己的词表）
"""
""" ✍️用torch自带的DataLoader类封装数据
    定义data生成器
    使用DataLoader处理数据（传入batch，shuffle）
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)  # ✍️Bert有自己的词表和大小，无需使用这里
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                    input_id = self.padding(input_id)  # ✍️👀纪鑫萍加的，测试 todo
                        # ✍️使用Bert时，要用Bert的词表
                        # ✍️将文本转化成词表里对应的序列数字，用于进embedding层
                else:
                    input_id = self.encode_sentence(title)  # ✍️自己写，自己整词表
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)  # ✍️解决不同长度的文本放入一个batch里训练。最大长度应该按95%的内容都小于这个数
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
        # ✍️    # ↑ shuffle=True 时会随机打乱数据顺序，受 random.seed() 和 torch.manual_seed() 控制
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])


# 来自 c4_model.py 文件的代码
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
建立网络模型结构
"""

""" ✍️
__init__
1. 定义 参数
2. 根据配置中的模型类型定义encoder
3. 定义 分类线性映射
4. 定义 pooling
5. 定义 loss计算方式
__forward__
1. 输入编码层
2. 获取编码层结果（RNN和Bert输出两个结果
3. pooling得到句向量
4. 分类层
5. 计算loss / 获取预测值
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]  # ✍️分类的数量（根据任务所需进行设置）
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False  # ✍️作为后面区分embedding和❓的标识
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # ✍️Embedding 层的权重矩阵是随机初始化的，受种子影响

        # ✍️下面各种神经网络层的初始化 所有这些参数的初始化都受种子控制

        # ✍️根据配置中的模型类型定义encoder
        if model_type == "fast_text":
            self.encoder = lambda x: x  # ✍️没有中间层，相当于直接用embedding的结果
            # ✍️lamda 参数列表: 函数体 【输入x，返回x】❓
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":  # ❓
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":  # ✍️把几个cnn堆起来的写法
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            # ✍️一定要使用路径文件，不然相当于随机初始化
            # ✍️Bert的效果好，来自两个方面：1.预训练 2.transformer
            hidden_size = self.encoder.config.hidden_size  # ❓
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        # ✍️进行分类映射
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]  # ✍️其实可以在这里判断，定义pooling层
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            # sequence_output:batch_size, max_len, hidden_size
            # pooler_output:batch_size, hidden_size
            x = self.encoder(x)  # ✍️Bert中的encoder自带embedding层（与其他的网络层的区别
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x = self.encoder(x)  # input shape:(batch_size, sen_len, input_dim)

        if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果 ‼️结果是两个时，取RNN/Bert的前一个
            x = x[0]
        # 可以采用pooling的方式得到句向量
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])  # ❓x.shape[1]=max_len❓
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)
        # ✍️做pooling时，对句子长度的纬度做
        # ✍️output shape: (batch_size, input_dim) 有没有1？squeeze掉了没有了

        # 也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]
        predict = self.classify(x)  # input shape:(batch_size, input_dim)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        # ✍️
        # L_out = L_in + 2*padding - kernel_size + 1
        # 2*padding = kernel_size - 1
        # padding = int((kernel_size - 1) / 2)
        # 输入10，输出10，kernel_size=3,
        # padding = 1, 10=10+2-3+1=10
        pad = int((kernel_size - 1) / 2)  # 保持每层长度不变，便于堆叠和处理（pooling
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):  # x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


"""
✍️‼️第二个卷积（门）做一个sigmoid，再和第一个卷积相乘，相当于也是保留重要信息，忽略不重要的信息
"""
class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)  # ❓


class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        # ModuleList类内可以放置多个模型，取用时类似于一个列表
        # ✍️
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        # 仿照bert的transformer模型结构，将self-attention替换为gcnn ‼️
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  #之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  #一层线性
            l1 = torch.relu(l1)  #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1)  #二层线性
            x = self.bn_after_ff[i](x + l2)  #残差后过bn
        return x

"""
先过一个RNN，输出的结果再过一个CNN
"""
class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)  # ✍️用全部时间步的状态
        x = self.cnn(x)
        return x


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]  # ✍️把Bert的输出送到LSTM❓
            # ✍️Bert的第二个输出指的是[CLS]token的输出
        x, _ = self.rnn(x)
        return x


class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]  # ✍️Bert的第二个输出指的是[CLS]token的输出
        x = self.cnn(x)
        return x


class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True  # ✍️加载某些层时的设置
            # ❓

    def forward(self, x):
        layer_states = self.bert(x)[2]  # (13, batch, len, hidden)

            # ✍️当 output_hidden_states=True 时，BERT 返回一个元组：
                # [0]: last_hidden_state - 最后一层输出
                # [1]: pooler_output - [CLS] token 的池化输出
                # [2]: hidden_states - 所有 13 层的隐藏状态（这就是 [2] 的含义）
                # ✍️第 0 层：Embedding 层的输出；第 1-12 层：12 个 Transformer Encoder 层的输出
        layer_states = torch.add(layer_states[-2], layer_states[-1])
            # ✍️将两层的输出逐元素相加（element-wise addition）=输出形状=(batch_size, sequence_length, hidden_size)
        return layer_states


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    # Config["class_num"] = 3
    # Config["vocab_size"] = 20
    # Config["max_length"] = 5
    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))

    # model = TorchModel(Config)
    # label = torch.LongTensor([1,2])
    # print(model(x, label))


# 来自 evaluate.py 文件的代码
# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""
""" ✍️

__init__（配置，模型，日志）
1. 定义：模型、日志
2. 加载验证数据
3. 定义统计测试结果的变量

eval
1. 日志记录（第几轮的评估），开启评估模式，清空上一轮的统计结果
2. 循环验证数据开启验证和统计
    如果GPU可用，将数据复制到GPU
    通过模型得到预测值
    调用write_stats比较测试值和真实值
3. 计算、返回准确率

write_stats（测试值，真实值）
1. 断言：测试值和真实值数据量相等
2. 获取测试值中概率最大的角标，与真实值比较是否相等，统计正误

show_stats
1. 根据统计值输出正确率

"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():  # ‼️禁用梯度计算
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)


# 来自 config.py 文件的代码
# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/train_tag_news.json",
    "valid_data_path": "../data/valid_tag_news.json",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"/Users/juju/PycharmProjects/py_project01/nlp_project02/week6_语言模型和预训练/01_reference/bert-base-chinese",
    "seed": 987
}

