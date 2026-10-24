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
