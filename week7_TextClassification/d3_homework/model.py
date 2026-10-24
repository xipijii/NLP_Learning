from torch import nn
from transformers import BertModel
import torch


class TorchModel(nn.Module):

    def __init__(self, config):
        super(TorchModel, self).__init__()
        pretrain_model_path = config["pretrain_model_path"]
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]
        vocab_size = config["vocab_size"]
        model_type = config["model_type"]
        layer_num = config["layer_num"]
        self.use_bert = False

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)

        # ✍️fast_text, lstm, gru?, rnn, cnn, gated_cnn, stack_gated_cnn, rcnn
        # bert, bert_lstm, bert_cnn, bert_mid_layer
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=layer_num
                                   , batch_first=True)  # ‼️batch_first=True
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=layer_num
                                  , batch_first=True)  # ‼️batch_first=True
        elif model_type == "cnn":
            # self.encoder = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size)
            # ‼️使用nn的cnn，不能使卷积特征=文本长度，所以需要自己实现
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(pretrain_model_path
                                                     # , num_layers=layer_num  # ❌自己有配置，需要改配置
                                                     , return_dict=False)  # ‼️
            hidden_size = self.encoder.config.hidden_size  # ❓
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_rnn":
            self.use_bert = True
            self.encoder = BertRNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classification = nn.Linear(hidden_size, class_num)

        self.pooling_type = config["pooling_type"]
        self.activation = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        if not self.use_bert:
            x = self.embedding(x)

        x = self.encoder(x)
        if len(x) == 2:  # ❓if isinstance(x, tuple)
            x, _ = x
        # if self.pooling_type == "avg":
        #     x = nn.AvgPool1d(x.shape[2])
        # elif self.pooling_type == "max":
        #     x = nn.MaxPool1d(x.shape[2])
        if self.pooling_type == "avg":
            # 🤦🤦🤦‼️‼️‼️pooling需要先定义再使用
            self.pooling = nn.AvgPool1d(x.shape[1])
        elif self.pooling_type == "max":
            self.pooling = nn.MaxPool1d(x.shape[1])
        # 🤦🤦🤦‼️‼️‼️pooling之后的结果需要转成(batch, hidden_size)
        x = self.pooling(x.transpose(-1, -2)).squeeze()

        # y_pre = self.activation(x)  # ❌
        y_pre = self.classification(x)

        if y is not None:
            return self.loss(y_pre, y.squeeze())  # ‼️
        else:
            return self.activation(y_pre)


class CNN(nn.Module):

    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)  # kernel_size一般是奇数

        self.cnn = nn.Conv1d(hidden_size, hidden_size
                             , kernel_size, padding=pad)  # ‼️

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


"""
第二个卷积做一个sigmoid，再和第一个卷积相乘，相当于也是保留重要信息，忽略不重要的信息
"""


class GatedCNN(nn.Module):

    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        # self.sigmoid = nn.Sigmoid
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        gate = self.gate(x)
        gate = torch.sigmoid(gate)
        return torch.mul(a, gate)


"""
模仿transformer中的encoder部分
    将gatedcnn视为attention，进行残差
    再进行前反馈
"""


class StackGatedCNN(nn.Module):

    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.layer_num = config["layer_num"]

        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for _ in range(self.layer_num))
        self.ln_after_gcnn = nn.ModuleList(
            nn.LayerNorm(hidden_size) for _ in range(self.layer_num))
        self.ff_linear_1 = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(self.layer_num))
        self.ff_linear_2 = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(self.layer_num))
        self.ln_after_ff = nn.ModuleList(
            nn.LayerNorm(hidden_size) for _ in range(self.layer_num))

    def forward(self, x):
        for i in range(self.layer_num):
            gcnn_output = self.gcnn_layers[i](x)  # ‼️都需要加
            add_output = gcnn_output + x
            x = self.ln_after_gcnn[i](add_output)
            ff_output = self.ff_linear_1[i](x)
            ff_output = torch.relu(ff_output)
            ff_output = self.ff_linear_2[i](ff_output)
            x = self.ln_after_ff[i](ff_output + x)
        return x


class RCNN(nn.Module):

    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.cnn = GatedCNN(config)  # ‼️

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x


class BertLSTM(nn.Module):

    def __init__(self, config):
        super(BertLSTM, self).__init__()
        pretrain_model_path = config["pretrain_model_path"]
        layer_num = config["layer_num"]
        self.bert = BertModel.from_pretrained(pretrain_model_path
                                              # , layer_num=layer_num
                                              , return_dict=False)
        hidden_size = self.bert.config.hidden_size
        # ‼️Bert之后的hidden_size来自自己的配置
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        x, pooler_output = self.bert(x)
        x = self.lstm(x)
        return x


class BertRNN(nn.Module):

    def __init__(self, config):
        super(BertRNN, self).__init__()
        pretrain_model_path = config["pretrain_model_path"]
        layer_num = config["layer_num"]
        self.bert = BertModel.from_pretrained(pretrain_model_path
                                              # , layer_num=layer_num
                                              , return_dict=False)
        hidden_size = self.bert.config.hidden_size
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        x, _ = self.bert(x)
        x, _ = self.rnn(x)
        return x


"""
使用Bert里的多层，需要设置参数，并在第三个返回值上操作取出
"""


class BertMidLayer(nn.Module):

    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        pretrain_model_path = config["pretrain_model_path"]
        self.bert = BertModel.from_pretrained(pretrain_model_path
                                              # , num_layers=layer_num  # ❌
                                              , return_dict=False)
        self.bert.config.output_hidden_states = True  # ‼️多层使用

    def forward(self, x):
        layer_states = self.bert(x)[2]
        x = torch.add(layer_states[2], layer_states[3])
        # ‼️tensor运算都用torch里面的方法
        return x


def choose_optimizer(model, config):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), learning_rate)
