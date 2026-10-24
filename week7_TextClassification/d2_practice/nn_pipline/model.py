import torch
from torch import nn
from config import Config
from transformers import BertModel
from torch.optim import Adam, SGD

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

选择优化器（传入模型和配置）
1. 获取优化器配置参数
2. 获取学习率
3. 根据优化器类型定义优化器
"""


class TorchModel(nn.Module):

    # def __init__(self, vocab_size, hidden_size, output_size):  # ❌参数来自配置文件
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.layer_num = config["layer_num"]  # ‼️如果是给init内部用的，就不用加self了
        vocab_size = config["vocab_size"]  # ❓
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]  # ❓
        self.use_bert = False

        self.embedding = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)

        if Config["model_type"] == "fast_text":
            self.encoder = lambda x: x
        elif Config["model_type"] == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, self.layer_num, batch_first=True)
        elif Config["model_type"] == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, self.layer_num, batch_first=True)
        elif Config["model_type"] == "cnn":
            # self.encoder = nn.Conv1d(hidden_size, hidden_size, self.layer_num)  # ❓
            self.encoder = CNN(config)
        elif Config["model_type"] == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif Config["model_type"] == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif Config["model_type"] == "rcnn":
            self.encoder = RCNN(config)
        elif Config["model_type"] == "bert":
            self.encoder = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
            self.use_bert = True
            hidden_size = self.encoder.config.hidden_size  # ‼️bert配置有自己的hidden_size
        elif Config["model_type"] == "bert_lstm":
            self.encoder = BertLSTM(config)
            self.use_bert = True
            hidden_size = self.encoder.bert.config.hidden_size
        elif Config["model_type"] == "bert_cnn":
            self.encoder = BertCNN(config)
            self.use_bert = True
            hidden_size = self.encoder.bert.config.hidden_size
        elif Config["model_type"] == "bert_mid_layer":
            self.encoder = BertMidLayer(config)
            self.use_bert = True
            hidden_size = self.encoder.bert.config.hidden_size

        self.classification = nn.Linear(hidden_size, class_num)
        self.pooling_type = config["pooling_type"]

        self.loss = nn.functional.cross_entropy  # ‼️

    # 当输入真实标签，返回loss；无真实标签，返回预测值
    def forward(self, x, y=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        if isinstance(x, tuple):  # ✅
            x = x[0]

        # x = self.pooling(x.shape[1])  # 对文本长度做 ❌长度是在这里才能知道的，定义应该在这里定义
        # pooling的纬度是文本长度的纬度，
        if self.pooling_type == "avg":
            self.pooling = nn.AvgPool1d(x.shape[1])
        elif self.pooling_type == "max":
            self.pooling = nn.MaxPool1d(x.shape[1])
        x = self.pooling(x.transpose(1, 2)).squeeze()

        y_pre = self.classification(x)

        if y is not None:
            # return self.loss(y_pre, y)  # ❌
            return self.loss(y_pre, y.squeeze())  # ✅
        else:
            return torch.softmax(y_pre, dim=-1)

class CNN(nn.Module):

    def __init__(self, config):
        super(CNN, self).__init__()

        hidden_size = config["hidden_size"]
        # layer_num = config["layer_num"]

        kernel_size = config["kernel_size"]
        padding_size = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding_size)
            # ‼️重要参数：kernel_size, padding

    def forward(self, x):  # x.shape = (batch, max_len, hidden_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):

    def __init__(self, config):
        super(GatedCNN, self).__init__()

        self.cnn = CNN(config)
        self.activation = torch.sigmoid

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.activation(self.cnn(x))
        # return np.dot(x1, x2)
        return torch.mul(x1, x2)

class RCNN(nn.Module):

    def __init__(self, config):
        super(RCNN, self).__init__()

        hidden_size = config["hidden_size"]
        layer_num = config["layer_num"]

        self.rnn = nn.RNN(hidden_size, hidden_size, layer_num, batch_first=True)
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.rnn(x)
        x = self.cnn(x)
        return x

class BertLSTM(nn.Module):

    def __init__(self, config):
        super(BertLSTM, self).__init__()

        hidden_size = config["hidden_size"]
        layer_num = config["layer_num"]

        self.bert = BertModel.from_pretrained(config["model_path"])  # ❓参数是什么
        self.lstm = nn.RNN(hidden_size, hidden_size, layer_num, batch_first=True)

    def forward(self, x):
        x = self.bert(x)
        x = self.lstm(x)
        return x

class BertCNN(nn.Module):

    def __init__(self, config):
        super(BertCNN, self).__init__()

        hidden_size = config["hidden_size"]
        layer_num = config["layer_num"]

        self.bert = BertModel.from_pretrained(config["model_path"])  # ❓参数是什么
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)
        x = self.cnn(x)
        return x

class BertMidLayer(nn.Module):

    def __init__(self, config):
        super(BertMidLayer, self).__init__()

        self.bert = BertModel.from_pretrained(config["model_path"], return_dict=False)  # ❓❓❓参数是什么
        self.bert.config.output_hidden_states = True

    def forward(self, x):  # ‼️
        # ✍️当output_hidden_states = True时，Bert返回一个包含所有层状态的元组
        hidden_states = self.bert(x)[2]
        layers_state = torch.add(hidden_states[-2], hidden_states[-1])
        return layers_state

# 堆叠门控卷积神经网络
class StackGatedCNN(nn.Module):

    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.layer_num = config["layer_num"]
        self.hidden_size = config["hidden_size"]

        self.gated_cnn_layers = nn.ModuleList(GatedCNN(config) for _ in range(self.layer_num))
        self.layer_norm_after_gate = nn.ModuleList(nn.LayerNorm(self.hidden_size) for _ in range(self.layer_num))
        # self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)  # ❌
        self.linear1 = nn.ModuleList(nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.layer_num))
        self.relu = torch.relu
        # self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)  # ❌
        self.linear2 = nn.ModuleList(nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.layer_num))
        self.layer_norm_after_ff = nn.ModuleList(nn.LayerNorm(self.hidden_size) for _ in range(self.layer_num))

    def forward(self, x):
        # 仿照bert的transformer模型结构，将self-attention替换为gcnn ‼️
        for i in range(self.layer_num):
            # # gate_cnn 相当于attention层
            # gated_x = self.gated_cnn_layers(x)
            # add_x = gated_x + x
            # x = self.layer_norm_after_gate(add_x)
            # ff_x = self.linear1(x)
            # ff_x = self.relu(ff_x)
            # ff_x = self.linear2(ff_x)
            # add_x = ff_x + x
            # x = self.layer_norm_after_ff(add_x)
            # gate_cnn 相当于attention层
            gated_x = self.gated_cnn_layers[i](x)
            add_x = gated_x + x
            x = self.layer_norm_after_gate[i](add_x)
            ff_x = self.linear1[i](x)
            ff_x = self.relu(ff_x)
            ff_x = self.linear2[i](ff_x)
            add_x = ff_x + x
            x = self.layer_norm_after_ff[i](add_x)
        return x

def choose_optimizer(model, config):  # ‼️
    optimizer = config["optimizer"]
    lr = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr)

if __name__ == '__main__':
    # model = BertModel.from_pretrained(Config["model_path"], return_dict=False)
    x = torch.LongTensor([[1, 2, 3, 7, 8, 9], [2, 4, 6, 1, 3, 5]])
    # sequence_output, pooler_output = model(x)
    # print(sequence_output)
    # print(f"shape={sequence_output.shape}")

    print("=========================")
    Config["model_type"] = "stack_gated_cnn"
    model = TorchModel(Config)
    sequence_output, _ = model(x)
    print(sequence_output)
    print(f"shape={sequence_output.shape}")
