import torch.nn
from torch.optim import Adam, SGD
from torch import nn
from torchcrf import CRF
from transformers import BertModel

"""
建立网络模型结构
__init__
1. 从config中获取：hidden_size，vocab_size，class_num，num_layers，是否使用条件随机场
2. 定义网络层：embedding，lstm，分类层，条件随机场，loss函数

forward
1. 预测：embedding，layer，分类层获取预测值
2. 如果输入真实标签：
（1）使用条件随机场：
    掩码：标识真实标签（大于-1的）
    对（预测值，真实标签，掩码）进行条件随机场层的处理【取反，获取loss值】
（2）不使用：
    使用交叉熵函数获取loss值
3. 未输入真实标签：
（1）使用条件随机场：
    使用条件随机场层进行解码（这里是维特比解码）
（2）不使用：
    返回预测值
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.use_crf = config["use_crf"]
        self.use_bert = config["use_bert"]

        # todo
        if self.use_bert:
            self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.bert.config.hidden_size
            self.classification = nn.Linear(hidden_size, class_num)
        else:
            vocab_size = config["vocab_size"] + 1
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True
                                 , bidirectional=True, num_layers=num_layers)
            self.classification = nn.Linear(hidden_size * 2, class_num)

        self.crf_layer = CRF(class_num, batch_first=True)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # ❓ignore_index

    def forward(self, x, target=None):
        if self.use_bert:  # todo
            x, _ = self.bert(x)
        else:
            x = self.embedding(x)
            x, _ = self.layer(x)

        predict = self.classification(x)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


"""
选择优化器
1. 从config中获取优化器和学习率
2. 根据优化器类型，定义并返回优化器
"""


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    from config import Config

    model = TorchModel(Config)
