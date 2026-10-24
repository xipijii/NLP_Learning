import os.path
from torch import nn
import torch.cuda
from transformers import BertModel, BertTokenizer
import numpy as np
import json
import random
from torch.utils.data import Dataset, DataLoader

"""
基于pytorch的Bert模型实现sft
"""

"""
__init__
1.定义Bert
2.获取Bert的输出纬度
3.定义分类层
4.dp层增加随机性❓
5.定义交叉熵loss
__forward__
1.送入模型
2.得到分类结果
3.计算交叉熵
"""


class TorchModel(nn.Module):

    def __init__(self, pretrain_model_path, tokenizer):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        input_dim = self.bert.config.hidden_size
        self.classify = nn.Linear(input_dim, tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.1)
        # self.loss = nn.functional.cross_entropy(ignore_index=-100)  # ❌
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, mask=None, y=None):
        if mask is not None:
            # 显式增加维度，直接塞给 BERT 一个 4 维张量 [64, 1, 120, 120]
            # 这样它就会跳过内部的猜测和乱升维逻辑
            mask = mask.unsqueeze(1)
        x, _ = self.bert(x, attention_mask=mask)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


"""
加载语料文件
"""
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # ← 添加这行，去除首尾空白和换行符
            if not line:  # ← 跳过空行
                continue
            try:
                data = json.loads(line)
                corpus.append((data["question"], data["answer"]))
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON解析错误，跳过该行: {e}")
                print(f"   问题行内容: {repr(line[:100])}")
    return corpus


"""
加载数据
"""
# todo


"""
训练程序
1.定义参数
2.加载模型和tokenizer
3.迁移模型
4.加载优化器
5.开启轮次训练：
（1）开启训练模式
（2）定义loss列表
（3）分批：
    加载数据
    迁移数据
    模型计算loss
    计算梯度
    权重更新
    梯度归0
    收集loss
（4）使用本轮模型生成句子，输出
（5）保存权重
"""


def build_model(pretrain_model_path, tokenizer):
    model = TorchModel(pretrain_model_path, tokenizer)
    return model


def generate_sentence(openings, model, tokenizer):
    model.eval()
    # openings = tokenizer.encode(openings) + [tokenizer.sep_token_id]
    openings = tokenizer.encode(openings, add_special_tokens=False)
    openings = [tokenizer.cls_token_id] + openings + [tokenizer.sep_token_id]
    with torch.no_grad():
        # 生成文本超过80字则终止迭代
        while len(openings) <= 80:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def create_pad_mask(len_x, len_y, max_length):
    mask = np.zeros((max_length, max_length))
    for i in range(len_x + len_y):
        for j in range(len_x + len_y):
            if i < len_x and j < len_x:
                mask[i][j] = 1
            elif i < len_x and j >= len_x:
                mask[i][j] = 0
            elif i >= len_x and j < len_x:
                mask[i][j] = 1
            elif i >= len_x and j >= len_x and i >= j:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    return mask


def build_dataset(batch_size, tokenizer, max_length, corpus):
    dataset = []
    for question, answer in corpus:
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + question_ids + [tokenizer.sep_token_id] + answer_ids + [tokenizer.sep_token_id]
        y = len(question_ids) * [-1] + [-1] + answer_ids + [tokenizer.sep_token_id] + [-1]
        # 构建掩码
        mask = create_pad_mask(len(question_ids) + 2, len(answer_ids) + 1, max_length)
        # padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [-1] * (max_length - len(y))  # 交叉熵计算时，认-1
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = torch.LongTensor(mask)
        dataset.append([x, y, mask])

    return DataLoader(dataset, batch_size, shuffle=True)


def main(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 64
    train_sample = 50000
    max_length = 120
    learning_rate = 1e-5
    corpus = load_corpus(corpus_path)

    pretrain_model_path = r"/week6_LanguageModelsAndPretraining/01_reference/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    model = build_model(pretrain_model_path, tokenizer)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data = build_dataset(batch_size, tokenizer, max_length, corpus)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_data in train_data:
            x, y, mask = batch_data
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            loss = model(x, mask, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print(f"======第{epoch + 1}轮loss平均值={np.mean(watch_loss)}")
        print(generate_sentence("如何学习语言模型？", model, tokenizer))
        print(generate_sentence("如何成为AI工程师？", model, tokenizer))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)

    return


if __name__ == '__main__':

    main("train_data.json", False)

