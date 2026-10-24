import json

import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

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
3. title转换为对应的input_ids（Bert用自己的词表）(要补齐‼️)
"""


class DataGenerator(Dataset):

    def __init__(self, data_path, config):
        super(DataGenerator, self).__init__()
        self.config = config
        self.data_path = data_path
        self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())  # ‼️dict1.items() 获取所有的key-value键值对
        self.config["class_num"] = len(self.label_to_index)
        if config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []

        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)  # ‼️
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    # self.tokenizer.encode(title, )  # ❌参数 ➕最大长度，是否pad到最大长度
                    input_ids = self.tokenizer.encode(title, max_length=self.config["max_length"]
                                                      , pad_to_max_length=True)
                else:
                    input_ids = self.sentence_to_ids(title)
                input_ids = self.padding(input_ids)  # ‼️
                x = torch.LongTensor(input_ids)
                y = torch.LongTensor([label])
                self.data.append([x, y])
        return

    def sentence_to_ids(self, sentence):
        input_ids = [self.vocab.get(char, self.vocab.get("[UNK]")) for char in sentence]
        return self.padding(input_ids)

    def padding(self, input_ids):
        max_len = self.config["max_length"]
        input_ids = input_ids[:max_len]
        input_ids += [0] * (max_len - len(input_ids))
        return input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(path):
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            word = line.strip()  # ‼️用于去除字符串两端的空白字符：空格、换行符、制表符、回车符
            vocab[word] = index + 1  # ✍️给padding留位
    return vocab


def load_data(data_path, config):
    data_generator = DataGenerator(data_path, config)
    data_loader = DataLoader(data_generator, batch_size=config["batch_size"], shuffle=True)
    return data_loader
