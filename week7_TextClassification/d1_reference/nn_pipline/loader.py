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
    3. 定义词表和词表大小
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
