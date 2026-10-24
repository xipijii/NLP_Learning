from torch.utils.data import Dataset, DataLoader
import torch
import json
import jieba
from transformers import BertTokenizer

"""
数据加载
"""

"""
实现数据加载器
__init__
1. 定义：配置、数据路径
2. 加载：字表（定义长度）、schema
3. 数据加载

load
1. 利用空行隔开句子
2. 利用空格隔开输入和label
3. 对输入进行token化，并补齐；补齐label
4. 张量化

encode_sentence
1. 如果词表路径=词表，对文本进行jieba分词后，进行token化
2. 否则，对字符进行token化
3. 补齐

padding
对序列进行补齐或阶段，使其可以在一个batch内运算

__len__
获取长度

__getitem__
获取数据

load_schema
加载计划（读取json文件）
"""


class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path

        self.use_bert = config["use_bert"]
        if self.use_bert:  # todo
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        else:
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))

                input_ids = self.encode_sentence(sentence)

                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.use_bert:  # todo
            input_id = [self.tokenizer.vocab.get(char, self.tokenizer.vocab["[UNK]"]) for char in text]
        elif self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)


"""
加载字表或词表
1. 读取词表文件
2. 每一行去空格
3. 赋值index（需要+1，0留给padding位）
"""


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict


"""
load_data
1. 用自定义的数据生成器 提取数据
2. 使用torch自带的DataLoader对数据进行batch分组和打乱
"""


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    dg = DataGenerator("../ner_data/train.txt", Config)
