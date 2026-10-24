
import json
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import jieba
import torch

"""
定义 数据生成器
1. 获取配置、数据路径
2. 词表、定义词表大小
3. 加载标准问的idx
4. 每轮训练量的大小
5. 数据集类型
6. 加载数据
"""

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        super(DataGenerator, self).__init__()
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "word.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        max_length = self.config["max_length"]
        input_id = input_id[:max_length]
        input_id += [0] * (max_length - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type  # ❓
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()
        else:
            return self.data[index]

    # todo
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        a, n = random.sample(standard_question_index, 2)
        a_p_list = self.knwb[a]
        if len(a_p_list) < 2:
            return self.random_train_sample()
        s1, s2 = random.sample(a_p_list, 2)
        s3 = random.choice(self.knwb[n])
        return [s1, s2, s3]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict

def load_schema(schema_path):
    with open(schema_path, encoding="utf-8") as f:
        return json.loads(f.read())

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
