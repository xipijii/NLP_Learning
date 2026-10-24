import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import random
"""  DataGenerator
__init__
1. 接收参数变量
2. 定义词表
    Bert使用tokenizer
    其余使用自定义方法加载
3. 定义词表大小，分类大小
4. 加载数据

load()
（0表负样本，1表正样本）
读取行数据：
    label
    sentence转成id（Bert与非Bert）
    收集

encode_sentence

padding

__len__

__getitem__
"""
""" split_data

"""
""" load_data
1. 使用generator实现dataset接口
2. 使用dataloader进行批量处理和打乱，提升并行效率
"""
""" load_vocab
留一个padding位
"""

class DataGenerator(Dataset):

    def __init__(self, path, config):
        super(DataGenerator, self).__init__()
        self.path = path
        self.config = config

        if config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)  # ‼️ 手动更改字典的索引值，会出现索引不连续的情况
        config["class_num"] = 2
        config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        # x_dataset, y_dataset = [], []  # ❌
        self.data = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                label = int(line[0])
                sentence = line[2:].strip()
                # ‼️Bert要是用tokenizer
                if self.config["model_type"] == "bert":
                    # input_ids = self.tokenizer(sentence
                    #                            , max_length=self.config["max_length"]
                    #                            # , padding=self.config["max_length"])
                    #                            , padding="max_length")  # ❌是要encode方法
                    input_ids = self.tokenizer.encode(sentence
                                               , max_length=self.config["max_length"]
                                               # , padding=self.config["max_length"])
                                               , padding="max_length"
                                               , truncation=True)  # ✅truncation=True超过指定长度截断
                else:
                    input_ids = self.encode_sentence(sentence)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor([label])])  # ‼️

    def encode_sentence(self, sentence):
        input_ids = [self.vocab.get(char, self.vocab.get("[UNK]")) for char in sentence]
        input_ids = self.padding(input_ids)
        return input_ids

    def padding(self, input_ids):
        max_len = self.config["max_length"]
        input_ids = input_ids[:max_len]
        input_ids += [0] * (max_len - len(input_ids))
        return input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 读取文件，随机打乱，80%存入训练集文件，20%存入验证集文件
def split_dataset(path, config):
    original_data = []
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == 0:  # 跳过标题行
                continue
            original_data.append(line.strip())
    random.shuffle(original_data)  # ‼️
    # 数据分割
        # 计算分割点（第几条数据开始做切分，使用列表功能直接切割）
    split_index = int(0.8 * (len(original_data)))
    train_data = original_data[: split_index]
    valid_data = original_data[split_index:]
    # 文件输出
    with open(config["train_data_path"], 'w', encoding="utf-8") as f:
        for line in train_data:
            f.write(line + '\n')
    with open(config["valid_data_path"], 'w', encoding="utf-8") as f:
        for line in valid_data:
            f.write(line + '\n')
    return

def load_data(path, config):
    dg = DataGenerator(path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=True)
    return dl

def load_vocab(path):
    # vocab = {"[PAD]": 0}  # ❌embedding自动加了
    vocab = {}  # ‼️embedding自动加pad了
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            word = line.strip()
            if word:  # ‼️
                vocab[word] = index + 1
    # vocab["[UNK]"] = len(vocab)  # ❌词表中有
            # ‼️索引不连续通常就是因为手动添加特殊 token（如 PAD、UNK）导致
            # ✍️这样的话，词表长度需要根据最大索引+1来（不然embeeding时会出现索引越界的情况）
    return vocab
