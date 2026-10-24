import torch
from torch import nn
from torch.utils.data import DataLoader  # ‼️
import jieba
import numpy as np
import matplotlib.pyplot as plt

"""
使用torch训练一个分词模型
1. 网络模型结构
    embedding, rnn, linear, crossEntropy
2. 加载字表
3. 构造训练数据（构建Dataset类）
    init：
        
    load()：
        实现x 字符串转整数序列
        实现y 字符串转label（jieba分词后，对字符串进行打标处理，标识每个字符是否是一个词的边界）
        根据最大长度补齐
        加入data属性（格式：[sequence， label]
    len:
    item:
"""


class TorchRNNModel(nn.Module):

    def __init__(self, vocab, input_size, hidden_size, output_size):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=input_size,
                                      padding_idx=0,  # ✍️补0的角标
                                      )
        self.rnn = nn.RNN(input_size=input_size,  # ✍️输入每个字的纬度大小
                          hidden_size=hidden_size,  # ✍️隐藏层大小
                          batch_first=True,  # ✍️为了匹配embedding
                          )
        # self.linear = nn.Linear(input_size, output_size)
        self.linear = nn.Linear(hidden_size, output_size)  # ⚠️
        self.activation = nn.Softmax(dim=-1)  # ✍️获取哪个分类中的概率值最大时，不用加也行，因为不改变数据的大小关系
        # self.loss = nn.CrossEntropyLoss
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)  # ‼️

    def forward(self, x, y=None):
        x = self.embedding(x)
        # ❌(len(vocab), input_size)
        # ✍️(batch_size, max_length, input_size)
        output, hidden = self.rnn(x)
        # (batch_size, max_length, hidden_size) ❓✅

        y_pre = self.linear(output)  # (batch_size, max_length, output_size) ❓
        if y is not None:  # y (batch_size, max_length, 1)
            # return self.loss(y_pre, y)
            return self.loss(y_pre.view(-1, 2), y.view(-1))
        else:
            return self.activation(y_pre)


# def build_vocab(path):
#     vocab = ["<pad>"]  # ❌vocab应该是个字典
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             vocab.append(line.split()[0])
#     vocab.append("<unk>")
#     return vocab

def build_vocab(path):
    vocab = {"<pad>": 0}
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            vocab[line.split()[0]] = index + 1
    vocab["<unk>"] = len(vocab)
    return vocab


def build_dataset(batch_size, sentence_max_length, corpus_path, vocab):
    dataset = Dataset(sentence_max_length, corpus_path, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class Dataset:
    def __init__(self, sentence_max_length, corpus_path, vocab):
        # ✍️corpus_path, vocab
        self.sentence_max_length = sentence_max_length
        self.corpus_path = corpus_path
        self.vocab = vocab
        # self.data = []  # ❌
        self.load()

    """
    ✍️循环语料 ‼️
        实现x 字符串转整数序列
        实现y 字符串转label（jieba分词后，对字符串进行打标处理，标识每个字符是否是一个词的边界）
        根据最大长度补齐
        加入data属性（格式：[sequence， label]
    """

    def load(self):
        self.data = []
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sentence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])

    # ✍️补齐 ‼️或固定到特定长度
    def padding(self, sequence, label):  # ❓为什么这个方法写在类里？
        sequence = sequence[:self.sentence_max_length]
        sequence += [0] * (self.sentence_max_length - len(sequence))
        label = label[:self.sentence_max_length]
        label += [-100] * (self.sentence_max_length - len(label))
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab.get("<unk>")) for char in sentence]
    return sequence


def sentence_to_label(sentence):
    words = jieba.cut(sentence)
    # label = [0] * (len(sentence) - 1)  # ❌
    label = [0] * len(sentence)
    pointer = 0  # ‼️指针用法
    for word in words:
        pointer += len(word)  # ‼️
        # label[sentence.find(word) + len(word)] = 1  # ❌
        # label[pointer] = 1  # ❌
        label[pointer - 1] = 1
    return label


def evaluate(model):
    pass


def main():
    epoch_num = 20
    # train_sample_size = 5000  # DataLoader无需使用
    batch_size = 20
    corpus_path = "corpus.txt"
    vocab = build_vocab("chars.txt")
    sentence_max_length = 18
    input_size = 50  # 256
    hidden_size = 100  # 128
    output_size = 2
    learning_rate = 0.01  # 为什么0.01会越训越高❓  ✅1e-3
    watch_loss = []
    log = []

    model = TorchRNNModel(vocab, input_size, hidden_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # x_dataset, y_dataset = build_dataset(batch_size, sentence_max_length, corpus_path, vocab) # ❌
    dataloader = build_dataset(batch_size, sentence_max_length, corpus_path, vocab)
    # ‼️正确使用方法 ⬆️

    model.train()
    for epoch in range(epoch_num):
        # ❌ ✍️不是这么用的
        # for index in range(train_sample_size // batch_size):
        #     x_dataset, y_dataset = build_dataset(batch_size, sentence_max_length, corpus_path, vocab)
        for x, y in dataloader:
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        # acc = evaluate(model)
        log.append(np.mean(watch_loss))
        print(f"第{epoch + 1}轮，loss平均值为{np.mean(watch_loss)}")

    # ===============================✍️ 下面是抄的==============================
    # 保存模型
    torch.save(model.state_dict(), "model.bin")  # 将训练好的模型保存下来
    # 画图
    print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(test_list, model_path):
    vocab = build_vocab("chars.txt")
    sentence_max_length = 18
    input_size = 50  # 256
    hidden_size = 100  # 128
    output_size = 2
    model = TorchRNNModel(vocab, input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))

    test_seq_list = []
    for test_str in test_list:
        test_seq = [vocab.get(word, vocab.get("<unk>")) for word in test_str]
        test_seq = test_seq[:sentence_max_length]
        test_seq += [0] * (sentence_max_length - len(test_seq))
        test_seq_list.append(test_seq)
    y_pre_tensor = model(torch.LongTensor(test_seq_list))
    with torch.no_grad():
        for sentence, y_pre in zip(test_list, y_pre_tensor):
            new_str = ""
            for char, label in zip(sentence, y_pre):
                new_str += char
                if np.argmax(label) == 1:
                    new_str += " | "
            print(new_str)


if __name__ == '__main__':
    main()
    # input_strings = ["同时，国内有望出台，新汽车刺激方案",
    #                  "沪胶后市有望延续强势！",
    #                  "经过两个交易日的强势调整后",
    #                  "昨日上海天然橡胶期货价格再度大幅上扬"]
    # predict(input_strings, "model.bin")
