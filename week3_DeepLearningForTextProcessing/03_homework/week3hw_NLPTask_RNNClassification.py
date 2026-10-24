from torch import nn
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

"""
给定一个字符串，输出字符"a"在字符串中的位置
注意：
 1.分类任务
 2.不能用pooling了（用rnn）
思路：
 1.模型：embedding, rnn(pooling不考虑语序), linear(输入为rnn输出纬度, 输出为字符串长度), softmax, crossEntropy
 2.词表构建
 3.数据构造：随机安排10个字符的字符串，随机生成数字0-10（如果10就不插入，否则插入对应位置）
"""

"""
老师讲解
1. 当字符不存在时，如何设计？ 线性层多加一个纬度表示不存在的情况；数据构造部分返回字符串的长度
2. rnn的输出，推荐使用完整的信息取出最后一维（rnn的变体-如LSTM 会发现是不一样的）
3. python方法：随机采样；字符串中特定字符的位置.index()
4. 将词表写入json文件
5. 为何这里的正确率是逐步上升的？
"""


class TorchRNNModel(nn.Module):

    def __init__(self, vocab, char_dim, input_size, hidden_size, output_size):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=input_size, padding_idx=0)  # ✍️embedding_dim
        # self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            # ✍️ batch_first在传入张量时，第一纬代表batch_size；这样设置就能跟embedding层连上
        self.linear = nn.Linear(hidden_size, output_size)  # ✍️
        self.activation = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)  # (20, 10, 256)
        _, x = self.rnn(x)  # ✍️第一个返回值包含所有步的信息，第二个返回值包含最终整体信息    (1, 20, 128)
        x = x.squeeze()
        y_pre = self.linear(x)  # ✍️
        # y_pre = self.activation(x) # ❌交叉熵函数内自动进行了softmax
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return self.activation(y_pre)

def build_vocab():
    # 1. 创建词表 第一个位置：占位符（padding token）
    vocab = ["<pad>"]

    # 2. 添加"你/我/他"
    vocab.extend(["你", "我", "他"])

    # 3. 添加其他常用中文字符（约 50 个）
    chinese_chars = [
        "的", "了", "在", "是", "和", "有", "这", "个", "上", "中",
        "为", "人", "们", "以", "到", "就", "说", "要", "大", "小",
        "多", "好", "很", "都", "把", "着", "看", "想", "能", "可",
        "去", "来", "出", "进", "过", "还", "也", "但", "会", "自",
        "从", "而", "之", "用", "所", "使", "并", "如", "因", "哈"
    ]
    vocab.extend(chinese_chars)

    # 4. 添加英文字母（大小写）
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    vocab.extend(letters)  # 26 个小写字母

    # 5. 添加数字
    digits = [str(i) for i in range(10)]
    vocab.extend(digits)

    # 6. 补充一些标点符号
    punctuations = ["，", "。", "？", "！", "；", "：", "?", "（", "）", "~"]

    vocab.extend(punctuations)

    # 7. 确保总长度为 100，最后一个位置留给未知符 <unk>
    # 如果超过 100，截断；如果不足 100，补充一些常见字符
    while len(vocab) < 99:  # 留出 2 个位置给 pad 和 unk
        vocab.append(f"char_{len(vocab)}")

    # 截断到 99 个
    vocab = vocab[:99]

    # 8. 最后一个位置：未知符（unknown token）
    vocab.append("<unk>")

    # 9. 转换为字典：键为字符，值为索引位置
    vocab_dict = {char: idx for idx, char in enumerate(vocab)}

    return vocab_dict

def build_data(vocab, sentence_length, check_char):
    sentence = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    ran = random.randint(0, 9)
    # if ran != 10:  # ❌
    sentence = sentence[:ran] + list(check_char) + sentence[ran + 1:]
    x = [vocab.get(char, vocab.get("<unk>")) for char in sentence]
    y = ran if x.count(vocab.get(check_char)) > 0 else -1
    # y = ran if x.count(ran) > 0 and ran != 10 else -1
    if y == -1:
        print(sentence)
    return x, y

def build_dataset(total_num, vocab, sentence_length, check_char):
    x_dataset = []
    y_dataset = []
    for _ in range(total_num):
        x, y = build_data(vocab, sentence_length, check_char)
        x_dataset.append(x)
        # y_dataset.append([y])
        y_dataset.append(y)  # ‼️交叉熵函数 不需要张量处理
    return torch.LongTensor(x_dataset), torch.LongTensor(y_dataset)


def evaluate(model, vocab, sentence_length, check_char):
    model.eval()
    total_num = 1000
    x_dataset, y_dataset = build_dataset(total_num, vocab, sentence_length, check_char)
    y_pre_dataset = model(x_dataset)
    correct, wrong = 0, 0
    with torch.no_grad():  # ‼️
        for y, y_pre in zip(y_dataset, y_pre_dataset):
            # if float(y_pre) >= 0.5 and int(y) == 1:
            #     correct += 1
            # elif float(y_pre) < 0.5 and int(y) == 0:
            #     correct += 1
            # else:
            #     wrong += 1
            if np.argmax(y_pre).item() == int(y):
                correct += 1
            else:
                wrong += 1
    acc = correct / total_num
    print(f"共检测样本{total_num}个，正样本数量为{correct}，负样本数量为{wrong}，正确率为{acc}")
    return acc


def main():
    epoch_num = 20
    train_sample_num = 1000  # 2000
    batch_size = 40  # 20
    learning_rate = 0.001  # 0.005
    vocab = build_vocab()
    check_char = "我"
    char_dim = 30  # 20🆚老师的 30
    rnn_input_size = 128
    hidden_size = 256
    sentence_length = 10
    watch_loss = []
    log = []

    # model = TorchRNNModel(vocab, char_dim, rnn_input_size, hidden_size, sentence_length)
    model = TorchRNNModel(vocab, char_dim, char_dim, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), learning_rate)

    x_dataset, y_dataset = build_dataset(train_sample_num, vocab, sentence_length, check_char)

    model.train()
    for epoch in range(epoch_num):
        for index in range(train_sample_num // batch_size):
            x = x_dataset[index * batch_size: (index + 1) * batch_size]
            y = y_dataset[index * batch_size: (index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        acc = evaluate(model, vocab, sentence_length, check_char)
        log.append([acc, np.mean(watch_loss)])
        print(f"第{epoch + 1}轮，loss平均值为{np.mean(watch_loss)}，准确率为{acc}")

    # ===============================✍️ 下面是抄的==============================
    # 保存模型
    torch.save(model.state_dict(), "model.bin")  # 将训练好的模型保存下来
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predict(model_path, pre_num):
    vocab = build_vocab()
    sentence_length = 10
    check_char = "我"
    sentence_list = []
    seq_list = []
    # 随机字符串
    for i in range(pre_num):
        sentence = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
        ran = random.randint(0, 10)
        if ran != 10:
            sentence = sentence[:ran] + list(check_char) + sentence[ran + 1:]
        sentence_list.append(sentence)
        seq_list.append([vocab.get(char, vocab.get("<unk>")) for char in sentence])
    # 加载模型来得出预测结果
    model = TorchRNNModel(vocab, sentence_length, 256, 128, 10)
    model.load_state_dict(torch.load(model_path))
    y_pre_list = model(torch.LongTensor(seq_list))
    # 输出结果
    with torch.no_grad():
        for x, y_pre in zip(sentence_list, y_pre_list):
            print(f"字符串为：【{"".join([char for char in x])}】，预测值为：{y_pre}，包含\"我\"这个字符的位置：{np.argmax(y_pre).item()}")

if __name__ == '__main__':
    main()
    # predict("model.bin", 10)