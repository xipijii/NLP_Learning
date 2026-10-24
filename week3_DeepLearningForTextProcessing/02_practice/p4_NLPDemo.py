from torch import nn
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

"""
基于pytorch的网络便携
实现一个网络完成一个简单nlp任务：
    判断文本中是否有某些特定字符(你/我/他)出现
"""


class TorchModel(nn.Module):

    # def __init__(self, vocab_size, input_size, output_size):  # ❌
    def __init__(self, vocab, sentence_length, input_size, output_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=input_size, padding_idx=0)
        # self.pooling = nn.AvgPool1d(vocab_size)  # ❌
        self.pooling = nn.AvgPool1d(sentence_length)
        self.linear = nn.Linear(input_size, output_size)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sentence_length, char_vector) -- (20, 10, 5)
        x = x.transpose(1, 2)  # (batch_size, char_vector, sentence_length) -- (20, 5, 10)
        x = self.pooling(x)  # (batch_size, char_vector, 1) -- (20, 5, 1)
        x = x.squeeze()  # ✍️ 输入网络层前 需要将池化后的最后1维去掉  -- (20, 5)
        linear_output = self.linear(x)  # (batch_size, 1) -- (20, 1)
        y_pre = self.activation(linear_output)  # (batch_size, 1) -- (20, 1)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre


# 构建单个数据
# x=[整数序列], y=标量-1为正样本 -0为负样本
# def build_data(str_seq, check_char_seq):
#     for char_seq in str_seq:
#         if char_seq in check_char_seq:
#             return str_seq, 1
#     return str_seq, 0

def build_data(sentence_length, vocabs, check_chars):
    str_seq = [random.choice(list(vocabs.keys())) for _ in range(sentence_length)]
    if set(check_chars) & set(str_seq):
        y = 1
    else:
        y = 0
    # x = [vocabs.get(char, vocabs.get("<unk>")) for char in str_seq]
    x = [vocabs[char] if char in vocabs else vocabs["<unk>"] for char in str_seq]
    return x, y


# 构建数据集
# 1. 确定检查字符的序列
# 2. 随机生成固定长度的字符串并获取其整数序列
# 3. 构造单个数据（字符串整数序列，及其标识-预测值）
# def build_dataset(total_num, sentence_length, char_dim, vocabs, check_chars):
#     check_char_seq = [vocabs.get(char, vocabs.get("<unk>")) for char in check_chars]
#     x_dataset = []
#     y_dataset = []
#     for i in range(total_num):
#         # str_seq = [random.randint(1, 99) for _ in range(sentence_length)]
#         str_seq = [random.randint(1, len(vocabs) - 1) for _ in range(sentence_length)]
#         x, y = build_data(str_seq, check_char_seq)
#         x_dataset.append(x)
#         y_dataset.append([y])
#     # return torch.LongTensor(x_dataset), torch.LongTensor(y_dataset)  # ❌
#     return torch.LongTensor(x_dataset), torch.FloatTensor(y_dataset)

def build_dataset(total_num, sentence_length, vocabs, check_chars):
    x_dataset = []
    y_dataset = []
    for i in range(total_num):
        x, y = build_data(sentence_length, vocabs, check_chars)
        x_dataset.append(x)
        y_dataset.append([y])
    return torch.LongTensor(x_dataset), torch.FloatTensor(y_dataset)


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

# 评估方法
def evaluate(model, vocabs, check_chars):
    model.eval()
    test_sample_num = 200
    sentence_length = 10
    correct, wrong = 0, 0
    x_dataset, y_dataset = build_dataset(test_sample_num, sentence_length, vocabs, check_chars)
    y_pre_dataset = model(x_dataset)

    for y, y_pre in zip(y_dataset, y_pre_dataset):
        if float(y_pre) >= 0.5 and int(y) == 1:
            correct += 1
        elif float(y_pre) < 0.5 and int(y) == 0:  # ✍️ 漏了
            correct += 1
        else:
            wrong += 1
    acc = correct / test_sample_num
    print(f"共检测样本{test_sample_num}个，正确数量{correct}个，错误数量{wrong}个，正确率为{acc * 100:.2f}%")
    percent = sum(y_dataset) / test_sample_num * 100
    print(f"正样本比例：{percent.item():.2f}%")  # ‼️✍️
    return correct / test_sample_num

# 模型训练
# 1. 参数设置
# 2. 构造模型、优化器
# 3. 构造训练集
# 4. 开启训练（模式）- 循环固定量batch_size的样本
#   a. 获取batch_size的样本
#   b. 获取loss值（并收集）
#   c. 梯度计算、权重更新、梯度归0
# 5. 评估模型
# 6. 保存模型、打印准确率和loss曲线
def main():
    epoch_num = 20
    train_sample_size = 1000
    batch_size = 20
    vocab = build_vocab()
    sentence_length = 10
    check_chars = ["你", "我", "他"]
    char_dim = 20
    learning_rate = 0.005
    watch_loss = []
    log = []

    model = TorchModel(vocab, sentence_length, char_dim, 1)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_dataset, y_dataset = build_dataset(train_sample_size, sentence_length, vocabs=vocab, check_chars=check_chars)

    model.train()
    for epoch in range(epoch_num):

        for index in range(train_sample_size // batch_size):
            x_batch = x_dataset[index * batch_size: (index + 1) * batch_size]
            y_batch = y_dataset[index * batch_size: (index + 1) * batch_size]

            loss = model(x_batch, y_batch)

            loss.backward()
            optim.step()
            optim.zero_grad()

            watch_loss.append(loss.item())

        print(f"第{epoch + 1}轮训练，平均loss值为{np.mean(watch_loss)}")

        acc = evaluate(model, vocab, check_chars)

        log.append([acc, np.mean(watch_loss)])

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

"""
1. 加载模型
2. 构造训练数据及其整数序列
3. 使用模型预测
4. 输出结果
"""
def predict(model_path, str_list):
    # ✍️先构造模型，后加载训练好的权重
    vocab = build_vocab()
    sentence_length = 10
    input_size = 20
    # ✍️
    model = TorchModel(vocab, sentence_length, input_size, 1)
    model.load_state_dict(torch.load(model_path))

    seq_list = []
    for str in str_list:
        # seq_list.append([vocab.get(char, vocab.get("<unk>")) for char in str])
        seq_list.append([vocab.get(char, vocab["<unk>"]) for char in str])

    x_set = np.array(seq_list)
    y_pre_set = model(torch.LongTensor(x_set))

    for x, y_pre in zip(str_list, y_pre_set):
        print(f"str={x}，预测值={y_pre}")



if __name__ == '__main__':
    main()

    # # 预测：判断文本中是否有某些特定字符(你/我/他)出现
    # # str_list = ["今天我替中国欢迎你1", "快快乐乐每一天123", "ggaabbcc他来"]
    # vocab = build_vocab()
    # str_list = []
    # for i in range(10):
    #     random_str = "".join([random.choice(list(vocab.keys())) for _ in range(10)])
    #     str_list.append(random_str)
    # predict("model.bin", str_list)
    #

