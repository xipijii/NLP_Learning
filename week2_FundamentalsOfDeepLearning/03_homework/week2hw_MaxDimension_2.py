import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律(作业)：x是一个5维向量，判断最大数所在的纬度【np.argmax(x)能直接返回最大的纬度】
    训练数据构造、输出的是5纬度的向量、loss函数得改、预测代码
"""


class ClassificationModel(nn.Module):

    def __init__(self, input_size=5, output_size=5):
        super(ClassificationModel, self).__init__()
        # self.linear1 = nn.Linear(input_size, 3)
        # self.linear2 = nn.Linear(3, output_size)  # ‼️结果更差
        self.linear = nn.Linear(input_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # linear1 = self.linear1(x)
        # y_pre = self.linear2(linear1)
        y_pre = self.linear(x)
        print(y_pre.shape)
        if y is not None:  # 计算loss
            return self.loss(y_pre, y)
                # y的形状为(batch_size, 1),那如何和y_pre计算loss❓
                # 内部转换成了one-hot矩阵【p4_CrossEntropy2.py的line33】
        else:  # 返回预测值（需要softmax一下将数值全部控制在0-1）
            return torch.softmax(y_pre, dim=1)  # dim=1表示将第2维进行映射变换（归一化）


def build_data():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y


def build_dataset(total_num):
    x_dataset = []
    y_dataset = []
    for i in range(total_num):
        x, y = build_data()
        x_dataset.append(x)
        y_dataset.append(y)
    return torch.FloatTensor(x_dataset), torch.LongTensor(y_dataset)


"""
1. 构建数据
2. 使用模型预测
3. 对比真实值和预测值，记录正误
‼️开启评估模式：训练模式下，张量不能直接转换为 numpy 数组，因为 numpy 不支持梯度计算。
"""


def evaluate(model):
    total_num = 100
    x_dataset, y_dataset = build_dataset(total_num)
    correct, wrong = 0, 0

    model.eval()  # ‼️
    with torch.no_grad():
        y_pre_dataset = model(x_dataset)
        for y, y_pre in zip(y_dataset, y_pre_dataset):
            # y是类别索引，y_pre是概率分布
            if int(y) == np.argmax(y_pre).item():
                correct += 1
            else:
                wrong += 1
    acc = correct / total_num
    print(f"本轮预测数据{total_num}个，其中正确预测{correct}个，错误预测{wrong}个，正确率为{acc:.2f}")
    return acc


"""
1. 参数设置
2. 构建预测模型
3. 构建优化器
4. 构造数据
5. 开始预测：数据分组；求loss；梯度计算、更新、重置
6. 评估并记录准确率
7. 画图
"""


def main():
    epoch_num = 30
    train_sample_num = 5000
    batch_size = 50
    learning_rate = 0.05
    input_size, output_size = 5, 5
    watch_loss = []
    log = []

    model = ClassificationModel(input_size, output_size)

    optim = torch.optim.Adam(model.parameters(), learning_rate)

    x_dataset, y_dataset = build_dataset(train_sample_num)

    model.train()  # ❓
    for epoch in range(epoch_num):
        # model.train()  # ❓
        for index in range(train_sample_num // batch_size):
            x = x_dataset[index * batch_size: (index + 1) * batch_size]
            y = y_dataset[index * batch_size: (index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            # watch_loss.append(loss)  # ❌Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
            watch_loss.append(loss.item())

        acc = evaluate(model)
        print(f"第{epoch + 1}轮：loss值={np.mean(watch_loss)}")
        log.append((acc, np.mean(watch_loss)))

    torch.save(model.state_dict(), "model.bin")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

"""
1. 构建数据
2. 加载已训练的模型
3. 预测，输出
"""
def predict(model_path):
    x_dataset, y_dataset = build_dataset(10)
    # model = ClassificationModel(torch.load(model_path))
    # ‼️加载模型
    model = ClassificationModel()
    model.load_state_dict(torch.load(model_path))

    y_pre_dataset = model(x_dataset)

    model.eval()
    with torch.no_grad():
        for x, y_pre in zip(x_dataset, y_pre_dataset):
            print(f"输入值为{x}，预测值为{np.argmax(y_pre)}，预测的概率分布为{y_pre}")


if __name__ == '__main__':
    main()
    # predict("model.bin")