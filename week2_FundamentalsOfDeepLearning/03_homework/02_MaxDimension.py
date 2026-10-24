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

class TorchModel(nn.Module):

    def __init__(self, input_size=5, output_size=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # self.activation = nn.Softmax(dim=1)  # 按行计算 交叉熵函数自带softmax
        self.loss = nn.CrossEntropyLoss()  # ⚠️这是一个类

    def forward(self, x, y=None):
        # linear_output = self.linear(x)
        # y_pre = self.activation(linear_output)
        y_pre = self.linear(x)  # ‼️  -> (batch_size, 5)
        # print(y_pre.shape)
        if y is None:  # 返回预测值
            # return y_pre  # ‼️这里需要做一下softmax能保证加和为1
            return torch.softmax(y_pre, dim=-1)  # ‼️这里做一下softmax能保证加和为1，axis=-1代表对输入的最后一纬做归一化
        else:  # 求loss值
            # return self.loss(y, y_pre) ❌
            return self.loss(y_pre, y)


def build_data():
    x = np.random.random(5)
    y = np.argmax(x)
    # ❌‼️不需要保证真实值也是一个概率分布吗，如[0,0,1,0,0]
    # y = np.zeros(5)
    # index = np.argmax(x)
    # y[index] = 1
    return x, y

def build_dataset(total_num):
    x_dataset = []
    y_dataset = []
    for i in range(total_num):
        x, y = build_data()
        x_dataset.append(x)
        # y_dataset.append([y]) ❌
        y_dataset.append(y)
    # return x_dataset, y_dataset ❌
    return torch.FloatTensor(x_dataset), torch.LongTensor(y_dataset)  # ‼️

"""
1. 构造验证集数据
2. 使用模型进行验证集的预测
3. 计算准确率
4. 输出
"""
def evaluate(model):
    model.eval()  # ‼️
    total_num = 100
    correct, wrong = 0, 0
    x_dataset, y_dataset = build_dataset(total_num)
    with torch.no_grad():  # ‼️
        y_pre_dataset = model(x_dataset)
        for y, y_pre_tensor in zip(y_dataset, y_pre_dataset):
            # y = np.argmax(y_tensor)  ‼️返回的是'numpy.ndarray'类型
            y_pre = np.argmax(y_pre_tensor).item()  # ‼️返回的是整数类型
            if y == y_pre:
                correct += 1
            else:
                wrong += 1
    acc = correct / total_num
    print(f"共包含样本数{total_num}，正确预测{correct}个，错误预测{wrong}个，准确率为{acc:.2f}")
    return acc


def main():
    # 初始化参数
    epoch_num = 25
    sample_num = 5000
    batch_size = 20
    learning_rate = 0.01
    input_size = 5
    output_size = 5
    watch_loss = []
    log = []

    # 预测模型
    model = TorchModel(input_size, output_size)

    # 定义优化器
    optim = torch.optim.Adam(model.parameters(), learning_rate)

    # 构造训练数据❓
    x_dataset, y_dataset = build_dataset(sample_num)

    # 开启模型训练
    for epoch in range(epoch_num):
        """
        1. 开启训练模式
        2. 根据batch_size分组训练数据
        3. 模型预测
        4. 求loss值（添加
        5. 梯度计算、更新、重置
        6. 模型预测
        """
        model.train()
        # for index in range(int(sample_num/batch_size)):
        for index in range(sample_num//batch_size):
            # x_batch = x_dataset[index * batch_size: index * (batch_size + 1)] ❌
            x_batch = x_dataset[index * batch_size: (index + 1) * batch_size]
            # y_batch = y_dataset[index * batch_size: index * (batch_size + 1)] ❌
            y_batch = y_dataset[index * batch_size: (index + 1) * batch_size]
            # for x, y in zip(x_batch, y_batch):
            loss = model(x_batch, y_batch)  # python 的语法糖 =model.forward(x_batch, y_batch)
            watch_loss.append(loss.item())
            loss.backward()
            optim.step()
            optim.zero_grad()
        acc = evaluate(model)
        print(f"第{epoch + 1}轮：loss值为{np.mean(watch_loss)}")
        log.append((acc, np.mean(watch_loss)))

    torch.save(model.state_dict(), "model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

"""
1. 构建预测数据
2. 加载模型
3. 使用模型预测
4. 输出预测结果
"""
def predict(model_path):
    x_dataset, y_dataset = build_dataset(10)
    # model = torch.load(model_path)
    # y_pre_dataset = model(torch.FloatTensor(x_dataset))  ❌
    model = TorchModel(5)  # ‼️
    model.load_state_dict(torch.load(model_path))  # ‼️
    y_pre_dataset = model(x_dataset)
    model.eval()  # ‼️
    for x, y_pre in zip(x_dataset, y_pre_dataset):
        print(f"x为：{x}，y的预测值为：{y_pre}")


if __name__ == '__main__':
    main()
    # predict("model.bin")