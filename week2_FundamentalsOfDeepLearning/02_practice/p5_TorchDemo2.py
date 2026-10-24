import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本 ✅

作业：x是一个5维向量，判断最大数所在的纬度【np.argmax(x)能直接返回最大的纬度】
    训练数据构造、输出的是5纬度的向量、loss函数得改、预测代码
练习：用大模型、print等各种方法学习代码；人眼很容易判断的规律、让模型学习
"""
# 定义模型
class TorchModel(nn.Module):

    # def __init(self, input_size, output_size=1):
    def __init__(self, input_size, output_size=1):
        super(TorchModel, self).__init__()  # ⚠️
        self.linear = nn.Linear(input_size, output_size)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        linear_output = self.linear(x)
        y_pre = self.activation(linear_output)
        if y is not None:  # 计算损失函数
            return self.loss(y_pre, y)
        else:  # 输出预测值
            return y_pre


# 单个数据构造
def build_data():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0

# 多个数据构造
def build_dataset(total_num):
    x_dataset = []
    y_dataset = []
    for i in range(total_num):
        x, y = build_data()
        x_dataset.append(x)
        y_dataset.append([y])
    return torch.FloatTensor(x_dataset), torch.FloatTensor(y_dataset)

# 评估模型
def evaluate(model):
    correct, wrong = 0, 0
    sample_num = 100
    # 1.构造100组数据
    x_dataset, y_dataset = build_dataset(sample_num)
    # print(f"本轮评估样本数量为：{sample_num}，其中正样本数量为：{sum(y_dataset)}，负样本数量为：{sample_num - sum(y_dataset)}")
    print(f"本轮评估样本数量为：{sample_num}，其中正样本数量为：%d，负样本数量为：%d" % (sum(y_dataset), sample_num - sum(y_dataset)))
    # 2.评估模型准确率
    y_pre_dataset = model(x_dataset)
    for y, y_pre in zip(y_dataset, y_pre_dataset):
        if y_pre > 0.5 and y == 1:
            correct += 1
        elif y_pre <= 0.5 and y == 0:
            correct += 1
        else:
            wrong += 1
    acc = correct / sample_num
    print(f"准确率：{acc * 100:.2f}%")
    return acc


def main():
    # 参数设置
    epoch_num = 20
    train_sample_num = 5000
    batch_size = 20
    input_size = 5
    output_size = 1
    learning_rate = 0.005  # ‼️学习率调整小，初始正确率就低？？
    watch_loss = []
    log = []
    # 构造模型
    model = TorchModel(input_size, output_size)
    # 构造优化器
    optim = torch.optim.Adam(model.parameters(), learning_rate)
    # 构造数据（实际是读取数据）
    x_dataset, y_dataset = build_dataset(train_sample_num)
    # 开启训练
    model.train()
    for epoch in range(epoch_num):
        # # 构造数据（实际是读取数据）
        # x_dataset, y_dataset = build_dataset(train_sample_num)
        # 小批量开始预测
        for index in range(train_sample_num // batch_size):
            x = x_dataset[index * batch_size: (index + 1) * batch_size]
            y = y_dataset[index * batch_size: (index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归0
            # watch_loss.append(loss) ❌
            # watch_loss.append(loss.item)  # ❌
            watch_loss.append(loss.item())  # ✅
        # 评估本轮的模型测试效果
        acc = evaluate(model)

        print(f"第{epoch + 1}轮：loss值为：{np.mean(watch_loss)}")
        # 保存准确率和loss值
        log.append((acc, np.mean(watch_loss)))

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图 ⚠️ 抄的 ⚠️
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
