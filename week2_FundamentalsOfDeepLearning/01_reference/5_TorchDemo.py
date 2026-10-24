# coding:utf8

# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
作业：x是一个5维向量，判断最大数所在的纬度【np.argmax(x)能直接返回最大的纬度】
    训练数据构造、输出的是5纬度的向量、loss函数得改、预测代码
练习：用大模型、print等各种方法学习代码；人眼很容易判断的规律、让模型学习
"""


# 👀 参考2_GradientDescent.py

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层
        # ✍️ 输入是5纬向量，输出是正样本/负样本（可以设置为大于0.5是正类，小于0.5是负类）
        # 👀 参考2_GradientDescent.py，这里搭建了模型、参数初始化（看输出
        self.activation = torch.sigmoid  # nn.Sigmoid() sigmoid归一化函数
        # ✍️ 输出的值处于（0，1）之间
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        # ✍️ 交叉熵适用于多分类任务的场景，这里使用均方差 适用于使用1个数字来代表就可以了（接近于1是正类）

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:  # ‼️如果输入y，计算loss值，否则输出预测结果
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率 ‼️
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():  # ✍️ 即使没有调用计算梯度的方法，也会记录一些中间值，加了no_grad就不会记录了，减负
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  # 👀pytorch
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            # 👀t梯度计算那三行
            optim.step()  # 更新权重 # 👀
            optim.zero_grad()  # 梯度归零 # 👀
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")  # ✍️ 将训练好的模型保存下来
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重 ‼️
    print("==========训练好的权重==========")
    print(model.state_dict())
    # ✍️为什么总能做对❓权重A与输入对位相乘，再+b，得到的结果与真实值吻合。
    # 但是实际上两个数比较接近时，可能就会出错 [0.08889086,0.15229675,0.31082123,0.03504317,0.88920843]
    # 可以调整参数重新练练，看看效果能不能更好
    # 这个任务的完美模型样子：权重A[1,0,0,0,-1] b[0]
    # 这样的话，如果第1个数>第5个数，线性层输出>0，过完激活层sigmoid（看图像）的输出就>0.5；否则，<0.5
    print("==============================")

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)

    # ✍️ 思考：思考改动拿到的结果是否合理？【简单的任务能帮助理解深度学习的基本原理】
    #   比如：怎样做才能让准确率在第一轮就很高❓第一轮训练拉长。。。train_sample变大=5000
    # 比如还是5000轮，我怎样让准确率曲线拉得更长一点❓ 将学习率变低（
