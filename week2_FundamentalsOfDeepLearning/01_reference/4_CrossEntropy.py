import torch
import torch.nn as nn
import numpy as np

'''
手动实现交叉熵的计算
torch
1.调用torch的交叉熵
2.设计矩阵（3个样本做3分类）
3.设计真实分类结果（3个样本的正确类别）
4.计算交叉熵的loss值（这里nn.CrossEntropyLoss()自动实现softmax并计算）
diy
1.实现softmax激活函数（将结果逐行归一化）
2.将真实分类结果转化为one-hot矩阵
3.实现交叉熵函数（手动实现softmax，再计算）
'''

# 使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()
# 假设有3个样本，每个都在做3分类
pred = torch.FloatTensor([[0.3, 2.1, 0.3],  # 预测值 转变成张量 n*class_num
                          [0.19, 1.2, 0.9],
                          [0.5, 0.4, 0.2]])  # n*class_num
# 正确的类别分别为1,2,0
target = torch.LongTensor([1, 2, 0])  # n
'''
[0,1,0]
[0,0,1]   
[1,0,0]
'''
loss = ce_loss(pred, target)  # ✍️ 这里先对预测值做softMax，然后进行计算（对位相乘、相加、取负）
print(loss, "torch输出交叉熵")


# 实现softmax函数
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)


# 验证softmax函数
# print(torch.softmax(pred, dim=1))
# print(softmax(pred.numpy()))


# 将输入转化为onehot矩阵
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):  # 用于同时获取列表的索引和值。
        one_hot_target[i][t] = 1
    return one_hot_target


# 手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape  # 获取数量和分类类别数量
    pred = softmax(pred)  # softmax归一化
    target = to_one_hot(target, pred.shape)  # 转化成one-hot矩阵
    entropy = - np.sum(target * np.log(pred), axis=1)  # 按公式求解
    return sum(entropy) / batch_size


print(cross_entropy(pred.numpy(), target.numpy()), "手动实现交叉熵")

# print(np.log(2.7))
