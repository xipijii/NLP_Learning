import torch
from torch import nn
import numpy as np

cross_entropy = nn.CrossEntropyLoss()
x = torch.FloatTensor([
    [1.1, 2.3, 0.98],
    [2.1, 3.2, 1.57],
    [0.79, 2.32, 4.15]
])
y_true = torch.LongTensor([1, 1, 2])
torch_loss = cross_entropy(x, y_true)
print(torch_loss)


# 逐行归一化
def diy_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


# 目标值转为矩阵
def to_one_hot(y_true, shape):
    target = np.zeros(shape.shape)
    for i, v in enumerate(y_true):
        target[i][v] = 1
    return target


# 求交叉熵函数
def diy_cross_entropy(x, y_true):
    batch_size, class_num = x.shape
    x = diy_softmax(x)  # ‼️进行交叉熵计算loss之前，需要进行softmax进行整行归一化‼️
    target = to_one_hot(y_true, x)
    entropy = - np.sum(target * np.log(x), axis=1)
    return sum(entropy) / batch_size


print(diy_cross_entropy(x.numpy(), y_true))
