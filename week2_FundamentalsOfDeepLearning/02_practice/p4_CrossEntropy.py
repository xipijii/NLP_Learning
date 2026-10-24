import torch
from torch import nn
import numpy

"""
实现交叉熵
1.softmax归一化
2.将目标值进行one-hot矩阵处理
3.进行公式计算（按行处理-目标值*预测值），加和求平均
"""

# 1.torch实现交叉熵
cross_entropy = nn.CrossEntropyLoss()

# 2.设计输入和真实输出
x = torch.FloatTensor([
    [1.1, 2.3, 0.99],
    [2.2, 1.5, 0.98],
    [0.15, 1.03, 2.96]
])

target = torch.LongTensor([1, 0, 2])  # ‼️ target值必须是张量

# 3.通过交叉熵函数计算loss结果
loss = cross_entropy(x, target)
print(loss)


# 4.设计softmax函数
def diy_softmax(x):  # 逐元素操作：e^x /
    # return numpy.exp(x) / sum(numpy.exp(x))  # ‼️
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=1, keepdims=True)
    # 📒 axis=1 按行求和；keepdims 是否保持当前纬度


# 5.将目标值转化为one-hot矩阵（数组形式）
def one_hot(target, x):
    one_hot_target = numpy.zeros(x.shape)
    for i, v in enumerate(target):  # [1, 0, 2]
        #  📒 enumerate() 用于同时获取列表的索引和值
        one_hot_target[i][v] = 1
    return one_hot_target
# print(one_hot(target, x))


# 6.设计实现交叉熵函数
# 公式：- p(x) * log(q(x))
def diy_cross_entropy(x, target):  # ⚠️ 抄的 ⚠️
    batch_size, class_num = x.shape
    x = diy_softmax(x)
    one_hot_target = one_hot(target, x)
    entropy = - numpy.sum(one_hot_target * numpy.log(x), axis=1)  # 📒 axis=1 按行求和
    return sum(entropy) / batch_size


# 7.通过diy的交叉熵函数计算loss结果
diy_loss = diy_cross_entropy(x.numpy(), target.numpy())
print(diy_loss)
