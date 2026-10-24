import copy

from torch import nn
import torch
import numpy as np

"""
基于pytorch的网络编写
手动实现梯度计算和反向传播
加入激活函数
"""


# 定义一个模型，实现线性层，激活层sigmoid，和均方差损失函数
class TorchModel(nn.Module):

    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.layer = nn.Linear(input_size, input_size, bias=False)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss  # ✍️

    def forward(self, x, y=None):
        layer_output = self.layer(x)
        y_pre = self.activation(layer_output)
        print("TorchModel---y_pre", y_pre)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre


class DiyModel:

    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        layer_output = np.dot(x, self.weight.T)
        y_pre = diy_sigmoid(layer_output)
        print("DiyModel---y_pre", y_pre)
        if y is not None:
            return diy_mse_loss(y_pre, y)
        else:
            return y_pre

    # 手动实现梯度计算 todo
    # def diy_grad(self, y_pre, y_true, x):
    #     # 根据模型和loss求导数





def diy_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diy_mse_loss(y_pre, y):
    # return np.sum(np.sqrt(y_pre, y)) / len(y_pre)  # ❌
    # return np.sum(np.square(y_pre - y)) / len(y_pre)  # ✍️❌
    return np.sum(np.square(y_pre - y)) / y_pre.size  # ✍️


torch_model = TorchModel(2)
torch_weight = torch_model.state_dict()["layer.weight"]  # ✍️
print("初始化权重：", torch_weight)

diy_model = DiyModel(torch_weight)

x = np.array([[-0.5, 0.1],
              [-0.5, 0.1]])
y = np.array([[0.1, 0.2],
              [0.1, 0.2]])

torch_loss = torch_model(torch.FloatTensor(x), torch.FloatTensor(y))
diy_loss = diy_model.forward(x, y)

print(torch_loss)
print(diy_loss)

copy_weight = copy.deepcopy(torch_weight.numpy())

"""
1. 手动实现反向传播（计算梯度）

"""


"""
1. 手动实现Adam优化器
2. 手动实现梯度计算
3. 使用手动实现的优化器进行权重更新
4. 对比更新后的权重是否一致
"""

def diy_adam(weight, grad, lr=None):
    alpha = 1e-3 if lr is None else lr
    beta1 = 0.999
    beta2 = 0.9
    t = 0
    mt = 0
    vt = 0
    gt = 0
    e = 1e-8

    t += 1
    mt = beta1 * mt + (1 - beta1) * gt
    vt = beta2 * vt + (1 - beta2) * gt ** 2
    mt_ = mt / (1 - beta1 ** t)
    vt_ = vt / (1 - beta2 ** t)
    # weight = weight - alpha * mt_ / (np.sqrt(vt_ ** 2) + e)
    weight = weight - alpha * mt_ / (np.sqrt(vt_) + e)

    return weight


optimizer = torch.optim.Adam(torch_model.parameters())
torch_loss.backward()  # ‼️计算梯度
optimizer.step()  # ‼️权重更新
print("torch_adam更新后权重：", torch_model.state_dict()["layer.weight"])




# 手动实现权重更新  todo
# updated_weight = diy_adam(weight=copy_weight, grad=)
# print("diy_adam更新后权重：", updated_weight)