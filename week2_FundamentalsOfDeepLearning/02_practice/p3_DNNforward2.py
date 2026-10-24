import torch
from torch import nn
import numpy

"""
torch
 1.构建模型（定义、组合）
 2.实例化，取权重
diy
 1.构建模型（定义-用torch模型的权重构建公式、组合）
 2.通过模型预测
"""

# pytorch全连接层
class TorchModel(nn.Module):

    def __init__(self, input_size=3, hidden_size1=5, hidden_size2=2):
        super(TorchModel, self).__init__()  # ⚠️抄的
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)

    def forward(self, x):
        hidden_layer = self.layer1(x)
        y_pre = self.layer2(hidden_layer)
        return y_pre


class DiyModel:
    def __init__(self, weight1, bias1, weight2, bias2):
        self.w1 = weight1
        self.b1 = bias1
        self.w2 = weight2
        self.b2 = bias2

    def forward(self, x):
        layer1 = numpy.dot(x, self.w1.T) + self.b1
        layer2 = numpy.dot(layer1, self.w2.T) + self.b2
        return layer2


# 构建输入
x = numpy.array([
    [1.1, 1.32, 2.4],
    [2.31, 3.12, 1.11],
    [3.14, 1.26, 4.31]
])

# torch全连接层（线性层）预测结果
torch_model = TorchModel(3, 5, 2)
torch_pre_y = torch_model.forward(torch.FloatTensor(x))

# 获取模型和参数
# w1 = torch_model.state_dict()["layer1.weight"]  # ❌现在是张量的形式
w1 = torch_model.state_dict()["layer1.weight"].numpy()  # ✅现在是数组的形式，diy模型通过数组进行计算；pytorch网络层通过张量
b1 = torch_model.state_dict()["layer1.bias"].numpy()
w2 = torch_model.state_dict()["layer2.weight"].numpy()
b2 = torch_model.state_dict()["layer2.bias"].numpy()

# 模拟全连接层（线性层）预测结果
diy_model = DiyModel(w1, b1, w2, b2)
diy_pre_y = diy_model.forward(x)

# 对比结果
print("=====================================")
print("torch model：", torch_pre_y)
print("diy model：", diy_pre_y)
