import torch
from torch import nn
import numpy


# class TorchModel(nn.modules):  # ❌ nn.modules 是 nn 库中的一个子模块包，不是一个可以继承的类
class TorchModel(nn.Module):  # ✅ nn.Module 是 PyTorch 中所有神经网络模块都应该继承的基类

    def __init__(self, input_size=3, hidden_size1=5, hidden_size2=2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)  # 3 * 5
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)  # 5 * 2

    def forward(self, x):  # x自动转化为张量
        hidden_layer = self.layer1(x)  # n * 3 ---> n * 5
        out_layer = self.layer2(hidden_layer)  # n * 5 ---> n * 2
        return out_layer


class DiyModel:

    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    # ‼️ 不是单纯的乘积，是矩阵/张量的点乘
    # def forward(self, x):
    #     hidden_layer = self.w1 * x + self.b1
    #     out_layer = self.w2 * hidden_layer + self.b2
    #     return out_layer

    def forward(self, x):
        # hidden_layer = numpy.dot(self.w1, x) + self.b1  # ❌ A要转置的
        # y_pre = numpy.dot(self.w2, hidden_layer) + self.b2  # ❌ A要转置的
        hidden_layer = numpy.dot(x, self.w1.T) + self.b1  # 全连接层公式：y = x * A^T + b，是x在前‼️
        y_pre = numpy.dot(hidden_layer, self.w2.T) + self.b2
        return y_pre


x = numpy.array([[3.1, 1.3, 1.2],
                 [2.2, 13, 2.8]])

# 使用pytorch预测
torch_model = TorchModel(3, 5, 2)
torch_x = torch.FloatTensor(x)
model_pre_y = torch_model.forward(torch_x)

# 取出模型中的权重
print(torch_model.state_dict())
w1 = torch_model.state_dict()["layer1.weight"].numpy()  # .numpy() 将这个 PyTorch 张量转换成 NumPy 数组
b1 = torch_model.state_dict()["layer1.bias"].numpy()
w2 = torch_model.state_dict()["layer2.weight"].numpy()
b2 = torch_model.state_dict()["layer2.bias"].numpy()

# 使用权重到自己diy的模型
diy_model = DiyModel(w1, b1, w2, b2)
# diy_pre_y = diy_model.forward(torch_x)  # numpy.dot() 的参数应是 NumPy 数组‼️，不是PyTorch 张量（torch_x）
diy_pre_y = diy_model.forward(x)

print("========================================================================================================")
print(f"全连接网络模型预测结果：{model_pre_y}，模拟线性层模型预测结果：{diy_pre_y}")