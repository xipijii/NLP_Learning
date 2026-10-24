#coding:utf8

import torch
import torch.nn as nn
import numpy as np

"""
numpy手动实现模拟一个线性层
"""


#搭建一个2层的神经网络模型
#每层都是线性层
class TorchModel(nn.Module):
    """
    nn.Module
    神经网络基类：nn.Module 是 PyTorch 中所有神经网络模块的基类，继承它才能让 TorchModule 成为一个真正的 PyTorch 模型
    自动参数管理：当你在类中定义网络层（如 nn.Linear）时，nn.Module 会自动注册这些可学习参数，让你可以通过 model.parameters() 或 model.state_dict() 访问它们
    """
    def __init__(self, input_size=3, hidden_size1=5, hidden_size2=2):
        super(TorchModel, self).__init__()  # ❓
        """
        super(TorchModel, self)：获取 TorchModel 的父类（即 nn.Module）的引用
        .__init__()：调用父类的构造方法 
        """
        # 第一件事，定义要使用哪些网络层（组件）
        self.layer1 = nn.Linear(input_size, hidden_size1)  # w：3 * 5
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)  # 5 * 2

    def forward(self, x):  # ✍️ 默认x有两个纬度（矩阵）, n * 3 (n=batch_size)
        # 第二件事：定义这些网络层是如何组合的
        x = self.layer1(x)  # shape: (batch_size, input_size) -> (batch_size, hidden_size1)
        # y_pred = self.sig(x)  # shape: (batch_size, hidden_size1) -> (batch_size, hidden_size1)
        y_pred = self.layer2(x)  # shape: (batch_size, hidden_size1) -> (batch_size, hidden_size2)
        return y_pred


# 自定义模型
class DiyModel:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        # x: 2 * 3, w1.T: 3 * 5   = 2*5  b1: 1 * 5
        hidden = np.dot(x, self.w1.T) + self.b1  # 1*5
        y_pred = np.dot(hidden, self.w2.T) + self.b2  # 5*2
        return y_pred
    # 📒np.dot(a, b)


#随便准备一个网络输入
x = np.array([[3.1, 1.3, 1.2],
              [2.1, 1.3, 13]])
# 📒np.array()

#建立torch模型
torch_model = TorchModel(3, 5, 2)

print(torch_model.state_dict())

print("-----------")
#打印模型权重，权重为随机初始化
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_b1 = torch_model.state_dict()["layer1.bias"].numpy()
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_b2 = torch_model.state_dict()["layer2.bias"].numpy()
# 📒torch_model.state_dict()["layer1.weight"].numpy()  【.numpy()】
print(torch_model_w1, "torch w1 权重")
print(torch_model_b1, "torch b1 权重")
print("-----------")
print(torch_model_w2, "torch w2 权重")
print(torch_model_b2, "torch b2 权重")
print("-----------")
# 使用torch模型做预测
torch_x = torch.FloatTensor(x)
# 📒 将x转成torch的张量
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred)

# #把torch模型权重拿过来自己实现计算过程
diy_model = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
# #用自己的模型来预测
y_pred_diy = diy_model.forward(x)
print("diy模型预测结果：", y_pred_diy)
