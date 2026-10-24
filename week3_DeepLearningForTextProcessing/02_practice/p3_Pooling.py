from torch import nn
import torch

"""
NLP中的Pooling层处理
1. 调用torch中的avgPooling方法，传入池化纬度（文本长度）
2. 随机生成一个纬度为3*4*5的张量，代表3个4字字符串
3. 张量的最后两维做转置，后池化
"""

pooling_layer = nn.AvgPool1d(4)

x = torch.rand([3, 4, 5])
print("随机生成的张量：", x)
print("形状：", x.shape)
# x.transpose(1, 2)
x = x.transpose(1, 2)
print("转置后的张量：", x)
print("形状：", x.shape)

x = pooling_layer(x)
print("池化后的张量", x)
print("形状：", x.shape)

# x = torch.squeeze(x)
x = x.squeeze()  # ✅ squeeze方法去掉值为1的维度
print("压缩后的张量", x)
print("形状：", x.shape)
