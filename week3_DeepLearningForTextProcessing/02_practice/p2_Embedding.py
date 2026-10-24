import torch
from torch import nn
import numpy as np

"""
Embedding层的处理
1. 定义Embedding层（参数）
2. 构建词表
3. 实现字符串转化为整数序列
4. 通过整数序列（张量）去Embedding矩阵中找到对应的向量及其矩阵
"""

number_embedding_layer = 16  # 词表的长度
embedding_dim = 5  # 每个字符向量的纬度
embedding_layer = nn.Embedding(number_embedding_layer, embedding_dim)
# print("embedding层初始化权重：", embedding_layer.state_dict())
print("embedding层初始化权重：", embedding_layer.weight)  # ✅

# 构建词表 使用字典
vocab = {
    "[pad]": 0,
    "你": 1,
    "我": 2,
    "她": 3,
    "他": 4,
    "中": 5,
    "华": 6,
    "国": 7,
    "家": 8,
    "a": 9,
    "b": 10,
    "c": 11,
    "d": 12,
    "e": 13,
    "f": 14,
    "[unk]": 15
}


def str_to_sequence(string, vocab):
    seq = [vocab.get(s, vocab.get("[unk]")) for s in string][:7]
    if len(seq) < 7:
        # seq.append([0] * (7 - len(seq)))  # ❌
        # seq += [0] * (7 - len(seq)) # 🉑
        seq += [vocab.get("[pad]")] * (7 - len(seq))  # ✅
    return seq


str1 = "你好中国abc"
str2 = "中国欢迎你"
str3 = "hhhaaabcdef"

seq1 = str_to_sequence(str1, vocab)
seq2 = str_to_sequence(str2, vocab)
seq3 = str_to_sequence(str3, vocab)

print(seq1)
print(seq2)
print(seq3)

array = np.array([seq1, seq2, seq3])

embedding_output = embedding_layer(torch.LongTensor(array))  # 前提：将角标向量转换为张量
print(embedding_output)
