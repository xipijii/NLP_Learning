#coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理
'''

num_embeddings = 8  #通常对于nlp任务，此参数为字符集字符总数
embedding_dim = 5  #每个字符向量化后的向量维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    # ✍️ padding_idx=0 为第1个向量进行全0初始化-->补齐向量，占位符
print("随机初始化权重")
print(embedding_layer.weight)
print("################")

# 如何使用？
# 1️⃣构造字符表 ✍️字表和embedding层的向量是绑定的
vocab = {
    "[pad]": 0,  # 补齐向量
    "你": 1,
    "好": 2,
    "中": 3,
    "国": 4,
    "欢": 5,
    "迎": 6,
    "[unk]": 7  # 未知的，数据中的未知字符使用
}


#中国欢迎你 -> 中 国 欢 迎 你 -> [3,4,5,6,1] -> ✍️取出embedding层中对应的向量 embedding_layer([3,4,5,6,1]) -> 5*5 矩阵
#你好中国 -> 你 好 中 国 -> [1,2,3,4] -> embedding_layer([1,2,3,4]) -> 4*5 矩阵


# 为了让不同长度的训练样本能够放在同一个batch中，需要将所有样本补齐或截断到相同长度
# 限制最大长度为5
# padding 补齐
# [1,2,3,0,0]
# [1,2,3,4,0]
# [1,2,3,4,5]
# 截断
# [1,2,3,4,5,6,7] -> [1,2,3,4,5]
# ✍️尽量少截断，多补齐；一般设置99%的数据都满足的长度。截断量不是很大时，不影响结果

def str_to_sequence(string, vocab):
    seq = [vocab.get(s, vocab["[unk]"]) for s in string][:5]  # [:5]进行截断
    if len(seq) < 5:
        seq += [vocab["[pad]"]] * (5 - len(seq))
    return seq


string1 = "abcd"
string2 = "ddcc"
string3 = "feda"

sequence1 = str_to_sequence(string1, vocab)
sequence2 = str_to_sequence(string2, vocab)
sequence3 = str_to_sequence(string3, vocab)

print(sequence1)
print(sequence2)
print(sequence3)

x = torch.LongTensor([sequence1, sequence2, sequence3])
embedding_out = embedding_layer(x)
print(embedding_out)
