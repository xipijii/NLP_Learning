"""
使用pytorch实现Bert中的transformer
结构
1. embedding
2. self-attention
3. add+normalize
4. feed forward
5. add+normalize
6. pooler
"""
import torch
from torch import nn
import numpy as np


class TorchDiyTransformerModel(nn.Module):

    # 定义网络结构
    def __init__(self, vocab, max_len, hidden_size, layer_num, head_num
                 , attention_head_size):  # ✍️attention_head_size
        super(TorchDiyTransformerModel, self).__init__()

        self.token_embedding = nn.Embedding(num_embeddings=len(vocab)
                                            , embedding_dim=hidden_size)
        self.segment_embedding = nn.Embedding(num_embeddings=2
                                              , embedding_dim=hidden_size)
        self.position_embedding = nn.Embedding(num_embeddings=512
                                               , embedding_dim=hidden_size)
        # embedding 线性映射
        self.hh_linear = nn.Linear(hidden_size, hidden_size)

        # 【normalize】 hidden_size * hidden_size

        # QKV的三个线性层映射 hidden_size * hidden_size
        # Z的线性层 hidden_size * hidden_size
        # qk softmax
        self.softmax = nn.Softmax(dim=-1)
        # 【add+normalize】 hidden_size * hidden_size
        # feed forward
        self.ff_linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        # self.ff_gelu = nn.GELU  # ❌
        self.ff_gelu = nn.GELU()
        self.ff_linear2 = nn.Linear(4 * hidden_size, hidden_size)
        # 【add+normalize】 hidden_size * hidden_size

        # pooler线性映射 hidden_size * hidden_size
        # pooler非线性映射
        self.pl_tanh = nn.Tanh()

        # 参数
        self.max_len = max_len
        self.layer_num = layer_num
        self.head_num = head_num
        self.attention_head_size = attention_head_size

    def forward(self, x):
        # original_x = copy.deepcopy(x)  # 需不需要深度复制
        x_em1 = self.token_embedding(x)
        x_em2 = self.segment_embedding(torch.LongTensor(np.array([0] * len(x))))  # todo ？
        x_em3 = self.position_embedding(torch.LongTensor(np.array([i for i in range(len(x))])))  # todo ？
        x = x_em1 + x_em2 + x_em3
        # x = (x - np.mean(x)) / np.std(x)  # 归一化需要用计算实现还是用一个激活函数合适❓
        x = (x - torch.mean(x, dim=1, keepdims=True)) / torch.std(x, dim=1, keepdims=True)
        x = self.hh_linear(x)
        original_x = x.clone()  # 深拷贝会断开梯度传播，直接保存张量引用（浅拷贝）

        # 自注意力 + 分成多头
        q = self.hh_linear(x)
        q = q.reshape(self.max_len, self.head_num, self.attention_head_size).swapaxes(1, 0)
        k = self.hh_linear(x)
        k = k.reshape(self.max_len, self.head_num, self.attention_head_size).swapaxes(1, 0)
        v = self.hh_linear(x)
        v = v.reshape(self.max_len, self.head_num, self.attention_head_size).swapaxes(1, 0)
        q_k = self.softmax(torch.matmul(q, k.transpose(1, 2)))
        # ✍️torch.dot() 只能用于1D张量，矩阵乘法（批量矩阵乘法），应该使用 torch.matmul()

        q_k_v = torch.matmul(q_k, v)
        q_k_v = q_k_v.swapaxes(1, 0).reshape(-1, hidden_size)
        x = self.hh_linear(q_k_v)

        # 残差归一
        add = x + original_x  # ❌‼️这里不是和embedding层之后的x做残差了，而是和经过自注意力层的做
        x = (add - torch.mean(add, dim=1, keepdims=True)) / torch.std(add, dim=1, keepdims=True)
        x = self.hh_linear(x)

        # feed forward
        x = self.ff_linear1(x)
        x = self.ff_gelu(x)
        x = self.ff_linear2(x)

        # 残差归一
        add = x + original_x  # ❌‼️这里不是和embedding层之后的x做残差了，而是和经过前馈网络子层的做
        x = (add - torch.mean(add, dim=1, keepdims=True)) / torch.std(add, dim=1, keepdims=True)
        x = self.hh_linear(x)

        # pooler
        pooler_input = x[0]
        pooler_output = self.hh_linear(pooler_input)
        pooler_output = self.pl_tanh(pooler_output)

        return x, pooler_output


def build_vocab(path):
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            word = line.strip()
            vocab[word] = index
    return vocab


if __name__ == '__main__':
    vocab = build_vocab("vocab.txt")
    test_sentence = np.array([2450, 15486, 102, 2110])
    hidden_size = 768
    layer_num = 12
    head_num = 12
    attention_head_size = 64
    model = TorchDiyTransformerModel(vocab, len(test_sentence), hidden_size, layer_num, head_num, attention_head_size)
    output, pooler_output = model(torch.LongTensor(test_sentence))
    print(output)
    print(pooler_output)
