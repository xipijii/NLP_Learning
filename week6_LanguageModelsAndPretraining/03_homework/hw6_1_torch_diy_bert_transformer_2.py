from torch import nn
import torch
import math


class SimpleBert(nn.Module):

    def __init__(self, vocab_size, hidden_size, layer_num
                 , head_size, token_type, max_position_size, dropout_prob):
        super(SimpleBert, self).__init__()

        self.max_position_size = max_position_size
        self.token_type = token_type
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_position_size, hidden_size, padding_idx=0)
        self.segmentation_embedding = nn.Embedding(token_type, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout_prob)
        # self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size * hidden_size, eps=1e-2)  # ❌
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-12)
        # ❓

        self.encoder_layers = nn.ModuleList([
            SimpleSingleEncoder(hidden_size, head_size, dropout_prob)
            for _ in range(layer_num)
        ])

    def forward(self, x):
        batch_size, seq_len = x.shape  # ‼️

        position_input = torch.arange(0, seq_len, dtype=torch.long, device=x.device)  # ‼️
        position_input = position_input.unsqueeze(0).expand(batch_size, seq_len)  # ‼️
        type_input = torch.zeros_like(x)  # ‼️

        token_emb = self.token_embedding(x)
        position_emb = self.position_embedding(position_input)
        segmentation_emb = self.segmentation_embedding(type_input)

        embeddings = token_emb + position_emb + segmentation_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # ‼️通过编码器层实现
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class SimpleSingleEncoder(nn.Module):

    def __init__(self, hidden_size, head_size, dropout_prob):
        super(SimpleSingleEncoder, self).__init__()

        self.self_attention = SimpleSelfAttention(hidden_size, head_size, dropout_prob)
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-2)
        self.feed_forward = SimpleFeedForward(hidden_size)
        self.ff_dropout = nn.Dropout(dropout_prob)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-2)

    def forward(self, x):
        attention_output = self.self_attention(x)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + x)
            # ‼️残差机制 传入：attention+dropout 层的输入和输出

        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.ff_dropout(feed_forward_output)
        feed_forward_output = self.output_layer_norm(feed_forward_output + attention_output)

        return feed_forward_output


class SimpleSelfAttention(nn.Module):

    def __init__(self, hidden_size, head_size, dropout_prob):
        super(SimpleSelfAttention, self).__init__()

        self.head_size = head_size
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // head_size

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.value_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)  # ‼️输出投影

    # 多头处理
    def tran_score(self, x):
        # x = x.size()[:-1] + (self.head_size, self.attention_head_size)  # ‼️
        new_x_shape = x.size()[:-1] + (self.head_size, self.attention_head_size)
        # ‼️以上是torch.Size形式，是一个新的张量形状，需要展平
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        # 线性投影 + 多头
        query_ = self.tran_score(self.query_linear(x))
        key_ = self.tran_score(self.key_linear(x))
        value_ = self.tran_score(self.value_linear(x))

        # 计算注意力分数 + 注意力概率
        # attention_score = torch.dot(query_, key_.transpose(0, 1))  # ‼️转置的是后两纬
        # attention_score = torch.dot(query_, key_.transpose(-1, -2))  # ‼️多头矩阵用matmul计算
        attention_score = torch.matmul(query_, key_.transpose(-1, -2))
        # ‼️漏了一步：除以dk
        # attention_score = attention_score / torch.sqrt(self.attention_head_size)  # ‼️torch.sqrt的参数必须是个张量
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        attention_pro = self.softmax(attention_score)
        # ✍️‼️漏了一步



        # 上下文层
        attention_output = torch.matmul(attention_pro, value_)

        # 形状变换：
        #   (batch_size, head_size, max_len, attention_head_size)
        #   (batch_size, max_len, head_size, attention_head_size)
        #   (batch_size, max_len, hidden_size)
        # y = attention_output.permute(0, 2, 1, 3)  # ‼️内存布局不再连续了
        y = attention_output.permute(0, 2, 1, 3).contiguous()
        # y = y.size()[:-2] + self.hidden_size  # ‼️torchSize是元组，加元组应该是(x,)
        new_shape = y.size()[:-2] + (self.hidden_size, )
        y = y.view(*new_shape)

        y = self.dense(y)  # ❓位置对吗

        return y


class SimpleFeedForward(nn.Module):

    def __init__(self, hidden_size):
        super(SimpleFeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x


if __name__ == '__main__':
    bert = SimpleBert(
        vocab_size=10000,
        hidden_size=768,
        head_size=8,
        layer_num=6,
        token_type=2,
        max_position_size=512,
        dropout_prob=0.1
    )

    batch_size = 2
    seq_len = 128

    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    # 前向传播
    y = bert(input_ids)
    print(input_ids.shape)
    print(y.shape)

    # 测试单个编码器层
    print("\n测试单个编码器层：")
    single_encoder = SimpleSingleEncoder(768, 12, 0.1)
    test_input = torch.randn(batch_size, seq_len, 768)
    encoder_output = single_encoder(test_input)
    print(test_input.shape)
    print(encoder_output.shape)

    # 测试单个注意力头
    print("\n测试简化多头注意力：")
    self_attention = SimpleSelfAttention(768, 12, 0.1)
    test_input = torch.randn(batch_size, seq_len, 768)
    attention_output = self_attention(test_input)
    print(test_input.shape)
    print(attention_output.shape)
