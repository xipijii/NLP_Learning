import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
用了更多的pytorch的自带层
实际不用拆得这么好
"""

"""
一 构建类：构建所需参数和需要调用的方法
1. Bert类（参数 + embedding层 + ln, dp + 多层encoder）
2. encoder类（attention + dp&ln + feed forward + dp&ln）
3. 多头自注意力机制类（线性投影&多头处理 + 计算注意力分数 + 获取注意力概率 + 上下文层）
4. feed forward类（线性投影 + gelu非线性映射 + 线性投影）
二 注意事项
1. dropout的使用
（1）embedding组合之后先ln再dp，为了保证缩放和平移数据的准确性
（2）attention/feed forward之后的输出做dropout，再和输入做残差
（3）获取到注意力概率之后做dropout，降低在训练过程中过拟合的风险，提升模型的泛化能力
2. init中，即使有些网络层使用同样的结构、参数相同，也避免定义一个后在forward中复用（为了语义清晰 + 避免意外共享状态）
"""


class SimpleBERT(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072,
                 max_position_embeddings=512, hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, type_vocab_size=2):
        super(SimpleBERT, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # 嵌入层
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        # ✍️torch中含【归一化 + 学习缩放和平移的学习层】
        # ✍️epsilon（极小值），防止除以零的数值稳定性参数，当方差接近0时，加上这个小数避免除零错误
        self.dropout = nn.Dropout(hidden_dropout_prob)  # ✍️❓为何要dropout

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            SimpleBERTEncoderLayer(hidden_size, num_attention_heads, intermediate_size,
                                   hidden_dropout_prob, attention_probs_dropout_prob)
            for _ in range(num_hidden_layers)
        ])  # ✍️‼️nn.ModuleList 确保了层只创建一次、参数被正确管理、可以在训练中持续学习

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        # 创建位置ID ❓
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            # ✍️生成一个从 0 开始的一维张量，包含从 0 到 n-1 的整数序列
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
            # ✍️unsqueeze(0)：在第0维添加一个新维度，从 1D 变成 2D【 unsqueeze(0) 功能：在指定位置插入一个大小为 1 的新维度】
            # ✍️ expand(batch_size, seq_len)：将第0维扩展到 batch_size，实现批次广播【expand() 功能：将大小为 1 的维度扩展到指定大小，不复制数据（只是改变视图）

        # 创建token类型ID
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            # ✍️默认全为0

        # 嵌入层
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 组合嵌入
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # 通过编码器层
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class SimpleBERTEncoderLayer(nn.Module):
    """简化的BERT编码器层"""

    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 hidden_dropout_prob, attention_probs_dropout_prob):
        super(SimpleBERTEncoderLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # 自注意力机制
        self.self_attention = SimpleMultiHeadAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.attention_output_dropout = nn.Dropout(hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # 前馈网络
        self.intermediate = SimpleFeedForward(hidden_size, intermediate_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        # 自注意力子层
        attention_output = self.self_attention(hidden_states)
        attention_output = self.attention_output_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)

        # 前馈网络子层
        intermediate_output = self.intermediate(attention_output)  # ✍️feed forward
        intermediate_output = self.output_dropout(intermediate_output)
        layer_output = self.output_layer_norm(intermediate_output + attention_output)
        # ✍️1.add 加上原始输入，避免偏离原本的信息；2.减去均值再除以标准差，做缩放和平移

        return layer_output


class SimpleMultiHeadAttention(nn.Module):
    """简化的多头注意力机制"""

    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(SimpleMultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # 查询、键、值投影
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        """转置张量以计算注意力分数"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # ✍️解包成1个list
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # 线性投影
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 注意力概率
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 上下文层 ✍️知道是施加在V上之后要做拼接和平铺，但是具体代码指的是做啥不清楚
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # ❓
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        # ❓
        context_layer = context_layer.view(*new_context_layer_shape)
        # ❓

        # 输出投影 ✍️是计算出Z之后的线性层，进一步完成特征重组，整体显著提升模型的表示能力与表达灵活性
        attention_output = self.dense(context_layer)

        return attention_output


class SimpleFeedForward(nn.Module):
    """简化的前馈网络"""

    def __init__(self, hidden_size, intermediate_size):
        super(SimpleFeedForward, self).__init__()

        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


# 使用示例
"""
创建简化Bert模型
1.模型构造
2.创建示例输入
    batch_size=2, seq_len=128
    构建数据（0-词表大小随机数，size是上面的
3.输入模型
    对比输入和模型输出的大小
创建简化。。。。。。。
"""
if __name__ == "__main__":
    # 创建简化BERT模型
    bert = SimpleBERT(
        vocab_size=10000,  # 使用较小的词汇表
        hidden_size=512,  # 使用较小的隐藏层
        num_hidden_layers=6,  # 使用较少的层数
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512
    )

    # 创建示例输入
    batch_size = 2
    seq_len = 128

    input_ids = torch.randint(0, 10000, (batch_size, seq_len))  # ✍️❌2个句子的文本；而是2个单句的文本

    # 前向传播
    outputs = bert(input_ids)

    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {outputs.shape}")

    # 测试单个注意力头
    print("\n测试简化多头注意力:")
    multi_head_attn = SimpleMultiHeadAttention(hidden_size=512, num_attention_heads=8, dropout_prob=0.1)
    test_input = torch.randn(batch_size, seq_len, 512)
    attn_output = multi_head_attn(test_input)
    print(f"注意力输入形状: {test_input.shape}")
    print(f"注意力输出形状: {attn_output.shape}")

    # 测试单个编码器层
    print("\n测试简化编码器层:")
    encoder_layer = SimpleBERTEncoderLayer(
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    layer_output = encoder_layer(test_input)
    print(f"编码器层输入形状: {test_input.shape}")
    print(f"编码器层输出形状: {layer_output.shape}")
