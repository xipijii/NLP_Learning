# 手写 Transformer / BERT 实现

用 PyTorch 从零实现 BERT 的 Transformer 结构，并手工推导 + 程序验证其参数量（约 **1.02 亿**）。

- 语言 / 框架：Python · PyTorch

## 一、文件说明

| 文件 | 说明 | 是否依赖外部模型 |
| --- | --- | --- |
| `torch_diy_bert_transformer.py` | **主实现**：完整、可运行的简化版 BERT（多头自注意力 + 残差连接 + LayerNorm + 前馈网络） | 否，自包含 |
| `bert_cal_float.py` | 手工推导 BERT 参数量，并与 `bert-base-chinese` 实际参数比对 | 是，需本地 `bert-base-chinese` 权重 |
| `vocab.txt` | BERT 中文词表（21128 词），供草稿版读取 | — |

## 二、Transformer 结构与我对其的理解

BERT 的编码器由 N 层结构相同的 Transformer 块堆叠而成，单层数据流向如下：

```
输入 token ids
   │
   ├─ token embedding    （词向量）
   ├─ position embedding （位置编码）
   └─ segment embedding  （句段编码）       ── 三者相加
   │
   ▼
LayerNorm + Dropout
   │
   ▼   ←── 以下结构重复 layer_num 次
┌────────────────────────────────────────┐
│ 1. 多头自注意力（Multi-Head Self-Attention） │
│ 2. 残差连接 + LayerNorm（Add & Norm）        │
│ 3. 前馈网络（Feed-Forward，GELU）            │
│ 4. 残差连接 + LayerNorm（Add & Norm）        │
└────────────────────────────────────────┘
   │
   ▼
最终隐藏表示（取 [CLS] 向量经 Pooler 用于下游任务）
```

对应到代码（`hw6_1_torch_diy_bert_transformer_2.py`）：

- `SimpleBert.forward` —— 三种 embedding 相加、LayerNorm、循环堆叠编码层；
- `SimpleSingleEncoder` —— 一个 Transformer 块（注意力子层 + FFN 子层，各自带残差 + 归一）；
- `SimpleSelfAttention` —— 多头自注意力；
- `SimpleFeedForward` —— 前馈网络。

### 几个关键点的理解

1. **为什么是多头（Multi-Head）**：把 `hidden_size` 拆成 `head_size` 组，每组用更小的 `attention_head_size = hidden_size / head_size` 独立计算注意力，最后再拼接。目的是让模型在不同子空间里捕捉不同的注意力关系。代码里对应 `tran_score` 的 reshape 与 `permute(0, 2, 1, 3)`。

2. **为什么除以 √d_k**：注意力分数是 query 与 key 的点积，维度 d_k 越大点积方差越大，softmax 后容易进入饱和区、梯度变小。除以 `√d_k` 起缩放作用（`attention_score / math.sqrt(self.attention_head_size)`）。

3. **残差连接 + LayerNorm**：子层输出加上子层输入（残差）再做归一化，缓解深层网络的梯度消失 / 退化问题。代码中 `attention_layer_norm(attention_output + x)` 即标准写法。

4. **前馈网络**：每个 token 独立地做两次线性变换，中间用 GELU 激活。标准 BERT 中间维度是 `4 × hidden_size`；本主实现为方便演示简化为 `hidden_size → hidden_size`（草稿版 `hw6_1_torch_diy_bert_transformer.py` 用的是正确的 `4 ×`，可对照阅读）。

5. **位置编码**：Transformer 本身没有顺序概念，需靠 position embedding 显式注入位置信息。

## 三、参数量为什么是 1.02 亿（`hw6_2_bert_cal_float.py`）

以 `bert-base-chinese`（hidden=768，层数=12，词表=21128）为例：

| 模块 | 计算方式 | 说明 |
| --- | --- | --- |
| Embedding | `21128×768 + 2×768 + 512×768` | token + segment + position |
| 每层自注意力 | `3×(768×768+768) + (768×768+768)` | Q/K/V 投影 + 输出投影 |
| 每层 FFN | `768×3072+3072 + 3072×768+768` | 中间维度 4×768=3072 |
| 每层两个 LayerNorm | `(768+768) × 2` | 各含 gamma + beta |
| Pooler | `768×768 + 768` | 取 [CLS] 向量后的线性层 |

脚本中：

- `cal_float()` 按上表手工累加；
- `cal_bert_float()` 用 `state_dict()` 遍历真实模型权重求和。

两者结果一致：**102,267,648 ≈ 102M**。

> 注意：`cal_bert_float()` 需要本地存在 `bert-base-chinese` 模型权重，请替换脚本里 `from_pretrained(...)` 的路径。

## 四、运行方式

```bash
# 主实现（简化 BERT 前向，自包含）
python hw6_1_torch_diy_bert_transformer_2.py

# 草稿版（需同目录 vocab.txt）
python hw6_1_torch_diy_bert_transformer.py

# 参数量验证（需本地 bert-base-chinese 权重）
python hw6_2_bert_cal_float.py
```

依赖：

```bash
pip install torch transformers
```
