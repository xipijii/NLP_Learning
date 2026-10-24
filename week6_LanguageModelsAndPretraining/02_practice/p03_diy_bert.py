"""
通过numpy 手动矩阵运算实现Bert结构
一 使用引入的Bert模型，对4个字的句子进行网络层计算，输出向量结果
1. 从transformer模块中引入BertModel包，加载模型
2. 获取模型权重，开启评估模式❓
3. 在模型里对测试文本进行张量输入，获取每个字和[CLS]对应的向量
4. 输出查看Bert模型权重矩阵的所有名称（diy使用）

二 diy实现Bert
1. self-attention部分使用 - softmax的实现（查看公式来实现
2. feed forward部分使用 - gelu激活函数的实现（查看公式来实现
3. 定义Bert类
 (1) 初始化：
    定义参数（与预训练config.json文件中的值保持一致）- 头数，隐单元大小，transformer层数
    加载模型权重（一 4.）
 (2) 定义“加载模型权重”方法：
    embedding部分（3个矩阵，1个normalization层权重）
    transformer部分：
        自注意力机制：
            QKV三个线性层的W和b的大小
            QKV每个后面都会过一个h*h的线性层❓
        残差机制：
            归一化层（W和b）
        feed forward层（两个线性层中间加一个gelu激活层）
    pooler层（用于对[CLS]token进行非线性映射，得到一个更适合下游句子级任务的表示-尤其是分类任务）
 (3) 定义embedding层的处理（叠加再归一化）
    token embedding，从词表中获取文本中字符对应的向量
    position embedding，文本的长度范围下进行遍历，获取位置序列[0,1,2,3]
    segment embedding，属于第一句还是第二句，单输入时为[0,0,0,0]
    加和后归一化（平移+缩放；线性层使其学习缩放和平移的参数）
 (4) 定义“按照索引获取embedding层”的方法
 (5) 定义执行全部的transformer层计算
    x传入n次，调用单层计算
 (6) 定义执行单层transformer层计算
    获取第i层的权重矩阵
    取出i层的参数
    调用自注意力方法
    调用（残差机制）归一方法   （对多头自注意力的输出 加入原始输入再归一）
    调用feed forward处理（线性层+激活层+线性层）（2组权重）
    调用（残差机制）归一方法   （对feed forward的输出 加入原始输入再归一）
 (7) 定义自注意力机制的计算方法
        x.shape = max_len * hidden_size
        q,k,v_w .shape = hidden_size * hidden_size
        q,k,v_b .shape = hidden_size ❓‼️
    调用【多头机制方法】计算 Q, K, V - (max_len * hidden_size)
    计算qk-   head_num * max_len * max_len
    计算qkv-  head_num * max_len * (hidden_size / head_num)
    过一个线性层 ‼️多头拼接后的线性层进一步完成特征重组，整体显著提升模型的表示能力与表达灵活性
 (8) 定义多头机制的方法
    获取x的 max_len 和 hidden_size
    在x的hidden_size纬度进行切分，切成 头数 * 头数大小，即 768 = 12 * 64
    将头数与max_len做纬度交换，便于计算
 (9) 定义前反馈网络的计算方法
    第一个线性层 -
    激活层 - gelu
    第二个线性层 -
 (10) 定义链接[CLS]token的输出层 - 非线性映射，使其更适合下游句子级任务的表示（尤其分类任务）
    对文本中的第一个向量（应该是[CLS]token对应的向量）过一个线性层
    过tanh激活函数，进行非线性映射

三 使用自定义的和torch封装的Bert对4字文本向量进行计算，输出最后的结果
"""



