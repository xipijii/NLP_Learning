# NLP 学习笔记与代码实战（nlp_project02）

一套结构化的自然语言处理（NLP）课程代码与作业仓库，按周组织，覆盖从深度学习基础到大规模语言模型（LLM）、多模态、知识图谱与对话系统的完整学习路线。

主要使用 **Python + PyTorch** 实现，代码内注释为中文，理论与实践（`reference` / `practice` / `homework`）相结合。

> English version: [README.md](README.md)。

---

## 目录结构

```
.
├── week2_FundamentalsOfDeepLearning           # 深度学习基础
├── week3_DeepLearningForTextProcessing        # 面向文本处理的深度学习
├── week4_ChineseWordSegmentationAndTF-IDFFeatures  # 中文分词 & TF-IDF
├── week5_WordEmbeddingsAndTextRepresentations # 词向量与文本表示
├── week6_LanguageModelsAndPretraining         # 语言模型与预训练
├── week7_TextClassification                   # 文本分类
├── week8_TextMatching                         # 文本匹配
├── week9_SequenceLabeling                     # 序列标注 / NER
├── week10_TextGeneration                      # 文本生成
├── week11_LargeLanguageModels_Part1           # 大语言模型（一）
├── week12_LargeLanguageModels_Part2           # 大语言模型（二）
├── week13_LargeLanguageModels_Part3           # 大语言模型（三）
├── week14_LargeLanguageModelApplications      # 大语言模型应用
├── week15_MultimodalLargeModels               # 多模态大模型
├── week16_KnowledgeGraphs                     # 知识图谱
├── week17_DialogueSystems                     # 对话系统
└── logs/                                      # 运行日志
```

> 注：仓库从 week2 开始（week1 未收录）。

---

## 各周内容一览

| 周 | 主题 | 主要内容 |
| --- | --- | --- |
| **Week 2** | 深度学习基础 | 梯度下降、DNN 前向传播、交叉熵、PyTorch 入门（`p2_GradientDescent`、`p3_DNNforward`、`p4_CrossEntropy`、`p5_TorchDemo`） |
| **Week 3** | 面向文本的深度学习 | 反向传播与 Adam、Embedding、Pooling、RNN/CNN 前向、BatchNorm、Dropout；作业：RNN 文本分类 |
| **Week 4** | 中文分词 & TF-IDF | 上午：中文分词；下午：新词发现、TF-IDF、基于 TF-IDF 的搜索引擎、文本摘要、文本相似度 |
| **Week 5** | 词向量与文本表示 | Word2Vec、KMeans 聚类、向量距离计算（`hw5_word2vec_kmeans_distance.py`） |
| **Week 6** | 语言模型与预训练 | 从零实现 NNLM、手写 Transformer / BERT、手工推导并验证 BERT 参数量（约 1.02 亿） |
| **Week 7** | 文本分类 | 文本分类完整 pipeline（`config` / `loader` / `model` / `evaluator` / `main`），训练数据 + 测试数据 |
| **Week 8** | 文本匹配 | 相似度函数（`p1_similarity_function.py`）；作业：句子对匹配，含训练/验证数据与 `predictor.py` |
| **Week 9** | 序列标注 / NER | 基于 BERT + LSTM + CRF 的命名实体识别，支持 Viterbi 解码 |
| **Week 10** | 文本生成 | 基于 LSTM 的语言模型（NNLM）、Seq2Seq 机器翻译、Transformer（`attention-is-all-you-need`） |
| **Week 11** | 大语言模型（一） | BERT 监督微调（SFT，`bert_sft.py`） |
| **Week 12** | 大语言模型（二） | 主流大模型对比：Qwen-7B、Ernie 4.5 MoE、GLM4、DeepSeek-v3、BaiChuan、ChatGLM、Kimi-K2.6 |
| **Week 13** | 大语言模型（三） | 大模型时代的 NER：BERT + CRF/LSTM 训练与评测 |
| **Week 14** | 大语言模型应用 | 差旅报销 Agent 示例，演示 LLM 的 Function Call 能力（`Reimbursement_agent_demo.py`） |
| **Week 15** | 多模态大模型 | BPE 分词器实现与解码（`bpe_tokenizer_decoding.py`） |
| **Week 16** | 知识图谱 | 三元组抽取、构建知识图谱并写入 Neo4j（`build_graph.py`） |
| **Week 17** | 对话系统 | 任务型多轮对话 Web 服务（FastAPI 后端 + ChatGPT 风格前端 + 槽位填充） |

---

## 技术栈 / 依赖

- **语言**：Python 3
- **深度学习框架**：PyTorch
- **NLP / 预训练模型**：Transformers（Hugging Face）、BERT 等
- **其他常用库**：NumPy、pandas、openpyxl
- **任务型对话**：FastAPI、Uvicorn
- **知识图谱**：Neo4j
- **大模型调用**：OpenAI SDK（阿里云百炼等）

依赖通过项目内 `.venv` 虚拟环境管理，未提供顶层 `requirements.txt`（部分子目录如 `week17` 自带 `requirements.txt`）。

---

## 快速开始

```bash
# 1. 克隆仓库
git clone <仓库地址>
cd nlp_project02

# 2. 创建并激活虚拟环境（如已有 .venv 可直接激活）
python -m venv .venv
source .venv/bin/activate

# 3. 安装依赖（示例，按需安装）
pip install torch transformers numpy pandas openpyxl
# 序列标注（week9 / week13）还需：
pip install pytorch-crf
# 对话系统（week17）还需：
pip install fastapi "uvicorn[standard]"

# 4. 进入某个周目录运行对应脚本
cd week6_LanguageModelsAndPretraining/02_practice
python p02_nnlm.py
```

> 各周脚本大多可直接运行；需要预训练模型权重或数据集的部分，请根据脚本内注释自行下载。

---

## 命名约定

- `reference` / `01_reference` / `d1_reference`：参考代码（课程提供的标准实现或示例）
- `practice` / `02_practice` / `d2_practice`：课堂练习
- `homework` / `03_homework` / `d3_homework` / `a3_homework`：课后作业

---

## 注意事项

- 本仓库为学习用途，部分作业代码中可能残留测试用的 API Key、数据库密码等占位信息，正式使用前请替换为环境变量。
- 模型权重（`.pth` / `.pt` / `.bin` / `.safetensors`）、预训练模型文件及大型数据集已通过 `.gitignore` 排除，不会随仓库提交，需自行准备。
- 部分数据文件以压缩包（`.zip` / `.rar`）形式保留在仓库中，解压后按脚本路径使用。
