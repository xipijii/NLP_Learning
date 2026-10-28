# NLP Learning Notes & Code (nlp_project02)

A structured, week-by-week NLP course repository covering a complete learning path — from deep learning fundamentals to large language models (LLMs), multimodal models, knowledge graphs, and dialogue systems.

Implemented mainly in **Python + PyTorch**, with Chinese code comments, combining theory with practice (`reference` / `practice` / `homework`).

> 中文说明见 [README.zh.md](README.zh.md)。

---

## Directory Structure

```
.
├── week2_FundamentalsOfDeepLearning           # Deep learning fundamentals
├── week3_DeepLearningForTextProcessing        # Deep learning for text processing
├── week4_ChineseWordSegmentationAndTF-IDFFeatures  # Chinese word segmentation & TF-IDF
├── week5_WordEmbeddingsAndTextRepresentations # Word embeddings & text representations
├── week6_LanguageModelsAndPretraining         # Language models & pretraining
├── week7_TextClassification                   # Text classification
├── week8_TextMatching                         # Text matching
├── week9_SequenceLabeling                     # Sequence labeling / NER
├── week10_TextGeneration                      # Text generation
├── week11_LargeLanguageModels_Part1           # Large language models (I)
├── week12_LargeLanguageModels_Part2           # Large language models (II)
├── week13_LargeLanguageModels_Part3           # Large language models (III)
├── week14_LargeLanguageModelApplications      # LLM applications
├── week15_MultimodalLargeModels               # Multimodal large models
├── week16_KnowledgeGraphs                     # Knowledge graphs
├── week17_DialogueSystems                     # Dialogue systems
└── logs/                                      # Runtime logs
```

> Note: the repository starts from week2 (week1 is not included).

---

## Weekly Overview

| Week | Topic | Key Content |
| --- | --- | --- |
| **Week 2** | Deep learning fundamentals | Gradient descent, DNN forward pass, cross-entropy, PyTorch basics (`p2_GradientDescent`, `p3_DNNforward`, `p4_CrossEntropy`, `p5_TorchDemo`) |
| **Week 3** | Deep learning for text | Backpropagation & Adam, embeddings, pooling, RNN/CNN forward pass, BatchNorm, Dropout; homework: RNN text classification |
| **Week 4** | Chinese word segmentation & TF-IDF | Morning: Chinese word segmentation; afternoon: new word discovery, TF-IDF, TF-IDF search engine, text summarization, text similarity |
| **Week 5** | Word embeddings & text representation | Word2Vec, KMeans clustering, vector distance (`hw5_word2vec_kmeans_distance.py`) |
| **Week 6** | Language models & pretraining | NNLM from scratch, hand-written Transformer / BERT, manually derive and verify BERT parameter count (~102M) |
| **Week 7** | Text classification | End-to-end text classification pipeline (`config` / `loader` / `model` / `evaluator` / `main`), train + validation data |
| **Week 8** | Text matching | Similarity functions (`p1_similarity_function.py`); homework: sentence-pair matching with train/valid data and `predictor.py` |
| **Week 9** | Sequence labeling / NER | Named entity recognition with BERT + LSTM + CRF, including Viterbi decoding |
| **Week 10** | Text generation | LSTM-based language model (NNLM), Seq2Seq machine translation, Transformer (`attention-is-all-you-need`) |
| **Week 11** | LLMs (I) | BERT supervised fine-tuning (SFT, `bert_sft.py`) |
| **Week 12** | LLMs (II) | Comparison of mainstream LLMs: Qwen-7B, Ernie 4.5 MoE, GLM4, DeepSeek-v3, BaiChuan, ChatGLM, Kimi-K2.6 |
| **Week 13** | LLMs (III) | NER in the LLM era: BERT + CRF/LSTM training and evaluation |
| **Week 14** | LLM applications | Travel reimbursement Agent demo demonstrating LLM Function Calling (`Reimbursement_agent_demo.py`) |
| **Week 15** | Multimodal large models | BPE tokenizer implementation and decoding (`bpe_tokenizer_decoding.py`) |
| **Week 16** | Knowledge graphs | Triple extraction, building a knowledge graph and writing it to Neo4j (`build_graph.py`) |
| **Week 17** | Dialogue systems | Task-oriented multi-turn dialogue web service (FastAPI backend + ChatGPT-style frontend + slot filling) |

---

## Tech Stack / Dependencies

- **Language**: Python 3
- **Deep learning framework**: PyTorch
- **NLP / pretrained models**: Transformers (Hugging Face), BERT, etc.
- **Other common libraries**: NumPy, pandas, openpyxl
- **Dialogue system**: FastAPI, Uvicorn
- **Knowledge graph**: Neo4j
- **LLM calls**: OpenAI SDK (e.g. Alibaba Cloud Bailian)

Dependencies are managed via the project's `.venv` virtual environment; there is no top-level `requirements.txt` (some subdirectories, e.g. `week17`, include their own).

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd nlp_project02

# 2. Create and activate a virtual environment (activate existing .venv if present)
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (as needed)
pip install torch transformers numpy pandas openpyxl
# For sequence labeling (week9 / week13):
pip install pytorch-crf
# For the dialogue system (week17):
pip install fastapi "uvicorn[standard]"

# 4. Enter a week's directory and run the corresponding script
cd week6_LanguageModelsAndPretraining/02_practice
python p02_nnlm.py
```

> Most scripts run directly; for scripts that require pretrained weights or datasets, download them as noted in the code comments.

---

## Naming Conventions

- `reference` / `01_reference` / `d1_reference`: reference code (standard implementations or examples provided by the course)
- `practice` / `02_practice` / `d2_practice`: in-class exercises
- `homework` / `03_homework` / `d3_homework` / `a3_homework`: homework assignments

---

## Notes

- This is a learning repository. Some homework scripts may still contain placeholder API keys or database passwords used for testing — replace them with environment variables before real use.
- Model weights (`.pth` / `.pt` / `.bin` / `.safetensors`), pretrained model files, and large datasets are excluded via `.gitignore` and are not committed; prepare them yourself.
- Some data files are kept as archives (`.zip` / `.rar`); extract them and follow the script paths.
