

# 安装所需库
# pip install cho openai dashscope
import os
import numpy as np

import chromadb  # ✍️一个向量数据库
from chromadb.config import Settings

from get_embeding import query_to_vector, MyEmbeddingFunction
from openai import OpenAI

#阿里百炼平台
#https://bailian.console.aliyun.com/?spm=5176.29619931.J__Z58Z6CX7MY__Ll8p1ZOR.1.183b521cJOJxLA&tab=doc#/doc

# 初始化Chroma客户端（Windows兼容模式）
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",  # 数据存储目录
    is_persistent=True  # 确保数据持久化
))


# 获取所有集合的名称
collections = client.list_collections()  # ✍️多个文件【可理解为多个节点，存储好几份】

# 遍历并删除每个集合
for collection in collections:
    client.delete_collection(name=collection.name)
    print(f"已删除集合: {collection.name}")


# 创建或获取集合
collection_name = "rag"
collection = client.create_collection(
    name=collection_name,
    embedding_function=MyEmbeddingFunction()  # ✍️指定embedding的方法是什么，这里是个embedding模型
)
print(f"已创建新集合: {collection_name}")

# 示例文档
# documents = [open(os.path.join("Heroes", p), "r", encoding="utf-8").read() for p in os.listdir("Heroes")]
documents = [
    "Windows 10是微软发布的操作系统",
    "向量数据库可以高效存储和检索嵌入向量",
    "Chroma是一个轻量级的开源向量数据库",
    "RAG技术可以增强大语言模型的回答准确性",
    "Python在Windows上有良好的开发环境支持"
]

# 为文档创建唯一ID
ids = [f"doc_{i}" for i in range(len(documents))]

# 添加文档到集合（Chroma会自动处理嵌入，也可使用自定义嵌入）
collection.add(
    documents=documents,  # ✍️这里就会对于所有文档通过指定的model来进行向量化
    ids=ids
)

#指定大模型接口
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 检索相关文档
def search_similar(query, top_k=1):
    results = collection.query(  # ✍️这里的collection就是向量数据库
        query_texts=[query],  # ✍️用户问
        n_results=top_k  # ✍️指定召回top几
    )
    return results

# 定义检索函数
def retrieve_documents(query, top_k=1):
    # 生成查询向量
    retrieved_docs = search_similar(query, top_k=top_k)    
    return retrieved_docs["documents"][0]

# 定义RAG生成函数
def rag_generate(query):
    # 检索相关文档
    retrieved_docs = retrieve_documents(query)
    print("\n检索到的相关文档:")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i}. {doc} ")
    # 构建提示词
    context = "\n".join(retrieved_docs)
    prompt = f"""基于以下信息回答问题:
{context}

问题: {query}
"""
    
    completion = client.chat.completions.create(
        model="qwen-plus-2025-07-14",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return completion.choices[0].message.content

# 测试RAG流程
if __name__ == "__main__":
    query = "什么是RAG，它如何工作？"
    print(f"查询: {query}")
    
    answer = rag_generate(query)
    print("\n生成的回答:")
    print(answer)
    

