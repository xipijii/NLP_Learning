import dashscope
import json
import numpy as np
from http import HTTPStatus
from chromadb import Documents, EmbeddingFunction, Embeddings

#调用阿里百炼平台的通用多模态表征模型接口
#需要在系统环境变量中配置自己的APIKey
#目前是限时免费使用,如果大量快速调用可能会被限流
def query_to_vector(text):
    input = [{'text': text}]
    # 调用模型接口
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",  # ✍️阿里百炼
        input=input
    )

    if resp.status_code == HTTPStatus.OK:
        result = {
            "status_code": resp.status_code,
            "request_id": getattr(resp, "request_id", ""),
            "code": getattr(resp, "code", ""),
            "message": getattr(resp, "message", ""),
            "output": resp.output,
            "usage": resp.usage
        }
        # output = json.dumps(result, ensure_ascii=False, indent=4)
        # print(output)
        vector = np.array(result["output"]["embeddings"][0]["embedding"])
        return vector
    else:
        print("调用失败")
        return None

#做一层封装，框架使用要求
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
    def __call__(self, input: Documents) -> Embeddings:
        return [query_to_vector(d) for d in input]

if __name__ == '__main__':
    v = query_to_vector("你好")
    print(v.shape)