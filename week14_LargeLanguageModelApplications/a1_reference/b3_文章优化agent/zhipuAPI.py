import os
from zai import ZhipuAiClient

#pip install zhipuai
#https://open.bigmodel.cn/ 注册获取APIKey

client = ZhipuAiClient(api_key=os.environ.get("zhipuApiKey")) #key填入环境变量
response = client.chat.completions.create(
    model="glm-3-turbo",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": "黑神话悟空好玩吗"},
    ],
)
print(response.choices[0].message.content)