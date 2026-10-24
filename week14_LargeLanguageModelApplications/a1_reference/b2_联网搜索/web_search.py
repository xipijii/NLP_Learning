import json
import os
from exa_py import Exa

#一个国外的搜索引擎接口
#https://dashboard.exa.ai/home

#pip install exa-py


exa = Exa(api_key = os.getenv("EXA_API_KEY"))

result = exa.search_and_contents(
  "北京景点",
  text = True,
  type = "auto"
)

print(result)