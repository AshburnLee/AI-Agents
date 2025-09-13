from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import create_react_agent

import os
os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"

import requests
import json

infer_server_url = "http://localhost:11434/api/generate"   # 使用正确的端点 !
# model_name = "llama3.2:1b"
model_name = "qwen3:1.7b"

# 定义天气工具
def get_weather(city: str) -> str:
    """获取给定城市的天气"""
    return f"{city} 的天气总是晴朗！"

# 定义提示
prompt = """
你是一个有用的助手。请严格按照以下 JSON 格式响应，仅返回 JSON，勿添加额外文本。
{
  "tool": "get_weather",
  "parameters": {
    "city": "<城市名>"
  }
}
查询：使用已有的工具给出上海的天气？
"""

# 调用 Ollama API
# prompt、format 和 stream 字段出现在 payload 中，因为它们是 Ollama /api/generate 端点的核心参数，用于定义输入、输出格式和响应方式。
# 所有参数：https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
payload = {
    "model": model_name,
    "prompt": prompt,
    "format": "json",  # 强制 JSON 输出
    "stream": False
}

try:
    response = requests.post(infer_server_url, json=payload)
    if response.status_code == 200:
        result = response.json().get("response")
        print("Ollama API 原始输出:", result,"\n")  # 调试输出
        parsed_result = json.loads(result)
        print(parsed_result)
        if parsed_result.get("tool") == "get_weather":
            city = parsed_result.get("parameters", {}).get("city")
            if city:
                print(get_weather(city))
            else:
                print("错误：缺少城市参数")
        else:
            print("未找到有效的工具调用")
    else:
        print(f"API 错误：{response.status_code}")
except json.JSONDecodeError:
    print("错误：模型输出不是有效的 JSON")
except Exception as e:
    print(f"运行时错误：{e}")

'''
如期输出
'''
