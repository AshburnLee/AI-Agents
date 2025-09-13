from langchain_ollama import ChatOllama
import os

os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"

infer_server_url = "http://localhost:11434/"
model_name = "qwen3:1.7b"

base_model = ChatOllama(
    model=model_name,
    base_url=infer_server_url,
    api_key="none",
    temperature=0,
    stream=False
)

'''
这里没有 humanMessage AIMessage 而是直接的response
'''
response = base_model.invoke("介绍你自己？")
print("直接模型响应:", response.content)

"""
done
"""
