from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from conponents.extract_ai_response import extract_aimessage_content, show_message_content

os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"
# 初始化 ChatOllama
infer_server_url = "http://localhost:11434/"
model_name = "qwen3:1.7b"
base_model = ChatOllama(
    model=model_name,
    base_url=infer_server_url,
    api_key="none",
    temperature=0,
    stream=False
)

# 定义提示模板
# 如何使用这个模板？？？
prompt_template = "你是一个博学的代理"

# 创建 ReAct 代理
tools = []
"""
创建 agent 时，{input} 就需要给出，不能在之后给出。蒙是不行的，你要用它期望的方式用它！
"""
agent = create_react_agent(
    model=base_model, 
    tools=tools, 
    prompt=prompt_template
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "介绍WEC项目"}]}
)

show_message_content(response)

'''
不对
'''
