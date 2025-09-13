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

# 创建 ReAct 代理
tools = []
agent = create_react_agent(model=base_model, tools=tools)

# 测试查询
query = {"messages": [{"role": "user", "content": "介绍你自己？"}]}
response = agent.invoke(query)
show_message_content(response)

'''
返回复合预期
'''

'''
ReAct是一个 agent 框架

这个code是创建了一个 react agent。除了react agent，如何创建最最基本的 agent？
或者说，from langgraph.prebuilt import create_react_agent 还有哪些可用的agent类型
'''
