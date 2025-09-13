from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import os

# 定义天气工具
def get_weather(city: str) -> str:
    """获取给定城市的天气"""
    return f"{city} 的天气总是晴朗！"

os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"
infer_server_url = "http://localhost:11434"
model_name = "qwen3:1.7b"   # 调用 tools

# 初始化 ChatOllama。参数名称是新版本更通用的名称，其变化是langchain_ollama
# 版本变化的提现，并且是先后兼容的，
agent = ChatOllama(
    model=model_name,
    base_url=infer_server_url,
    api_key="none",  # Ollama 本地模型不需要云端 Key 授权
    temperature=0,
    stream=False
)
tools = [get_weather]

# 创建 ReAct 代理
react_agent = create_react_agent(model=agent, tools=tools)

query = {"messages": [{"role": "user", "content": "请获取上海现在的天气"}]}
response = react_agent.invoke(query)

# 打印结果
print(response,"\n")

# 执行工具调用
for element in response.get("messages"):
    print(element, "\n")

'''
有输出
'''
