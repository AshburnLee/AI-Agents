from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.messages import SystemMessage, HumanMessage

# 初始化 ChatOllama
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

# 定义提示模板
'''
{input} 表示占位符，用于表示动态变量, 允许在提示文本中插入运行时提供的变量值。
LangChain 在invoke 时会将字典中的 input 值替换到 {input} 占位符处。
from_messages 是工厂方法，"{input}" 会被动态解析。
'''
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，回答用户的问题。"),
    ("human", "{input}")
])
print(prompt_template)

"""
HumanMessage(content="{input}") 只是个普通字符串，要让第二种写法支持动态变量替换，你需要用 HumanMessagePromptTemplate 来包裹内容！
"""
prompt_template = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("你是一个智能助手，回答用户的问题。"),
        HumanMessagePromptTemplate.from_template("{input}")  #  这样，{input} 会被正确识别并替换
    ]
)
print(prompt_template)

'''
两种方式的 prompt_template 对象完全相同
'''


# 创建基本代理 chain
'''
LangChain 的 链式操作（Chain） 语法
是 LangChain 的 Runnable 协议的一部分, 将提示模板、模型和解析器等组件组合成一个工作流
'''
basic_agent = prompt_template | base_model | StrOutputParser()
# basic_agent = prompt_template | base_model 

# 测试查询
query = {"input": "介绍欧洲上最流行的运动？"}
response = basic_agent.invoke(query)

# 显示结果
print("基本代理响应:", response)


"""
done
"""