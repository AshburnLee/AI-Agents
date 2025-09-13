from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


# 定义文档列表
docs = [
    Document(page_content="上海的天气通常在夏季炎热潮湿，冬季寒冷干燥。夏季平均气温约 30°C，冬季约 5°C。", metadata={"city": "上海", "source": "weather_guide"}),
    Document(page_content="北京的天气四季分明，夏季炎热，冬季非常寒冷且有沙尘暴。冬季气温可低至 -10°C。", metadata={"city": "北京", "source": "weather_guide"}),
    Document(page_content="广州的天气全年温暖，夏季多雨，冬季温和。年平均气温约 22°C。", metadata={"city": "广州", "source": "weather_guide"})
]

# 初始化 sentence-transformers 嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# 初始化 Chroma 向量存储
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="weather_collection"
)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 定义检索工具
def retrieve_weather_info(query: str) -> str:
    """根据查询检索天气信息"""
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

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
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
基于以下检索到的信息：{context}
以 JSON 格式响应，包含工具名称和参数：
{
  "tool": "retrieve_weather_info",
  "parameters": {
    "answer": "<基于检索信息的回答>"
  }
}
"""),
    ("human", "{input}")
])

# 创建 ReAct 代理
tools = [retrieve_weather_info]
# agent = create_react_agent(model=model, tools=tools, state_modifier=prompt_template, debug=True)
agent = create_react_agent(
    model=base_model,
    tools=tools
)

# 测试查询
# query = "上海"
query = {"messages": [{"role": "user", "content": "上海的天气"}]}

response = agent.invoke(query)
print(response)


'''
这是一个通过本地database 构建检索工具的 实例.
返回复合预期：
HumanMessage(content='上海的天气', additional_kwargs={}, response_metadata={}, id='80831fd0-3b89-496f-a230-ed536aa372bc'),
AIMessage(content='<think>\n好的，用户问的是上海的天气。我需要调用retrieve_weather_info这个函数来获取天气信息。函数的参数需要一个query，用户已经明确提到了上海，所以参数应该是"上海"。接下来要确保参数正确无误，然后生成对应的工具调用JSON。不需要其他额外的信息，直接返回结果吧。\n</think>\n\n'
tool_calls=[{'name': 'retrieve_weather_info', 'args': {'query': '上海'}, 'id': 'af9b3ad1-035c-4028-8313-10f3d5fec348', 'type': 'tool_call'}],
ToolMessage(content='上海的天气通常在夏季炎热潮湿，冬季寒冷干燥。夏季平均气温约 30°C，冬季约 5°C。', name='retrieve_weather_info', 
AIMessage(content='<think>\n好的，用户之前询问了上海的天气，我给出了回复。现在需要处理用户的最新查询。用户可能想进一步了解天气情况，或者有其他相关问题。首先，我需要确认用户是否需要更多详细信息，比如具体日期的天气预报，或者是否需要其他城市的天气信息。不过根据当前的对话历史，用户只提到了上海的天气，没有后续问题。因此，我应该保持回答简洁，确认已提供信息，并邀请用户继续提问。同时，检查是否有需要补充的内容，比如是否需要建议携带的衣物，或者是否需要提醒注意的天气现象。但根据之前的回复，用户可能已经满意，所以保持回复友好且信息完整即可。\n</think>\n\n上海的天气通常在夏季炎热潮湿，冬季寒冷干燥。夏季平均气温约 30°C，冬季约 5°C。若需具体日期的天气预报或更多细节，可进一步补充说明！',
'''

'''
搞清楚：invoke，prompt_template
'''

