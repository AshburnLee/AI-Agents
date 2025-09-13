from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import os

from langchain_ollama import ChatOllama
os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"
infer_server_url = "http://localhost:11434/"
model_name = "qwen3:1.7b"
llm = ChatOllama(
    model=model_name,
    base_url=infer_server_url,
    api_key="none",
    temperature=0,
    stream=False
)

# 定义提示模板
prompt = ChatPromptTemplate.from_template(
    "请将以下内容总结为不超过100字的中文摘要:\n{context}"
)

# 创建 Stuff 链
stuff_chain = create_stuff_documents_chain(llm, prompt)

docs = [
    Document(page_content="文章1：介绍AI技术在医疗领域的应用，包括诊断和治疗。"),
    Document(page_content="文章2：讨论AI在金融领域的风险管理应用。"),
]

# 调用链
result = stuff_chain.invoke({"context": docs})
print(result)

'''
如期输出
'''
