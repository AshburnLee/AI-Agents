

from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
import os

from huggingface_hub import login
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
login(token=hf_token)

# 1. 模型
from transformers import pipeline
from langchain.llms import HuggingFacePipeline  # 第一次使用时会下载内容
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # 由 Facebook 开发的 BART 模型，专门为文本摘要任务优化
llm = HuggingFacePipeline(pipeline=summarizer)


# from langchain_ollama import ChatOllama
# os.environ["http_proxy"] = "http://127.0.0.1:11434"
# os.environ["https_proxy"] = "http://127.0.0.1:11434"
# infer_server_url = "http://localhost:11434/"
# model_name = "qwen3:1.7b"
# llm = ChatOllama(
#     model=model_name,
#     base_url=infer_server_url,
#     api_key="none",
#     temperature=0,
#     stream=False
# )

'''
Sample url: https://ollama.com/
Output from qwen3:1.7b: 

Ollama's new app allows users to download and run large language models like DeepSeek-R1, Qwen 3, and Gemma 3 on macOS, Windows, and Linux, with features including model exploration, download, and access to resources like the blog, docs, GitHub, Discord, and X (Twitter). © 2025 Ollama Inc.

对于简单网页是可以识别并总结的。
'''

# 1. 网页加载 需要Proxy
os.environ["http_proxy"] = "192.168.31.130:7890"
os.environ["https_proxy"] = "192.168.31.130:7890"
url = "https://ollama.com/"  # 用户输入
# url = "https://www.hpcwire.com/off-the-wire/mitac-partners-with-daiwabo-to-expand-server-distribution-across-japan/?utm_source=twitter&utm_medium=social&utm_term=hpcwire&utm_content=0ca462ed-d256-453f-924d-49a1d20354c1"
loader = WebBaseLoader(url)
docs = loader.load()
print(docs)
print("====================")

# 3. 拆分文档
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# 4. 构建摘要链
# “stuff”, “map_reduce”, and “refine”.
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.invoke(split_docs) # invoke

# 5. 输出
print(summary)

'''
done
'''
