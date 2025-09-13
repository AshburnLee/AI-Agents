from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

from huggingface_hub import login
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
login(token=hf_token)

# 给出你的 llm
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # BART 模型，专门为文本摘要任务优化
llm = HuggingFacePipeline(pipeline=summarizer)

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [("system", "下面文档中的各个关键字之间的关系是什么？请给出这个关系让我很好地理解:\\n\\n{context}")]
)

# Instantiate chain
chain = create_stuff_documents_chain(llm, prompt)

# prepare docs
# 这里可以是任何Document对象，比如:

# url = "https://ollama.com/"
url = "https://python.langchain.com/docs/concepts/"
loader = WebBaseLoader(url)
docs = loader.load()
print(docs)

# Invoke chain
result = chain.invoke({"context": docs})
print(result)

'''
done
'''
