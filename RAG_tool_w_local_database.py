import datasets  # from hugginface
from langchain_core.documents import Document
# from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os


'''
Load raw dataset,
convert them into Document object,
store those Document in a list
'''
# Load the dataset, this is a dataset card from hugging face, datasets library can load it.
'''
这里的数据内容会被下载到本地  ~/.cache/huggingface/hub/
'''
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]
print(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",  # 支持中文的多语言模型
    model_kwargs={"device": "cpu"}  # 可改为 "cuda" 如果有 GPU
)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="guest_collection"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
'''
configure our Retiever using those Documents
Create the Retriever Tool
this tool processes the query and output the most relavent guest information
'''
# # BM25Retriever is a powerful text retieval alg that does not need embeddings
# bm25_retriever = BM25Retriever.from_documents(docs)

# this tool expects a 'query' as input
def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = retriever.invoke(query)
    print(results,"\n\n\n")
    if results:
        print("guest found")
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."

# name and description helps our agent to understand how to use those tools
guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)


print("step3")
'''
step3: integrate Tools with Agent
'''
# Generate the chat interface, including the tools
my_hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=my_hf_token,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
ragger = builder.compile()

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = ragger.invoke({"messages": messages})


print("🎩 ragger's Response:")
# 不符合预期
print(response)
# print(response['messages'][-1].content)

# ground truth
# ground_truth = extract_text('Ada Lovelace')
# print(ground_truth)

# results = bm25_retriever.invoke('Ada Lovelace')
# print(results)

"""
结果 AIMassage 是空白的，error
可能是 BM25Retriever，改为 sentence-transformers ，工具不好用就扔

改为 embedding_model 实现的retrever 后，相同的输出，返回到AIMessage是空的 ,
实际上它并没有调用 tool
"""
