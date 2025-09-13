from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 定义文档列表
docs = [
    Document(
        page_content="上海的天气通常在夏季炎热潮湿，冬季寒冷干燥。夏季平均气温约 30°C，冬季约 5°C。",
        metadata={"city": "上海", "source": "weather_guide"}
    ),
    Document(
        page_content="北京的天气四季分明，夏季炎热，冬季非常寒冷且有沙尘暴。冬季气温可低至 -10°C。",
        metadata={"city": "北京", "source": "weather_guide"}
    ),
    Document(
        page_content="广州的天气全年温暖，夏季多雨，冬季温和。冬季非常寒冷且有沙尘暴。年平均气温约 22°C。",
        metadata={"city": "广州", "source": "weather_guide"}
    )
]

# 初始化 BM25Retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 1  # 设置返回 top-2 文档

# 定义查询
query = "上海"

# 调用检索器
results = bm25_retriever.invoke(query)

# 输出结果
print("查询:", query)
print("\n检索结果:")
# 1 表示从1开始计数
for i, doc in enumerate(results, 1):
    print(f"\n文档 {i}:")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    
    
"""
输出不符合预期，总是输出广州的哪个doc ！！

不纠结在 BM25Retriever 了，输出总是不符合预期，需要的时候换一个工具，不吊死在一棵树上，况且它不一定用得上
"""
