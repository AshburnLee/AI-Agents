from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# 初始化 ChatOllama
infer_server_url = "http://localhost:11434"
model_name = "qwen3:1.7b"
llm = ChatOllama(
    model=model_name,
    base_url=infer_server_url,
    api_key="none",
    temperature=0,
    stream=False
)

# 定义提示模板，要求模型识别并解释食物相关名词
template = """你是一个语言分析专家。分析以下英文文本，识别所有与食物相关的名词（例如食材、菜肴、饮料等），并为每个名词提供简短的解释（20-50字）。以结构化格式返回结果，列出每个名词及其解释。如果没有食物相关名词，返回“未找到食物相关名词”。

英文文本: {input_text}

输出格式:
- 名词: [食物名词]
  解释: [简短解释]
- 名词: [食物名词]
  解释: [简短解释]
...

如果没有食物相关名词:
未找到食物相关名词
"""
prompt = PromptTemplate(input_variables=["input_text"], template=template)

# 创建 RunnableSequence
food_noun_chain = prompt | llm

# 定义函数，分析文本并返回食物名词及其解释
def explain_food_nouns(input_text):
    try:
        # 调用 RunnableSequence，传入输入文本
        result = food_noun_chain.invoke({"input_text": input_text})
        # 提取模型输出的文本内容
        return result.content.strip()
    except Exception as e:
        return f"错误: {str(e)}"

# 测试
if __name__ == "__main__":
    # 测试用例1：包含食物名词
    test_text1 = "I ate an apple and some bread with coffee this morning."
    print("测试文本1:", test_text1)
    print("结果:")
    print(explain_food_nouns(test_text1))
    print("=======================\n")

    # 测试用例2：不包含食物名词
    test_text2 = "The cat jumped onto the table."
    print("测试文本2:", test_text2)
    print("结果:")
    print(explain_food_nouns(test_text2))
    
