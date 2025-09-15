
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

# 初始化 ChatOllama 模型
infer_server_url = "http://localhost:11434"
model_name = "qwen3:1.7b"
llm = ChatOllama(
    model=model_name,
    base_url=infer_server_url,
    api_key="none",  # Ollama 本地模型不需要云端 API 密钥
    temperature=0,
    stream=False
)

# 第一步提示模板：将日文翻译成中文
translate_template = """你是一个专业翻译。将以下日文文本翻译成简体中文，保持准确和自然。

日文: {japanese_text}

中文: """
translate_prompt = PromptTemplate(input_variables=["japanese_text"], template=translate_template)

# 第二步提示模板：提取食物名词并解释
food_noun_template = """你是一个实物分析专家。分析以下中文文本，提取所有与食物相关的名词（例如食材、菜肴、饮料等），并解释各个名词（50字以内）。尽可能详细地输出。如果没有食物相关名词，返回“未找到食物相关名词”。

中文文本: {chinese_text}

如果没有食物相关名词:
未找到食物相关名词
"""
food_noun_prompt = PromptTemplate(input_variables=["chinese_text"], template=food_noun_template)

## 创建两个处理链
translate_chain = translate_prompt | llm | StrOutputParser()
food_noun_chain = food_noun_prompt | llm | StrOutputParser()

## 串联两个链：将翻译结果作为食物名词提取的输入
full_chain = RunnableSequence(
    translate_chain,
    lambda x: {"chinese_text": x.split("中文:")[-1].strip()},  # 提取中文翻译部分
    food_noun_chain
)

# 定义函数，翻译日文并分析食物名词
def translate_and_analyze_food_nouns(japanese_text):
    try:
        # 调用串联的链
        result = full_chain.invoke({"japanese_text": japanese_text})
        # 返回翻译结果和食物名词分析
        translation = translate_chain.invoke({"japanese_text": japanese_text}).split("中文:")[-1].strip()
        return {
            "翻译": translation,
            "食物名词分析": result.strip()
        }
    except Exception as e:
        return {"错误": f"处理失败: {str(e)}"}

# 测试
if __name__ == "__main__":
    # 测试用例1：包含食物名词
    test_text1 = "今日は水曜日で、私はトウモロコシ、麺、ご飯、そしてコンピューターを食べたいです。"
    print("。。。测试日文1:", test_text1)
    result1 = translate_and_analyze_food_nouns(test_text1)
    print("。。。翻译:", result1["翻译"])
    print("食物名词分析:")
    print(result1["食物名词分析"])
    print("\n")

