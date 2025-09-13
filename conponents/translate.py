
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch

## HF servless inference API
'''
# 假设 HUGGINGFACEHUB_API_TOKEN 已通过环境变量设置
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# 初始化 LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
# 初始化 ChatHuggingFace
chat = ChatHuggingFace(llm=llm, verbose=True)
'''

## Huggingface 模型加载方式
'''
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# 初始化本地 Qwen2.5-1.5B-Instruct 模型
model_name = "qwen3:1.7b "
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用 float16 节省内存
    device_map="auto"  # 自动分配到 GPU/CPU
)
# 创建 HuggingFace pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    device_map="auto"
)
# 初始化 LangChain 的 HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)
'''


## ollama 框架的模型加载方式
# 初始化 ChatOllama
from langchain_ollama import ChatOllama
infer_server_url = "http://localhost:11434"
model_name = "qwen3:1.7b"
llm = ChatOllama(
    model=model_name,
    base_url=infer_server_url,
    api_key="none",
    temperature=0,
    stream=False
)

# 定义翻译提示模板
template = """You are a professional translator. Translate the following Japanese text into Chinese accurately and naturally:

Japanese: {japanese_text}

Chinese: """
prompt = PromptTemplate(input_variables=["japanese_text"], template=template)

# Create a RunnableSequence (replacing old style LLMChain)
translation_chain = prompt | llm

# Translation function
def translate_japanese_to_chinese(japanese_text):
    # Run the sequence and extract the content
    result = translation_chain.invoke({"japanese_text": japanese_text})
    # Extract the translated text (assuming the model returns the full response)
    translation = result.content.split("Chinese:")[-1].strip()
    return translation, result

# 示例使用
if __name__ == "__main__":
    japanese_sentence = "今日は水曜日で、私はトウモロコシ、麺、ご飯、そしてコンピューターを食べたいです"
    chinese_translation, raw_res = translate_japanese_to_chinese(japanese_sentence)
    print(f"English: {japanese_sentence}")
    print(f"Chinese: {chinese_translation}")
    
    print(f"Raw output: {raw_res}")

# expect: 今天是星期三，我想吃玉米、面条、米饭和电脑。


'''
done
'''
