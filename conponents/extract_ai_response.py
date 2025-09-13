
from langchain_core.messages import HumanMessage, AIMessage

# print(type(element))
# <class 'langchain_core.messages.human.HumanMessage'>
# <class 'langchain_core.messages.ai.AIMessage'>
# content 是上述两个对象的属性，所以可以通过 dot 获取
def extract_aimessage_content(res):
    for element in res.get("messages"):
        if isinstance(element, AIMessage):
            content = element.content
            if '<think>' in content:
                content = content.split('</think>')[-1].strip()
            return content
    return None

def show_message_content(res):
    for element in res.get("messages"):
        if isinstance(element, HumanMessage) or isinstance(element, AIMessage):
            content = element.content
            if '<think>' in content:
                content = content.split('</think>')[-1].strip()
            print(">>> \n", content)
    return None


if __name__ == "__main__":
    
    """
    下面的输出形式不是由LLM决定的 ，而是由 langgraph ，langchain 决定的。
    """
    raw_response = {
        'messages': [HumanMessage(content='介绍你自己？', 
                                additional_kwargs={}, 
                                response_metadata={}, 
                                id='1e2b8af6-472c-43e3-bb29-2fbc7ffe2b53'), 
                    AIMessage(content='<think>\n嗯，用户让我介绍自己。首先，我需要确定用户的需求是什么。他们可能只是想了解我的功能，或者想测试我的知识。不过，作为AI助手，我需要保持专业和友好。\n\n接下来，我要考虑如何组织回答。应该包括我的身份、功能、应用场景，以及我的局限性。这样用户能全面了解我。同时，要避免使用技术术语，保持口语化。\n\n然后，检查是否有遗漏的信息。比如，我的训练数据截止到2023年10月，所以提到这一点可以增加可信度。另外，强调我的局限性，比如无法进行实时计算，这样用户知道我的能力边界。\n\n还要注意语气，保持亲切，避免生硬。可能需要加入一些表情符号或感叹号，让回答更生动。不过，用户可能希望保持专业，所以需要平衡。\n\n最后，确保回答结构清晰，分点说明，方便用户阅读。同时，结尾用友好的问候，让用户有参与感。检查有没有错别字或语法错误，保持简洁明了。\n</think>\n\n嗨！我是你的AI助手，一个由阿里巴巴集团研发的智能助手，专门帮助用户解决问题、提供信息和进行交流。我能够处理各种任务，比如回答问题、写文章、做计算、推荐内容，甚至陪你聊天。\n\n我的能力包括：\n1. **知识库**：基于2023年10月的训练数据，涵盖广泛领域（如科学、文化、技术等）。\n2. **多语言支持**：能用中文、英文、日文、韩语等语言交流。\n3. **功能**：支持数学计算、编程帮助、创意写作、时间管理、学习辅导等。\n4. **限制**：无法进行实时计算、无法体验物理世界、无法进行复杂决策。\n\n我希望能成为你生活中有用的伙伴，无论是学习、工作还是娱乐，都希望有帮助！如果你有任何问题，随时告诉我～ 😊', 
                            additional_kwargs={}, 
                            response_metadata={'model': 'qwen3:1.7b', 
                                                'created_at': '2025-07-30T06:42:14.59444914Z', 
                                                'done': True, 'done_reason': 'stop', 
                                                'total_duration': 33061919787, 
                                                'load_duration': 2274896373, 
                                                'prompt_eval_count': 11, 
                                                'prompt_eval_duration': 503229188, 
                                                'eval_count': 403, 
                                                'eval_duration': 30281871486, 
                                                'model_name': 'qwen3:1.7b'}, 
                            id='run--cab190b1-995a-486f-b873-ef8eae712127-0', 
                            usage_metadata={'input_tokens': 11, 'output_tokens': 403, 'total_tokens': 414})
                    ]
        }

    show_message_content(raw_response)


'''
done
map 的 value 中有两个对象构成的 list
'''
