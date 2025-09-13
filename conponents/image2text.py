from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# 加载模型
model_path = "nanonets/Nanonets-OCR-s"
model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# 定义 OCR 函数
def ocr_image(image_path):
    image = Image.open(image_path)
    prompt = """Extract the text from the image as if reading it naturally."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": f"file://{image_path}"}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

# 提取文字
image_path = "/home/junhui/workspace/AI-Agent/btm.png"
result = ocr_image(image_path)
print(result)

'''
模型下载失败，尝试使用servless inference API 
'''
