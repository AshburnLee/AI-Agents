from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# # load image from the IAM database (actually this model is meant to be used on printed text)
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# # url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUlv30cRN4GKkwpFdE8X7dvDffA-8wegemH9p7X2y3ZtjkvSMNUUteBKh_UjpuFKpzOpg&usqp=CAU'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image_path = "/home/junhui/workspace/AI-Agent/btm.png"  # Replace with your image path
image = Image.open(image_path)

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)
