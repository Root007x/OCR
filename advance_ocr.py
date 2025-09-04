from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

# Load processor and model
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Load image
img_path = "img/handwritten.jpg"
image = Image.open("your_image.jpg")

# Prepare inputs
inputs = processor(images=image, text="Describe the text in this image.", return_tensors="pt")

# Generate output ids
outputs = model.generate(**inputs)

# Decode outputs
result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)

