from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2

img_path = "img/handwritten.jpg"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


img = Image.open(img_path).convert("RGB")

pixel_values = processor(images=img, return_tensors = "pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

print("\n\n" + text.strip())

