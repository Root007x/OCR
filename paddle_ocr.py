from paddleocr import PaddleOCR

img_path = "img/handwritten.jpg"

# Initialize PaddleOCR with the updated parameter
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# Load and process image
result = ocr.predict(img_path)

# Extract text
text = " ".join([line[1][0] for line in result[0]])
print(text)