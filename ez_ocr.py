import easyocr

img_path = "img/handwritten.jpg"

reader = easyocr.Reader(['en'])
results = reader.readtext(img_path)
for bbox, text, conf in results:
    print(text, conf)