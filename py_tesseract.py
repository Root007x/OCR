from PIL import Image
import pytesseract as pt
import cv2

# # adding the tesseract.exe path
pt.pytesseract.tesseract_cmd = r"tesseract\tesseract.exe"


img_path = "img/handwritten.jpg"
img = cv2.imread(img_path)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# noise_removal = cv2.medianBlur(threshold_img, 3)


img_to_text = pt.image_to_string(noise_removal)

print(img_to_text)
