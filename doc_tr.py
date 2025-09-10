from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import json

# https://mindee.github.io/doctr/index.html

model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)

img_path = "img/report.png"

img_file = DocumentFile.from_images(img_path)

result = model(img_file)

result_dict = result.export()

all_text = []
for page in result_dict["pages"]:
    for block in page["blocks"]:
        for line in block["lines"]:
            line_text = " ".join([word["value"] for word in line["words"]])
            all_text.append(line_text)

full_text = "\n".join(all_text)

print(full_text)

