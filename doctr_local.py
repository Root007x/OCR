from doctr.models import ocr_predictor, db_resnet50, crnn_mobilenet_v3_large
from doctr.io import DocumentFile


detection_model = db_resnet50(pretrained=False, pretrained_backbone = False)
detection_model.from_pretrained(path_or_url="model/db_resnet50.pt", map_location = False)


recognize_model = crnn_mobilenet_v3_large(pretrained=False, pretrained_backbone = False) 
recognize_model.from_pretrained(path_or_url="model/crnn_mobilenet_v3_large_pt.pt", map_location = False)

model = ocr_predictor(det_arch=detection_model, reco_arch=recognize_model, pretrained=False)

img_file_path = "img/report.png"

img_data = DocumentFile.from_images(img_file_path)

result_dict = model(img_data)

result_dict = result_dict.export()


all_text = []
for page in result_dict["pages"]:
    for block in page["blocks"]:
        for line in block["lines"]:
            line_text = " ".join([word["value"] for word in line["words"]])
            all_text.append(line_text)

full_text = "\n".join(all_text)

print(full_text)


