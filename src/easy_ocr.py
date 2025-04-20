

import easyocr
import os
import cv2
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

reader = easyocr.Reader(['en'])  


cropped_dir = "cropped_plates"


def show_image_with_text(img, text):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Text: {text}")
    plt.axis("off")
    plt.show()


for file in os.listdir(cropped_dir):
    if file.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(cropped_dir, file)
        image = cv2.imread(img_path)

        
        results = reader.readtext(image)

        
        plate_text = " ".join([res[1] for res in results])
        print(f"📸 Image: {file} -> 🏷️ Plate: {plate_text}")

        
        show_image_with_text(image, plate_text)



cropped_dir = "/content/cropped_plates"
xml_dir = "/content/dataset/training/images"
output_labels = "/content/dataset/labels.txt"

os.makedirs(os.path.dirname(output_labels), exist_ok=True)

def extract_plate_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

   
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name:  
            return name.strip()
    return None

with open(output_labels, "w") as f:
    for image_file in os.listdir(cropped_dir):
        if image_file.endswith((".jpg", ".png")):
            base_name = "_".join(image_file.split("_")[:2])  
            xml_path = os.path.join(xml_dir, f"{base_name}.xml")

            if os.path.exists(xml_path):
                plate_text = extract_plate_from_xml(xml_path)
                if plate_text:
                    f.write(f"{image_file} {plate_text}\n")
                else:
                    print(f"No plate text found in: {xml_path}")
            else:
                print(f"Missing XML for: {image_file}")
