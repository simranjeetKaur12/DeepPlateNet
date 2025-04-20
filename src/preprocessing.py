import os
import xml.etree.ElementTree as ET
import cv2
import pandas as pd

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    img_size = root.find("size")
    img_width, img_height = int(img_size.find("width").text), int(img_size.find("height").text)

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin, ymin = int(bbox.find("xmin").text), int(bbox.find("ymin").text)
        xmax, ymax = int(bbox.find("xmax").text), int(bbox.find("ymax").text)
        objects.append({"label": label, "bbox": (xmin, ymin, xmax, ymax)})

    return img_width, img_height, objects


def process_dataset(main_directory):
    dataset = []  
    image_extensions = [".jpg", ".png", ".jpeg"]  

    # Loop through all state directories
    for state in os.listdir(main_directory):
        state_path = os.path.join(main_directory, state)
        if not os.path.isdir(state_path):  
            continue

       
        for file in os.listdir(state_path):
            if file.endswith(".xml"):  
                xml_path = os.path.join(state_path, file)  
                img_file = None

                
                for ext in image_extensions:
                    potential_img = os.path.join(state_path, file.replace(".xml", ext))
                    if os.path.exists(potential_img):
                        img_file = potential_img
                        break  

                if img_file:  
                    img_width, img_height, objects = parse_xml(xml_path)
                    dataset.append({
                        "image_path": img_file,
                        "xml_path": xml_path,
                        "image_size": (img_width, img_height),
                        "objects": objects
                    })
                else:
                    print(f"Warning: No matching image found for {xml_path}")

    return dataset  


dataset_path = "/content/dataset/training/images"  


dataset_info = process_dataset(dataset_path)


print(f"Total images processed: {len(dataset_info)}")
print("Sample Data:", dataset_info[:3])  


df = pd.DataFrame(dataset_info)
df_exploded = df.explode("objects")
df_exploded = pd.concat([df_exploded.drop(['objects'], axis=1), df_exploded['objects'].apply(pd.Series)], axis=1)
df_exploded[['xmin', 'ymin', 'xmax', 'ymax']] = pd.DataFrame(df_exploded['bbox'].tolist(), index=df_exploded.index)
df_exploded.drop('bbox', axis=1, inplace=True)
df_exploded.tail()

import cv2
import matplotlib.pyplot as plt

def show_image_with_bbox(image_path, bbox, label):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show image
    plt.imshow(img)
    plt.axis("off")
    plt.show()


sample_row = df_exploded.iloc[0]  
show_image_with_bbox(sample_row["image_path"], (sample_row["xmin"], sample_row["ymin"], sample_row["xmax"], sample_row["ymax"]), sample_row["label"])

import os

def convert_to_yolo_format(df, output_dir, class_mapping):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for _, row in df.iterrows():
        image_path = row["image_path"]
        label = row["label"]
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        img_width, img_height = row["image_size"]

        
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        class_id = class_mapping.get(label, 0)  

        
        image_name = os.path.basename(image_path).split('.')[0]
        yolo_file = os.path.join(output_dir, f"{image_name}.txt")

        with open(yolo_file, "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

class_mapping = {
    "plate": 0  
}

convert_to_yolo_format(df_exploded, "/content/yolo_labels", class_mapping)

import yaml


data_yaml = {
    "train": "/content/dataset_final//images",   
    "val": "/content/dataset_final/training/images",     
    "nc": 1, 
    "names": ["plate"]  
}


yaml_path = "/content/dataset_final/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"✅ data.yaml created at: {yaml_path}")
