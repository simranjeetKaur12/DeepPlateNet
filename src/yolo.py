
import pytesseract
import os
import glob
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")


model.train(
    data="/content/dataset_final/data.yaml",  
    epochs=50,  
    imgsz=640,  
    batch=16,   

)




model =  YOLO("best.pt") 


image_folder = "/content/video_images/*.png"  
image_paths = glob.glob(image_folder)


for img_path in image_paths:
    img = cv2.imread(img_path)  
    results = model(img)  
    for r in results:
        for box in r.boxes:
            xyxy = [int(i) for i in box.xyxy[0].tolist()]  
            conf = box.conf[0].item() 

            # Draw bounding box
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 3)  # Red box
            cv2.putText(img, f"Confidence: {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the image
    import cv2_imshow
    cv2_imshow(img)
    cv2.waitKey(0)

cv2.destroyAllWindows()




image_dir = "/content/dataset/training/images"  
cropped_dir = "cropped_plates"  
os.makedirs(cropped_dir, exist_ok=True)  


model = YOLO("best.pt")  

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

       
        results = model(image)

        for i, result in enumerate(results):
            for box in result.boxes.xyxy:  
                x1, y1, x2, y2 = map(int, box)  
                cropped_plate = image[y1:y2, x1:x2]  

                
                cropped_filename = f"{filename.split('.')[0]}_plate_{i}.jpg"
                cropped_path = os.path.join(cropped_dir, cropped_filename)
                cv2.imwrite(cropped_path, cropped_plate)

                print(f"Saved cropped plate: {cropped_filename}")

print("All license plates cropped and saved.")


plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['train/box_loss'], label="Box Loss")
plt.plot(df['epoch'], df['train/cls_loss'], label="Class Loss")
plt.plot(df['epoch'], df['val/box_loss'], label="Val Box Loss", linestyle="dashed")
plt.plot(df['epoch'], df['val/cls_loss'], label="Val Class Loss", linestyle="dashed")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("YOLO Loss Curve")
plt.legend()
plt.show()




df = pd.read_csv("runs/detect/train/results.csv")


plt.plot(df['epoch'], df['metrics/mAP50(B)'], label="mAP@50", marker='o')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label="mAP@50-95", marker='s')


plt.xlabel("Epochs")
plt.ylabel("mAP")
plt.title("YOLO Model Performance")
plt.legend()
plt.grid(True)

plt.show()



