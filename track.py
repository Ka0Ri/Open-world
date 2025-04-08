from ultralytics import YOLOE
import cv2
from PIL import Image
# Initialize a YOLOE model
model = YOLOE("yoloe-v8s-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt
names = ["apple", "cup", "computer mouse", "bottle", "pen"]
model.set_classes(names, model.get_text_pe(names))
image = Image.open("frame.jpg")
results = model.predict(image, show=True, conf=0.5, iou=0.5, device="cuda:0")
results[0].plot()  # Display the image with detections