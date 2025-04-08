from ultralytics import YOLOE
import cv2
import threading
import time
from PIL import Image
import tkinter as tk
from tkinter import simpledialog

# Initialize a YOLOE model
model = YOLOE("yoloe-v8s-seg.pt", task="segment")  # or select yoloe-11s/m-seg.pt for different sizes

# Shared text prompt
names = []
names_updated = False  # Flag to track if names were updated

def periodic_update_names():
    """Thread function to periodically ask the user to update the names list using a pop-up window."""
    global names, names_updated
    while True:
        time.sleep(5)  # Wait for 5 seconds
        # Create a pop-up window for user input
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        new_item = simpledialog.askstring("Update Names", "Enter a new item to track (or leave blank to skip):")
        root.destroy()  # Destroy the tkinter window after input

        if new_item:  # If the user entered a new item
            if new_item not in names:
                names.append(new_item)
                names_updated = True  # Set the flag to indicate names were updated
                print(f"Updated names: {names}")

# Start a thread to periodically ask for user input via GUI
thread = threading.Thread(target=periodic_update_names, daemon=True)
thread.start()

# Open the camera feed
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or replace with the camera index

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Update model classes if names were updated
    if names_updated:
        model.set_classes(names, model.get_text_pe(names))
        names_updated = False  # Reset the flag

    # Convert frame to PIL Image
    frame_detect = Image.fromarray(frame)

    # Perform prediction on the current frame
    results = model.predict(frame_detect)

    # Draw segmentation masks on the frame
    for result in results[0]:
        # Draw bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for cls, box in zip(classes, boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw class label
            label = model.names[int(cls)]
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame with segmentation masks
    cv2.imshow("YOLOE Segmentation", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()