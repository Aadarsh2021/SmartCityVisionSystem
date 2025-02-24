import sys
import cv2
import torch
import numpy as np
from screeninfo import get_monitors

# Add YOLOv7 directory to Python path
yolov7_dir = 'D:/smart_city_system/models/yolov7'
sys.path.append(yolov7_dir)

# Import the YOLOv7 model and utilities
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

# Load the YOLOv7 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load('D:/smart_city_system/models/yolov7/runs/train/exp24/weights/best.pt', map_location=device)
model.eval()  # Set the model to evaluation mode

# Get screen size dynamically
monitor = get_monitors()[0]
screen_width, screen_height = monitor.width, monitor.height

# Open IP webcam stream
ip_camera_url = 'http://192.168.1.10:8080/video'
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error opening video stream")
    exit()

# Class names based on your dataset mapping
class_names = ['Car', 'Bus', 'Truck', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Other person',
               'Rider', 'Traffic light', 'Traffic sign', 'Trailer', 'Train', 'Other vehicle']

# Counter for total people detected
total_people_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading the stream.")
        break

    # Resize and prepare frame for detection
    img_tensor = cv2.resize(frame, (416, 416))
    img_tensor = torch.from_numpy(img_tensor).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # Change to CxHxW format

    # Perform detection
    with torch.no_grad():
        pred = model(img_tensor)[0]

    # Apply Non-Max Suppression
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.4)

    # Initialize a counter for the current frame
    current_people_count = 0

    if pred[0] is not None and len(pred[0]):
        # Rescale boxes back to original image
        pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], frame.shape).round()
        
        for *xyxy, conf, cls in pred[0]:
            label = f'{class_names[int(cls)]} {conf:.2f}'

            # Only count people
            if class_names[int(cls)] in ['Pedestrian', 'Other person']:
                current_people_count += 1
                total_people_count += 1
                # Draw bounding box and label
                frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display counts on the frame
    cv2.putText(frame, f'Current Count: {current_people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Total Count: {total_people_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed frame
    cv2.imshow('Crowd Counting with YOLOv7', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
