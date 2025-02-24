import sys
import cv2
import torch
import time
from screeninfo import get_monitors

# Add YOLOv7 directory to Python path
yolov7_dir = 'D:/smart_city_system/models/yolov7'
sys.path.append(yolov7_dir)

# Import the YOLOv7 model and utilities
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

# Load the YOLOv7 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = attempt_load('D:/smart_city_system/models/yolov7/runs/train/exp24/weights/best.pt', map_location=device)
model.eval()  # Set the model to evaluation mode

# Get screen size dynamically
monitor = get_monitors()[0]  # Get the main monitor
screen_width, screen_height = monitor.width, monitor.height
print(f"Screen resolution: {screen_width}x{screen_height}")

# IP Webcam video stream URL
video_path = 'http://192.168.137.51:8080/video'  # IP Webcam URL

# Class names based on your dataset mapping
class_names = [
    'Car', 'Bus', 'Truck', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Other person',
    'Rider', 'Traffic light', 'Traffic sign', 'Trailer', 'Train', 'Other vehicle'
]

# Open the video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video source: {video_path}")
    sys.exit()

# Set a confidence threshold (adjust to control detection quality)
confidence_threshold = 0.25
nms_iou_threshold = 0.4

# Resize frame while maintaining aspect ratio
def resize_frame(frame, target_size=(416, 416)):
    h, w = frame.shape[:2]
    ratio = min(target_size[0] / h, target_size[1] / w)
    new_size = (int(w * ratio), int(h * ratio))
    resized_frame = cv2.resize(frame, new_size)
    top = (target_size[0] - new_size[1]) // 2
    bottom = target_size[0] - new_size[1] - top
    left = (target_size[1] - new_size[0]) // 2
    right = target_size[1] - new_size[0] - left
    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_frame, ratio, (top, left)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading the video.")
        break

    # Resize frame while maintaining aspect ratio
    frame_resized, ratio, (top, left) = resize_frame(frame)

    # Convert frame to a tensor and perform detection
    img_tensor = torch.from_numpy(frame_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=confidence_threshold, iou_thres=nms_iou_threshold)

    # Draw boxes on the original frame
    if pred[0] is not None and len(pred[0]):
        pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], frame.shape).round()

        for *xyxy, conf, cls in pred[0]:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the processed frame in a window
    cv2.imshow('YOLOv7 Object Detection - Live Feed', frame)

    # Increment frame count
    frame_count += 1

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
