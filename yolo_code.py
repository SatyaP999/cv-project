import torch
import cv2 as cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
from supervision.detection.core import Detections

model = YOLO('yolov8n.pt')




def convert_yolov8_to_supervision(results):
    bboxes = []
    confidences = []
    class_ids = []

    for result in results[0].boxes:
        if result.conf.item() > 0:  # Only consider detections with confidence > 0
            bboxes.append(result.xyxy[0].cpu().numpy())  # [x1, y1, x2, y2]
            confidences.append(result.conf.cpu().numpy())  # Confidence score
            class_ids.append(int(result.cls.cpu().numpy()))  # Class ID

    bboxes = np.array(bboxes)
    confidences = np.array(confidences).ravel()  # Ensure it's a 1D array
    class_ids = np.array(class_ids)

    if bboxes.shape[0] == 0:
        return Detections(xyxy=np.empty((0, 4)), confidence=np.empty((0,)), class_id=np.empty((0,)))

    detections = Detections(xyxy=bboxes, confidence=confidences, class_id=class_ids)
    return detections




def draw_boxes(frame, detections):
    for bbox, class_id in zip(detections.xyxy, detections.class_id):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        label = model.names[class_id]
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        
def detect_objects_in_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)  # Detect objects

    detections = convert_yolov8_to_supervision(results)  # Convert to supervision format
    draw_boxes(frame, detections)  # Draw bounding boxes
    
    # Debugging: Check the type of detections
    print(f"Detections Type: {type(detections)}")  # Should be a Detections object
    
    return detections  

    

# image_path = input()
# detect_an_image(image_path)
    