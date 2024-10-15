import torch
import cv2
import numpy as np
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import functional as F
from torchvision import transforms as T
from supervision import Detections
from PIL import Image

# Load the SSD model
weights = SSD300_VGG16_Weights.COCO_V1  # You can also use SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights)
model.eval()

# COCO class labels (replace these if using a different dataset)
coco_labels = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", 
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", 
    "umbrella", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "N/A", "dining table", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", 
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Create the mapping from class ID to label
model_labels = {i: label for i, label in enumerate(coco_labels)}

def convert_ssd_to_supervision(results):
    bboxes = []
    confidences = []
    class_ids = []

    for i in range(len(results['boxes'])):
        box = results['boxes'][i]
        score = results['scores'][i]
        label = results['labels'][i]

        if score.item() > 0:  # Only consider detections with confidence > 0
            bboxes.append(box.detach().cpu().numpy())
            confidences.append(score.item())
            class_ids.append(label.item())

    bboxes = np.array(bboxes)
    confidences = np.array(confidences)
    class_ids = np.array(class_ids)

    if len(bboxes) == 0:
        return Detections(xyxy=np.empty((0, 4)), confidence=np.empty((0,)), class_id=np.empty((0,)))

    detections = Detections(xyxy=bboxes, confidence=confidences, class_id=class_ids)
    return detections

def draw_boxes(frame, detections):
    for bbox, class_id in zip(detections.xyxy, detections.class_id):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        label = coco_labels[class_id]  # Use class_id to get the class name
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def detect_objects_in_frame_ssd(image, confidence_threshold=0.3):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image
    transform = T.Compose([
        T.Resize((300, 300)),  # SSD300 model input size
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization for the SSD300 model
    ])
    image_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension
    
    # Perform object detection
    with torch.no_grad():
        detections = model(image_tensor)[0]  # Get the detections

    # Extract bounding boxes, labels, and confidence scores
    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()

    class_counts = {}  # To store object counts
    detected_objects = []

    # Draw bounding boxes and count objects
    for i, box in enumerate(boxes):
        if scores[i] >= confidence_threshold:  # Only consider detections above confidence threshold
            class_id = labels[i]
            class_name = coco_labels[class_id]

            # Count the object
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Draw bounding box
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f'{class_name}: {scores[i]:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            detected_objects.append({
                'class_name': class_name,
                'confidence': scores[i],
                'box': (x_min, y_min, x_max, y_max)
            })

    print(f"Detected objects: {class_counts}")  # Print the object counts in console

    # Create a Detections object using the supervision library
    if len(boxes) == 0:
        return Detections(xyxy=np.empty((0, 4)), confidence=np.empty((0,)), class_id=np.empty((0,)))

    # Filter detections based on confidence threshold
    valid_indices = np.where(scores >= confidence_threshold)[0]
    valid_boxes = boxes[valid_indices]
    valid_confidences = scores[valid_indices]
    valid_class_ids = labels[valid_indices]

    # Create and return the Detections object
    detections = Detections(xyxy=valid_boxes, confidence=valid_confidences, class_id=valid_class_ids)
    return detections


# # Example usage
# if __name__ == "__main__":
#     image_path = '/Users/satyapraneeth/Desktop/cv-project/test_image_2.jpg'
#     image = cv2.imread(image_path)
#     detections = detect_objects_in_frame_ssd(image)  # Detect objects in the frame
#     cv2.imshow("Detected Objects", image)  # Show the image with detections
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
