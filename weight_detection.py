import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to estimate cow weight based on bounding box area (this is a placeholder function)
def estimate_weight(bbox):
    xmin, ymin, xmax, ymax = bbox
    area = (xmax - xmin) * (ymax - ymin)  # Area of the bounding box
    # Example: Linear relation between area and weight (you should replace this with a more accurate model)
    weight = area * 0.001  # Weight per square pixel (placeholder value)
    return weight

def process_video(video_path, output_path):
    print(f"Processing {video_path}...")
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None

    # Get the video writer initialized to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # Check if the video writer initialized successfully
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}.")
        cap.release()
        return None

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the current frame
        results = model(frame)

        # Extract detections
        detections = results.xyxy[0]  # xyxy format: xmin, ymin, xmax, ymax, confidence, class

        # Draw bounding boxes for cows and estimate weight
        for *box, conf, cls in detections:
            if model.names[int(cls)] == 'cow':
                weight = estimate_weight(box)  # Estimate the weight
                label = f'{model.names[int(cls)]} {conf:.2f} | {weight:.2f} kg'
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame with the detections and weight estimates to the output video
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"Processed video saved to {output_path}")
    return output_path