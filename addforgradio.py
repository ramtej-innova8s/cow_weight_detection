import gradio as gr
import torch
import cv2
import numpy as np
import tempfile
import os

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to estimate cow weight
def estimate_weight(bbox):
    xmin, ymin, xmax, ymax = bbox
    area = (xmax - xmin) * (ymax - ymin)
    weight = area * 0.001  # Simplified formula
    return weight

# Function to process the video, draw bounding boxes, and estimate weights
def process_video(video):
    # Create a temporary file to save the processed video
    output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0]

        for *box, conf, cls in detections:
            if model.names[int(cls)] == 'cow' and conf > 0.55:
                weight = estimate_weight(box)
                label = f'{model.names[int(cls)]} {conf:.2f} | {weight:.2f} kg'
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    
    # Return the path to the processed video
    return output_video_path

# Create the Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),
    outputs=gr.Video(),
    title="Cow Weight Estimation",
    description="Upload a video of cows to estimate their weight and get the video back with bounding boxes."
)

# Launch the interface
iface.launch(share=True)