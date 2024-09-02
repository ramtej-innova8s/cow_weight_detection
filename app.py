import streamlit as st
import torch
import cv2
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to estimate cow weight
def estimate_weight(bbox):
    xmin, ymin, xmax, ymax = bbox
    area = (xmax - xmin) * (ymax - ymin)
    weight = area * 0.001
    return weight

st.title('Cow Weight Estimation')
video_file = st.file_uploader("Upload a video", type=["mp4"])

if video_file is not None:
    st.video(video_file)
    video_path = './uploaded_video.mp4'
    with open(video_path, 'wb') as f:
        f.write(video_file.getbuffer())

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = './output_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
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

    st.success("Processing completed! Download the output video below.")
    with open(output_path, 'rb') as f:
        st.download_button('Download Output Video', f, file_name='output_video.mp4')