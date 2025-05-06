import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from gtts import gTTS
import os
import tempfile
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

ACTIONS = {
    "laptop": "Launching work mode.",
    "bed": "Time to rest. Activating sleep assistant.",
    "chair": "Ergonomic check: Sit upright!",
    "tv": "Entertainment mode ready.",
    "bottle": "Stay hydrated! Drink some water.",
}

st.title("🧠 Smart Room Assistant (Webcam Live Detection)")

start = st.button("Start Camera", key="start_btn")

def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

if start:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam not found.")
    else:
        st.info("Webcam started. Press STOP to end.")
        prev_labels = set()
        FRAME_WINDOW = st.image([])
        stop_btn = st.button("Stop Camera", key="stop_btn")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            # Detect objects
            results = model(frame)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame)

            # Get labels
            labels = set()
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    label = model.names[cls]
                    labels.add(label)

            # Detect new objects
            new_labels = labels - prev_labels
            for label in new_labels:
                message = ACTIONS.get(label, f"{label} detected.")
                st.write(f"**{label.capitalize()}** → {message}")
                speak(message)
                time.sleep(1)

            prev_labels = labels

            if stop_btn:
                break

        cap.release()
             
