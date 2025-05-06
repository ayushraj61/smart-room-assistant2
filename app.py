import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from gtts import gTTS
from PIL import Image
import cv2
import numpy as np

# Set Streamlit page title
st.set_page_config(page_title="Smart Room Assistant", layout="centered")

# Title
st.title("🧠 Smart Room Assistant")
st.write("Upload an image to detect objects and trigger smart responses.")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Action responses
ACTIONS = {
    "laptop": "Launching work mode.",
    "bed": "Time to rest. Activating sleep assistant.",
    "chair": "Ergonomic check: Sit upright!",
    "tv": "Entertainment mode ready.",
    "bottle": "Stay hydrated! Drink some water.",
}

def speak(text, lang="en"):
    """Convert text to speech and play it."""
    tts = gTTS(text=text, lang=lang)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    audio_file = open(temp_audio.name, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")
    audio_file.close()
    os.remove(temp_audio.name)

# Image uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(img_cv)

    # Annotate detections
    annotated_img = results[0].plot()

    # Show annotated image
    st.image(annotated_img, caption="Detected Objects", use_column_width=True)

    detected_labels = set()

    # Get detected object labels
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            detected_labels.add(label)

    # Show and speak responses
    for label in detected_labels:
        response = ACTIONS.get(label, f"Unrecognized object: {label}.")
        st.write(f"**{label.capitalize()}** detected → {response}")
        speak(response)
