
import streamlit as st
from ultralytics import YOLO
import cv2
import time

# Set Streamlit page configuration
st.set_page_config(page_title="Smart Room Assistant", layout="centered")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure this model file is present

# Function to trigger a chatbot response
def trigger_chatbot(detected_classes):
    if "person" in detected_classes:
        return "👋 Hello! I see someone in the room. How can I assist you?"
    elif "cell phone" in detected_classes:
        return "📱 Looks like you're using a phone. Need tech support?"
    elif "laptop" in detected_classes:
        return "💻 Working hard or hardly working? Let me know if you need assistance."
    else:
        return "✅ No actionable object detected."

# UI Layout
st.title("🤖 Smart Room Assistant")
st.markdown("This assistant detects objects and triggers AI responses based on what it sees.")

start_camera = st.checkbox("Turn On Camera")

FRAME_WINDOW = st.image([])

# Initialize webcam
cap = None
if start_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot access the webcam.")
    else:
        st.success("✅ Camera is live.")

        while start_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            # Run YOLO model
            results = model(frame)[0]
            labels = [model.names[int(cls)] for cls in results.boxes.cls]

            # Draw boxes
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Show webcam image
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Trigger chatbot and show response
            response = trigger_chatbot(labels)
            st.info(response)

        cap.release()
