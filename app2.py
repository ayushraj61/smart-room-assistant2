import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Streamlit page config
st.set_page_config(page_title="Smart Room Assistant", layout="wide")
st.title("🤖 Smart Room Assistant - YOLOv8")

# Sidebar settings
st.sidebar.header("Settings")
model_size = st.sidebar.selectbox("Model", ["n", "s", "m", "l", "x"], index=0)
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3, 0.05)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
img_size = st.sidebar.slider("Image Size", 320, 1280, 640, 32)
res_option = st.sidebar.selectbox("Webcam Resolution", ["640x480", "960x720", "1280x720"], index=0)

# Load model
model = YOLO(f"yolov8{model_size}.pt")
class_names = model.names
filter_classes = st.sidebar.multiselect("Filter Classes", options=list(class_names.values()))

# Create snapshot folder
os.makedirs("snapshots", exist_ok=True)
st.session_state.setdefault("detections", [])
st.session_state.setdefault("last_frame", None)

# Snapshot + reset
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🔁 Clear Log"):
        st.session_state["detections"] = []
        st.success("Detection log cleared.")
with col2:
    if st.button("📸 Snapshot"):
        frame = st.session_state.get("last_frame", None)
        if frame is not None:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"snapshots/manual_snapshot_{ts}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Saved: {filename}")
        else:
            st.warning("No frame to snapshot.")

# Input type selector
input_type = st.selectbox("Choose Input", ["Image", "Video", "Webcam"])
frame_display = st.image([])

# Image detection
if input_type == "Image":
    img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if img:
        image = Image.open(img).convert("RGB")
        frame = np.array(image)
        st.image(frame, caption="Original", use_column_width=True)

        results = model.predict(frame, conf=confidence, iou=iou_thresh, imgsz=img_size)
        annotated = results[0].plot()
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected", use_column_width=True)

        class_ids = results[0].boxes.cls.tolist()
        detected = [class_names[int(i)] for i in class_ids]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for obj in set(detected):
            if filter_classes and obj not in filter_classes:
                continue
            snap_file = f"snapshots/{obj}_{timestamp}.jpg"
            cv2.imwrite(snap_file, annotated)
            st.session_state["detections"].append({
                "object": obj, "timestamp": timestamp, "snapshot_file": snap_file
            })

        st.session_state["last_frame"] = annotated

# Video detection
elif input_type == "Video":
    vid = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if vid:
        with open("temp_video.mp4", "wb") as f:
            f.write(vid.read())
        cap = cv2.VideoCapture("temp_video.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            st.session_state["last_frame"] = frame.copy()

            results = model.predict(frame, conf=confidence, iou=iou_thresh, imgsz=img_size)
            annotated = results[0].plot()

            boxes = results[0].boxes
            class_ids = boxes.cls.tolist()
            detected = [class_names[int(i)] for i in class_ids]
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            for obj in set(detected):
                if filter_classes and obj not in filter_classes:
                    continue
                snap_file = f"snapshots/{obj}_{timestamp}.jpg"
                cv2.imwrite(snap_file, annotated)
                st.session_state["detections"].append({
                    "object": obj, "timestamp": timestamp, "snapshot_file": snap_file
                })

            frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        cap.release()

# Webcam detection using webrtc
elif input_type == "Webcam":
    st.info("Enable webcam access in browser. Use Chrome/Firefox.")

    width, height = map(int, res_option.split("x"))

    class YOLOProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (width, height))

            results = self.model.predict(img, conf=confidence, iou=iou_thresh, imgsz=img_size)

            if results and results[0].boxes:
                annotated = results[0].plot()
            else:
                annotated = img

            st.session_state["last_frame"] = annotated
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": width, "height": height}, "audio": False}
    )

# Detection log
if st.session_state["detections"]:
    st.markdown("### 📋 Detection Log")
    df = pd.DataFrame(st.session_state["detections"])
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Log as CSV", data=csv, file_name="detection_log.csv", mime="text/csv")

