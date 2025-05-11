import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
from PIL import Image
from pathlib import Path
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Streamlit page setup
st.set_page_config(page_title="Smart Room Assistant", layout="wide")
st.title("🤖 Smart Room Assistant")
st.caption("🔄 Final YOLOv8 App – Image, Video, Webcam – Streamlit-ready")

# Sidebar config
st.sidebar.header("⚙️ Detection Settings")
model_size = st.sidebar.selectbox("YOLOv8 Model", ["n", "s", "m", "l", "x"], index=4)
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
img_size = st.sidebar.slider("Image Size", 320, 1280, 640, 32)
res_option = st.sidebar.selectbox("Webcam Resolution", ["640x480", "960x720", "1280x720"], index=0)

# Load model
model = YOLO(f"yolov8{model_size}.pt")
class_names = model.names
filter_classes = st.sidebar.multiselect("Filter Classes", options=list(class_names.values()))

# Session and snapshot folder setup
os.makedirs("snapshots", exist_ok=True)
st.session_state.setdefault("detections", [])
st.session_state.setdefault("last_seen", {})
st.session_state.setdefault("last_frame", None)

# Snapshot and reset buttons
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🔁 Reset Detection Log"):
        st.session_state["detections"] = []
        st.success("Detection log cleared.")
with col2:
    if st.button("📸 Save Snapshot Now"):
        frame = st.session_state.get("last_frame", None)
        if frame is not None:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"snapshots/manual_snapshot_{ts}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Snapshot saved: {filename}")
        else:
            st.warning("No frame available yet.")

# Input type selector
input_type = st.selectbox("Select Input Type", ["Image", "Video", "Webcam"])
frame_display = st.image([])

# IMAGE input
if input_type == "Image":
    img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img:
        image = Image.open(img).convert("RGB")
        frame = np.array(image)
        st.image(frame, caption="Uploaded Image", use_column_width=True)

        results = model.predict(frame, conf=confidence, iou=iou_thresh, imgsz=img_size)
        annotated = results[0].plot()
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Detected Objects", use_column_width=True)

        class_ids = results[0].boxes.cls.tolist()
        detected = [class_names[int(i)] for i in class_ids]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for obj in set(detected):
            if filter_classes and obj not in filter_classes:
                continue
            st.success(f"Detected: {obj} at {timestamp}")
            snap_file = f"snapshots/{obj}_{timestamp}.jpg"
            cv2.imwrite(snap_file, annotated)
            st.session_state["detections"].append({
                "object": obj,
                "timestamp": timestamp,
                "snapshot_file": snap_file
            })
        st.session_state["last_frame"] = frame

# VIDEO input
elif input_type == "Video":
    vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
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
            boxes = results[0].boxes
            class_ids = boxes.cls.tolist()
            detected = [class_names[int(i)] for i in class_ids]

            for box, cls in zip(boxes, detected):
                if filter_classes and cls not in filter_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            now = time.time()
            for obj in set(detected):
                if filter_classes and obj not in filter_classes:
                    continue
                if now - st.session_state["last_seen"].get(obj, 0) > 5:
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file = f"snapshots/{obj}_{ts}.jpg"
                    cv2.imwrite(file, frame)
                    st.success(f"Detected: {obj} at {ts}")
                    st.session_state["last_seen"][obj] = now
                    st.session_state["detections"].append({
                        "object": obj, "timestamp": ts, "snapshot_file": file
                    })

            frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

# WEBCAM input
elif input_type == "Webcam":
    st.info("📸 Webcam works in Chrome or Firefox. Allow camera permissions if prompted.")
    width, height = map(int, res_option.split("x"))

    class YOLOProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (width, height))
            results = self.model.predict(img, conf=confidence, iou=iou_thresh, imgsz=img_size)
            boxes = results[0].boxes
            class_ids = boxes.cls.tolist() if boxes else []
            detected = [class_names[int(i)] for i in class_ids]
            now = time.time()

            if boxes:
                for box, cls in zip(boxes, detected):
                    if filter_classes and cls not in filter_classes:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, cls, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            for obj in set(detected):
                if filter_classes and obj not in filter_classes:
                    continue
                if now - st.session_state["last_seen"].get(obj, 0) > 5:
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file = f"snapshots/{obj}_{ts}.jpg"
                    cv2.imwrite(file, img)
                    st.success(f"Detected: {obj} at {ts}")
                    st.session_state["last_seen"][obj] = now
                    st.session_state["detections"].append({
                        "object": obj, "timestamp": ts, "snapshot_file": file
                    })

            st.session_state["last_frame"] = img.copy()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    try:
        webrtc_streamer(
            key="yolo-webrtc",
            video_processor_factory=YOLOProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": width, "height": height}, "audio": False}
        )
    except Exception as e:
        st.error("⚠️ Webcam failed to load.")
        st.exception(e)

# Detection log viewer
if st.session_state["detections"]:
    st.markdown("### 📥 Download Detection Log")
    df = pd.DataFrame(st.session_state["detections"])
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="detection_log.csv", mime="text/csv")
