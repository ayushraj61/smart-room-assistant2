import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import threading
import json
from datetime import datetime
from PIL import Image
import os
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from collections import defaultdict
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    FACE_BACKEND = "dlib"
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    FACE_BACKEND = None

# ─── OpenCV DNN Face Detection/Recognition Fallback ───
OPENCV_FACE_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_models")

def _download_opencv_face_models():
    """Download OpenCV face detection and recognition ONNX models if not present."""
    import urllib.request
    os.makedirs(OPENCV_FACE_MODELS_DIR, exist_ok=True)
    models = {
        "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    }
    for fname, url in models.items():
        fpath = os.path.join(OPENCV_FACE_MODELS_DIR, fname)
        if not os.path.exists(fpath):
            try:
                urllib.request.urlretrieve(url, fpath)
            except Exception as e:
                return False
    return True

@st.cache_resource
def load_opencv_face_models():
    """Load OpenCV FaceDetectorYN and FaceRecognizerSF."""
    if not _download_opencv_face_models():
        return None, None
    det_path = os.path.join(OPENCV_FACE_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    rec_path = os.path.join(OPENCV_FACE_MODELS_DIR, "face_recognition_sface_2021dec.onnx")
    try:
        detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.7, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
        return detector, recognizer
    except Exception:
        return None, None

if not FACE_RECOGNITION_AVAILABLE:
    _cv_face_detector, _cv_face_recognizer = load_opencv_face_models()
    if _cv_face_detector is not None:
        FACE_RECOGNITION_AVAILABLE = True
        FACE_BACKEND = "opencv"

# ─── Page Config ───
st.set_page_config(page_title="Smart Room Assistant", layout="wide")
st.title("Smart Room Assistant - YOLOv8")

# ─── Constants ───
FACES_DIR = "faces"
FACE_DB_FILE = os.path.join(FACES_DIR, "face_db.json")
FACE_MATCH_TOLERANCE = 0.45

# ─── Model Loading (cached, loads only once) ───
@st.cache_resource
def load_model(model_size, use_onnx=False):
    """Load and cache the YOLO model. Optionally export to ONNX for faster CPU inference."""
    model = YOLO(f"yolov8{model_size}.pt")
    if use_onnx:
        onnx_path = f"yolov8{model_size}.onnx"
        if not os.path.exists(onnx_path):
            model.export(format="onnx", imgsz=640, half=False, simplify=True)
        model = YOLO(onnx_path)
    return model

# ─── Face Database ───
os.makedirs(FACES_DIR, exist_ok=True)

def load_face_db():
    """Load face database from JSON file."""
    if os.path.exists(FACE_DB_FILE):
        with open(FACE_DB_FILE, "r") as f:
            db = json.load(f)
        # Convert lists back to numpy arrays
        for name in db:
            db[name] = [np.array(emb) for emb in db[name]]
        return db
    return {}

def save_face_db(db):
    """Save face database to JSON file."""
    serializable = {}
    for name in db:
        serializable[name] = [emb.tolist() for emb in db[name]]
    with open(FACE_DB_FILE, "w") as f:
        json.dump(serializable, f)

@st.cache_data
def get_face_db_cached(_hash):
    """Cache face DB in memory, reload when file changes."""
    return load_face_db()

def get_face_db():
    """Get face DB with cache invalidation based on file modification time."""
    if os.path.exists(FACE_DB_FILE):
        mtime = os.path.getmtime(FACE_DB_FILE)
    else:
        mtime = 0
    return get_face_db_cached(mtime)

def _opencv_detect_faces(image_rgb):
    """Detect faces using OpenCV DNN. Returns list of (top, right, bottom, left) and aligned faces."""
    h, w = image_rgb.shape[:2]
    _cv_face_detector.setInputSize((w, h))
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, faces = _cv_face_detector.detect(img_bgr)
    if faces is None:
        return [], [], img_bgr
    locations = []
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        top, right, bottom, left = y, x + fw, y + fh, x
        locations.append((top, right, bottom, left))
    return locations, faces, img_bgr

def _opencv_encode_face(img_bgr, face_raw):
    """Get 128-dim face embedding using OpenCV FaceRecognizerSF."""
    aligned = _cv_face_recognizer.alignCrop(img_bgr, face_raw)
    return _cv_face_recognizer.feature(aligned).flatten()

def register_face(name, image_rgb):
    """Register a face from an RGB image. Returns (success, message)."""
    if FACE_BACKEND == "dlib":
        face_locations = face_recognition.face_locations(image_rgb, model="hog")
        if len(face_locations) == 0:
            return False, "No face detected in this image."
        if len(face_locations) > 1:
            return False, f"Multiple faces ({len(face_locations)}) detected. Please use an image with only one face."
        encodings = face_recognition.face_encodings(image_rgb, face_locations)
        if len(encodings) == 0:
            return False, "Could not encode face. Try a clearer image."
        encoding = encodings[0]
        top, right, bottom, left = face_locations[0]
    else:
        locations, faces_raw, img_bgr = _opencv_detect_faces(image_rgb)
        if len(locations) == 0:
            return False, "No face detected in this image."
        if len(locations) > 1:
            return False, f"Multiple faces ({len(locations)}) detected. Please use an image with only one face."
        encoding = _opencv_encode_face(img_bgr, faces_raw[0])
        top, right, bottom, left = locations[0]

    db = load_face_db()
    if name not in db:
        db[name] = []
    db[name].append(encoding)
    save_face_db(db)

    # Save the face image for reference
    face_img_path = os.path.join(FACES_DIR, f"{name}_{len(db[name])}.jpg")
    face_crop = image_rgb[top:bottom, left:right]
    Image.fromarray(face_crop).save(face_img_path)

    return True, f"Face registered for '{name}' ({len(db[name])} image(s) total)."

def _match_encoding(encoding, known_encodings, known_names):
    """Match a face encoding against known encodings. Works for both backends."""
    if FACE_BACKEND == "dlib":
        distances = face_recognition.face_distance(known_encodings, encoding)
        if len(distances) > 0:
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            if best_distance <= FACE_MATCH_TOLERANCE:
                return known_names[best_idx], 1.0 - best_distance
    else:
        # OpenCV: use cosine similarity (higher = better match)
        best_score = 0
        best_name = None
        for i, known_enc in enumerate(known_encodings):
            score = _cv_face_recognizer.match(
                encoding.reshape(1, -1).astype(np.float32),
                known_enc.reshape(1, -1).astype(np.float32),
                cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best_score = score
                best_name = known_names[i]
        if best_score >= 0.363:  # OpenCV recommended cosine threshold
            return best_name, best_score
    return "Unknown", 0.0

def recognize_faces_in_frame(frame_rgb):
    """Find and recognize faces in a frame. Returns list of (name, location, confidence)."""
    db = get_face_db()
    if not db:
        return []

    # Build known arrays
    known_encodings = []
    known_names = []
    for name, embeddings in db.items():
        for emb in embeddings:
            known_encodings.append(np.array(emb))
            known_names.append(name)

    if FACE_BACKEND == "dlib":
        face_locations = face_recognition.face_locations(frame_rgb, model="hog")
        if not face_locations:
            return []
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name, conf = _match_encoding(encoding, known_encodings, known_names)
            results.append((name, location, conf))
        return results
    else:
        locations, faces_raw, img_bgr = _opencv_detect_faces(frame_rgb)
        if not locations:
            return []
        results = []
        for loc, face_raw in zip(locations, faces_raw):
            encoding = _opencv_encode_face(img_bgr, face_raw)
            name, conf = _match_encoding(encoding, known_encodings, known_names)
            results.append((name, loc, conf))
        return results

def draw_face_labels(frame_bgr, face_results):
    """Draw face recognition labels on a BGR frame."""
    for name, (top, right, bottom, left), conf in face_results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw box
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

        # Draw label background
        label = f"{name} ({conf:.0%})" if name != "Unknown" else "Unknown"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame_bgr, (left, top - label_size[1] - 10),
                      (left + label_size[0] + 5, top), color, -1)
        cv2.putText(frame_bgr, label, (left + 2, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame_bgr

# ─── Sidebar Settings ───
st.sidebar.header("Settings")

st.sidebar.subheader("Model")
model_size = st.sidebar.selectbox("Model Size", ["n", "s", "m", "l", "x"], index=1,
                                   help="n=fastest/least accurate, x=slowest/most accurate. 's' is best for Streamlit Cloud.")
use_onnx = st.sidebar.checkbox("Use ONNX (faster CPU inference)", value=False,
                                help="Exports model to ONNX format for 2-3x faster inference on CPU.")
use_tta = st.sidebar.checkbox("Test-Time Augmentation", value=False,
                               help="Runs inference with augmentations for better accuracy. Slower but more accurate.")

st.sidebar.subheader("Detection")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
                                help="Higher = fewer false positives, lower = detect more objects.")
iou_thresh = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
img_size = st.sidebar.slider("Inference Size", 320, 1280, 640, 32,
                              help="Lower = faster, higher = more accurate for small objects.")

st.sidebar.subheader("Preprocessing")
enable_clahe = st.sidebar.checkbox("Auto Contrast (CLAHE)", value=True,
                                    help="Enhances contrast for better detection in poor lighting.")
enable_denoise = st.sidebar.checkbox("Denoise", value=False,
                                      help="Reduces noise in image. Helps in low-light but slower.")

st.sidebar.subheader("Webcam / Video")
res_option = st.sidebar.selectbox("Resolution", ["640x480", "960x720", "1280x720"], index=0)
webcam_infer_size = st.sidebar.slider("Webcam Inference Size", 160, 640, 320, 32,
                                       help="Lower = faster webcam FPS. 320 is good for real-time.")
frame_skip = st.sidebar.slider("Frame Skip (process every Nth)", 1, 10, 2,
                                help="Skip frames for faster processing. 1=every frame, 3=every 3rd frame.")
enable_tracking = st.sidebar.checkbox("Object Tracking (BoT-SORT)", value=True,
                                       help="Track objects across frames with persistent IDs.")

if FACE_RECOGNITION_AVAILABLE:
    st.sidebar.subheader("Face Recognition")
    enable_face_recognition = st.sidebar.checkbox("Enable Face Recognition", value=True,
                                                   help="Recognize registered faces when 'person' is detected.")
else:
    enable_face_recognition = False

# Load model
model = load_model(model_size, use_onnx)
class_names = model.names
filter_classes = st.sidebar.multiselect("Filter Classes", options=list(class_names.values()))

# ─── Session State ───
os.makedirs("snapshots", exist_ok=True)
st.session_state.setdefault("detections", [])
st.session_state.setdefault("last_frame", None)
st.session_state.setdefault("object_counts", defaultdict(int))
st.session_state.setdefault("video_running", False)
st.session_state.setdefault("heatmap", None)

# ─── Preprocessing Pipeline ───
def preprocess_frame(frame):
    """Apply image preprocessing to improve detection accuracy."""
    processed = frame.copy()

    if enable_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if enable_denoise:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 6, 6, 7, 21)

    return processed

# ─── Letterbox Resize (preserve aspect ratio) ───
def letterbox_resize(frame, target_size):
    """Resize frame preserving aspect ratio with letterboxing."""
    h, w = frame.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    x_offset = (tw - nw) // 2
    y_offset = (th - nh) // 2
    canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
    return canvas

# ─── Run Detection / Tracking ───
def run_detection(frame, infer_size=None, track=False):
    """Run YOLO detection or tracking on a frame."""
    size = infer_size or img_size

    if track and enable_tracking:
        results = model.track(
            frame, conf=confidence, iou=iou_thresh, imgsz=size,
            persist=True, tracker="botsort.yaml", verbose=False
        )
    else:
        results = model.predict(
            frame, conf=confidence, iou=iou_thresh, imgsz=size,
            augment=use_tta, verbose=False
        )
    return results

# ─── Extract Detection Info ───
def extract_detections(results):
    """Extract detection details including confidence and track IDs."""
    detections = []
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return detections

    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    coords = boxes.xyxy.tolist()
    track_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(class_ids)

    for cls_id, conf, coord, track_id in zip(class_ids, confidences, coords, track_ids):
        name = class_names[int(cls_id)]
        if filter_classes and name not in filter_classes:
            continue
        detections.append({
            "object": name,
            "confidence": round(conf, 3),
            "track_id": int(track_id) if track_id is not None else None,
            "bbox": [round(c, 1) for c in coord],
        })
    return detections

# ─── Annotate Frame with Extra Info + Face Recognition ───
def annotate_frame(frame, results, fps=None, do_face_recognition=True):
    """Draw detections + FPS + object count + face labels overlay."""
    annotated = results[0].plot()

    # Face recognition on person detections
    face_results = []
    if do_face_recognition and enable_face_recognition:
        dets = extract_detections(results)
        person_dets = [d for d in dets if d["object"] == "person"]

        if person_dets:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = recognize_faces_in_frame(frame_rgb)
            if face_results:
                annotated = draw_face_labels(annotated, face_results)

    # FPS counter
    if fps is not None:
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Object count summary
    dets = extract_detections(results)
    counts = defaultdict(int)
    for d in dets:
        counts[d["object"]] += 1

    # Add recognized face names to counts
    for name, _, _ in face_results:
        if name != "Unknown":
            counts[name] = counts.get(name, 0) + 1

    y_pos = 60 if fps is None else 70
    for obj, count in sorted(counts.items()):
        cv2.putText(annotated, f"{obj}: {count}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_pos += 25

    return annotated, face_results

# ─── Update Heatmap ───
def update_heatmap(frame_shape, results):
    """Accumulate detection positions into a heatmap."""
    if st.session_state["heatmap"] is None:
        st.session_state["heatmap"] = np.zeros(frame_shape[:2], dtype=np.float32)

    heatmap = st.session_state["heatmap"]
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes.xyxy.tolist():
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_shape[1], x2), min(frame_shape[0], y2)
            heatmap[y1:y2, x1:x2] += 1
    st.session_state["heatmap"] = heatmap

# ─── Log Detections (deduplicated by unique object+identity) ───
def log_detections(detections, timestamp, face_results=None):
    """Log unique detections. Deduplicates by track_id OR by object+identity combo."""
    logged_ids = {d.get("track_id") for d in st.session_state["detections"] if d.get("track_id") is not None}
    logged_combos = {(d["object"], d.get("identity", "-")) for d in st.session_state["detections"]}

    # Build face name lookup from face_results
    face_names = {}
    if face_results:
        for name, (top, right, bottom, left), conf in face_results:
            if name != "Unknown":
                face_names[(top, right, bottom, left)] = (name, conf)

    for det in detections:
        # Try to match face to this person detection
        identified_name = None
        if det["object"] == "person" and face_names:
            px1, py1, px2, py2 = det["bbox"]
            for (ftop, fright, fbottom, fleft), (fname, fconf) in face_names.items():
                if fleft >= px1 and ftop >= py1 and fright <= px2 and fbottom <= py2:
                    identified_name = fname
                    break

        identity = identified_name or "-"
        combo = (det["object"], identity)

        # Skip if this track_id was already logged
        if det["track_id"] is not None and det["track_id"] in logged_ids:
            continue
        # Skip if this object+identity combo already logged (for non-tracked mode)
        if det["track_id"] is None and combo in logged_combos:
            continue

        st.session_state["detections"].append({
            "object": det["object"],
            "identity": identity,
            "confidence": det["confidence"],
            "track_id": det["track_id"],
            "bbox": str(det["bbox"]),
            "timestamp": timestamp,
        })
        if det["track_id"] is not None:
            logged_ids.add(det["track_id"])
        logged_combos.add(combo)

    # Update counts
    counts = defaultdict(int)
    for det in detections:
        counts[det["object"]] += 1
    st.session_state["object_counts"] = counts

# ══════════════════════════════════════════════
# FACE REGISTRATION UI (in sidebar expander)
# ══════════════════════════════════════════════
if not FACE_RECOGNITION_AVAILABLE:
    st.sidebar.info("Face recognition is not available (dlib/face_recognition not installed).")

if FACE_RECOGNITION_AVAILABLE:
  with st.sidebar.expander("Register / Manage Faces", expanded=False):
    reg_name = st.text_input("Person's Name", key="face_reg_name")

    reg_method = st.radio("Registration Method", ["Upload Photos", "Webcam Capture"], key="reg_method", horizontal=True)

    if reg_method == "Upload Photos":
        reg_images = st.file_uploader(
            "Upload face images", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="face_reg_upload",
            help="Upload clear face photos. One face per image."
        )

        if st.button("Register Uploaded Photos") and reg_name:
            if not reg_images:
                st.warning("Please upload at least one image.")
            else:
                for img_file in reg_images:
                    img = Image.open(img_file).convert("RGB")
                    img_array = np.array(img)
                    success, msg = register_face(reg_name.strip(), img_array)
                    if success:
                        st.success(msg)
                    else:
                        st.error(f"{img_file.name}: {msg}")
                get_face_db_cached.clear()

    elif reg_method == "Webcam Capture":
        st.session_state.setdefault("webcam_captures", [])

        ANGLE_GUIDES = ["Look straight", "Turn slightly left", "Turn slightly right", "Tilt up slightly", "Tilt down slightly"]
        current_step = len(st.session_state["webcam_captures"])

        if current_step < len(ANGLE_GUIDES):
            st.info(f"Step {current_step + 1}/{len(ANGLE_GUIDES)}: **{ANGLE_GUIDES[current_step]}**")
            cam_image = st.camera_input(f"Capture - {ANGLE_GUIDES[current_step]}", key=f"cam_capture_{current_step}")

            if cam_image is not None:
                img = Image.open(cam_image).convert("RGB")
                img_array = np.array(img)
                # Verify face is detected before accepting
                if FACE_BACKEND == "dlib":
                    face_locs = face_recognition.face_locations(img_array, model="hog")
                else:
                    face_locs, _, _ = _opencv_detect_faces(img_array)
                if len(face_locs) == 1:
                    st.session_state["webcam_captures"].append(img_array)
                    st.success(f"Captured! ({current_step + 1}/{len(ANGLE_GUIDES)})")
                    st.rerun()
                elif len(face_locs) == 0:
                    st.error("No face detected. Try again with better lighting.")
                else:
                    st.error("Multiple faces detected. Only one face per capture.")
        else:
            st.success(f"All {len(ANGLE_GUIDES)} angles captured!")

        if st.session_state["webcam_captures"] and reg_name:
            if st.button(f"Register {len(st.session_state['webcam_captures'])} Captures"):
                for img_array in st.session_state["webcam_captures"]:
                    success, msg = register_face(reg_name.strip(), img_array)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                st.session_state["webcam_captures"] = []
                get_face_db_cached.clear()
                st.rerun()

        if st.session_state["webcam_captures"]:
            if st.button("Reset Captures"):
                st.session_state["webcam_captures"] = []
                st.rerun()

    st.markdown("---")

    # Show registered faces
    db = get_face_db()
    if db:
        st.markdown("**Registered People:**")
        for name, embeddings in db.items():
            st.write(f"{name} ({len(embeddings)} photo(s))")
            face_imgs = [f for f in os.listdir(FACES_DIR) if f.startswith(f"{name}_") and f.endswith(".jpg")]
            if face_imgs:
                for fimg in face_imgs[:3]:
                    st.image(os.path.join(FACES_DIR, fimg), width=60)

            if st.button(f"Delete {name}", key=f"del_{name}"):
                db_fresh = load_face_db()
                if name in db_fresh:
                    del db_fresh[name]
                    save_face_db(db_fresh)
                    for f in os.listdir(FACES_DIR):
                        if f.startswith(f"{name}_") and f.endswith(".jpg"):
                            os.remove(os.path.join(FACES_DIR, f))
                    get_face_db_cached.clear()
                    st.success(f"Deleted '{name}'.")
                    st.rerun()
    else:
        st.info("No faces registered yet.")

# ─── Controls Row ───
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("Clear Log"):
        st.session_state["detections"] = []
        st.session_state["object_counts"] = defaultdict(int)
        st.session_state["heatmap"] = None
        st.success("Cleared.")
with col2:
    if st.button("Snapshot"):
        frame = st.session_state.get("last_frame", None)
        if frame is not None:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"snapshots/manual_snapshot_{ts}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Saved: {filename}")
        else:
            st.warning("No frame available.")

# ─── Live Object Count Dashboard ───
if st.session_state["object_counts"]:
    counts = st.session_state["object_counts"]
    st.markdown("### Live Object Count")
    count_cols = st.columns(min(len(counts), 6))
    for i, (obj, count) in enumerate(sorted(counts.items())):
        with count_cols[i % len(count_cols)]:
            st.metric(label=obj, value=count)

# ─── Input Selector ───
input_type = st.selectbox("Choose Input", ["Image", "Video", "Webcam"])

# ══════════════════════════════════════════════
# IMAGE DETECTION
# ══════════════════════════════════════════════
if input_type == "Image":
    img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if img:
        image = Image.open(img).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess
        processed = preprocess_frame(frame)

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

        # Detect
        results = run_detection(processed)
        annotated, face_results = annotate_frame(processed, results)

        with col_det:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected", use_container_width=True)

        # Log
        dets = extract_detections(results)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_detections(dets, timestamp, face_results)
        st.session_state["last_frame"] = annotated

        # Show recognized faces summary
        if face_results:
            recognized = [r for r in face_results if r[0] != "Unknown"]
            unknown = [r for r in face_results if r[0] == "Unknown"]
            if recognized:
                names = ", ".join([f"{r[0]} ({r[2]:.0%})" for r in recognized])
                st.success(f"Recognized: {names}")
            if unknown:
                st.warning(f"{len(unknown)} unknown face(s) detected.")

        # Show per-object crops
        if dets:
            st.markdown("#### Detected Object Crops")
            crop_cols = st.columns(min(len(dets), 5))
            for i, det in enumerate(dets):
                x1, y1, x2, y2 = map(int, det["bbox"])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    with crop_cols[i % len(crop_cols)]:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                 caption=f"{det['object']} ({det['confidence']:.0%})",
                                 use_container_width=True)

# ══════════════════════════════════════════════
# VIDEO DETECTION
# ══════════════════════════════════════════════
elif input_type == "Video":
    vid = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if vid:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vid.read())
        tmp_path = tmp.name
        tmp.close()

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        st.info(f"Video: {total_frames} frames @ {video_fps:.0f} FPS")

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            start_btn = st.button("Start Processing")
        with vcol2:
            stop_btn = st.button("Stop Processing")

        frame_display = st.empty()
        progress_bar = st.progress(0)
        fps_display = st.empty()
        live_log_display = st.empty()

        if stop_btn:
            st.session_state["video_running"] = False

        if start_btn:
            st.session_state["video_running"] = True
            frame_count = 0
            prev_time = time.time()
            face_frame_interval = 5
            last_face_results = []

            while cap.isOpened() and st.session_state["video_running"]:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                progress_bar.progress(min(frame_count / max(total_frames, 1), 1.0))

                if frame_count % frame_skip != 0:
                    continue

                # Preserve aspect ratio
                h, w = frame.shape[:2]
                target_w, target_h = 960, int(960 * h / w)
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                processed = preprocess_frame(frame)
                results = run_detection(processed, track=True)

                # FPS
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                # Face recognition on every Nth processed frame
                do_face = (frame_count // frame_skip) % face_frame_interval == 0
                annotated, face_results = annotate_frame(processed, results, fps=fps,
                                                          do_face_recognition=do_face)
                if face_results:
                    last_face_results = face_results
                elif not do_face and last_face_results:
                    annotated = draw_face_labels(annotated, last_face_results)

                dets = extract_detections(results)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_detections(dets, timestamp, last_face_results)

                update_heatmap(frame.shape, results)

                st.session_state["last_frame"] = annotated
                frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                fps_display.text(f"Processing FPS: {fps:.1f}")

                # Live log update (every 30 processed frames to avoid lag)
                if (frame_count // frame_skip) % 30 == 0 and st.session_state["detections"]:
                    df_live = pd.DataFrame(st.session_state["detections"])
                    live_log_display.dataframe(df_live, use_container_width=True)

            cap.release()
            os.unlink(tmp_path)
            st.success("Video processing complete.")

            if st.session_state["heatmap"] is not None:
                st.markdown("#### Detection Heatmap")
                heatmap = st.session_state["heatmap"]
                heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), use_container_width=True)

# ══════════════════════════════════════════════
# WEBCAM DETECTION (optimized)
# ══════════════════════════════════════════════
elif input_type == "Webcam":
    st.info("Enable webcam access in your browser. Works best in Chrome/Firefox.")

    width, height = map(int, res_option.split("x"))

    # Pre-load face DB for webcam thread
    _face_db_for_webcam = get_face_db()
    _known_encodings_webcam = []
    _known_names_webcam = []
    for name, embeddings in _face_db_for_webcam.items():
        for emb in embeddings:
            _known_encodings_webcam.append(emb)
            _known_names_webcam.append(name)

    class YOLOProcessor(VideoProcessorBase):
        # Shared detection log accessible from main thread
        detection_log = []
        detection_lock = threading.Lock()

        def __init__(self):
            self.model = model
            self.frame_count = 0
            self.last_annotated = None
            self.prev_time = time.time()
            self.fps = 0.0
            self.lock = threading.Lock()
            self.last_face_results = []
            self.face_interval = 10  # Face recognition every 10th processed frame
            self._logged_ids = set()
            self._logged_combos = set()

        def _recognize_faces(self, frame_rgb):
            """Thread-safe face recognition using pre-loaded DB."""
            if not _known_encodings_webcam:
                return []

            if FACE_BACKEND == "dlib":
                face_locations = face_recognition.face_locations(frame_rgb, model="hog")
                if not face_locations:
                    return []
                face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                results = []
                for encoding, location in zip(face_encodings, face_locations):
                    name, conf = _match_encoding(encoding, _known_encodings_webcam, _known_names_webcam)
                    results.append((name, location, conf))
                return results
            else:
                locations, faces_raw, img_bgr = _opencv_detect_faces(frame_rgb)
                if not locations:
                    return []
                results = []
                for loc, face_raw in zip(locations, faces_raw):
                    encoding = _opencv_encode_face(img_bgr, face_raw)
                    name, conf = _match_encoding(encoding, _known_encodings_webcam, _known_names_webcam)
                    results.append((name, loc, conf))
                return results

        def _draw_face_labels(self, frame, face_results):
            """Draw face labels on frame."""
            for name, (top, right, bottom, left), conf in face_results:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({conf:.0%})" if name != "Unknown" else "Unknown"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (left, top - label_size[1] - 10),
                              (left + label_size[0] + 5, top), color, -1)
                cv2.putText(frame, label, (left + 2, top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return frame

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # Frame skipping: reuse last result
            if self.frame_count % frame_skip != 0 and self.last_annotated is not None:
                return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")

            # Preprocess
            if enable_clahe:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe_obj.apply(l)
                lab = cv2.merge([l, a, b])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # YOLO detection
            if enable_tracking:
                results = self.model.track(
                    img, conf=confidence, iou=iou_thresh,
                    imgsz=webcam_infer_size, persist=True,
                    tracker="botsort.yaml", verbose=False
                )
            else:
                results = self.model.predict(
                    img, conf=confidence, iou=iou_thresh,
                    imgsz=webcam_infer_size, verbose=False
                )

            # Annotate
            if results and results[0].boxes and len(results[0].boxes) > 0:
                annotated = results[0].plot()

                # Face recognition on interval
                if enable_face_recognition and _known_encodings_webcam:
                    processed_count = self.frame_count // max(frame_skip, 1)
                    if processed_count % self.face_interval == 0:
                        # Check if any person detected
                        class_ids = results[0].boxes.cls.tolist()
                        has_person = any(class_names[int(c)] == "person" for c in class_ids)
                        if has_person:
                            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.last_face_results = self._recognize_faces(frame_rgb)

                    # Draw cached face results
                    if self.last_face_results:
                        annotated = self._draw_face_labels(annotated, self.last_face_results)
            else:
                annotated = img

            # FPS
            curr_time = time.time()
            self.fps = 1.0 / max(curr_time - self.prev_time, 1e-6)
            self.prev_time = curr_time

            cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Count overlay
            if results and results[0].boxes and len(results[0].boxes) > 0:
                class_ids = results[0].boxes.cls.tolist()
                counts = defaultdict(int)
                for cid in class_ids:
                    counts[class_names[int(cid)]] += 1
                # Add recognized face names
                for name, _, _ in self.last_face_results:
                    if name != "Unknown":
                        counts[name] = counts.get(name, 0) + 1
                y_pos = 60
                for obj, count in sorted(counts.items()):
                    cv2.putText(annotated, f"{obj}: {count}", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    y_pos += 25

            # Log unique detections (thread-safe)
            if results and results[0].boxes and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                class_ids_list = boxes.cls.tolist()
                conf_list = boxes.conf.tolist()
                coord_list = boxes.xyxy.tolist()
                track_id_list = boxes.id.tolist() if boxes.id is not None else [None] * len(class_ids_list)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for cls_id, conf_val, coord, tid in zip(class_ids_list, conf_list, coord_list, track_id_list):
                    obj_name = class_names[int(cls_id)]

                    # Match face identity
                    identity = "-"
                    if obj_name == "person" and self.last_face_results:
                        px1, py1, px2, py2 = coord
                        for fname, (ftop, fright, fbottom, fleft), fconf in self.last_face_results:
                            if fname != "Unknown" and fleft >= px1 and ftop >= py1 and fright <= px2 and fbottom <= py2:
                                identity = fname
                                break

                    combo = (obj_name, identity)
                    tid_int = int(tid) if tid is not None else None

                    # Deduplicate
                    if tid_int is not None and tid_int in self._logged_ids:
                        continue
                    if tid_int is None and combo in self._logged_combos:
                        continue

                    entry = {
                        "object": obj_name,
                        "identity": identity,
                        "confidence": round(conf_val, 3),
                        "track_id": tid_int,
                        "bbox": str([round(c, 1) for c in coord]),
                        "timestamp": timestamp,
                    }
                    with YOLOProcessor.detection_lock:
                        YOLOProcessor.detection_log.append(entry)
                    if tid_int is not None:
                        self._logged_ids.add(tid_int)
                    self._logged_combos.add(combo)

            with self.lock:
                self.last_annotated = annotated.copy()

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"width": {"ideal": width}, "height": {"ideal": height}},
            "audio": False,
        },
        async_processing=True,
    )

    # Live sync webcam detections to session state
    live_log_placeholder = st.empty()
    if ctx.state.playing:
        import time as _time
        while ctx.state.playing:
            with YOLOProcessor.detection_lock:
                for entry in YOLOProcessor.detection_log:
                    if entry not in st.session_state["detections"]:
                        st.session_state["detections"].append(entry)
                YOLOProcessor.detection_log.clear()
            if st.session_state["detections"]:
                _df = pd.DataFrame(st.session_state["detections"])
                with live_log_placeholder.container():
                    st.markdown("### Live Detection Log")
                    st.markdown(f"**Total detections:** {len(_df)} | **Unique objects:** {_df['object'].nunique()}")
                    if "identity" in _df.columns:
                        identified = _df[_df["identity"] != "-"]
                        if not identified.empty:
                            st.markdown(f"**Identified persons:** {', '.join(identified['identity'].unique())}")
                    st.dataframe(_df, use_container_width=True)
            _time.sleep(2)
    # Stream stopped — flush remaining logs
    if not ctx.state.playing and YOLOProcessor.detection_log:
        with YOLOProcessor.detection_lock:
            for entry in YOLOProcessor.detection_log:
                if entry not in st.session_state["detections"]:
                    st.session_state["detections"].append(entry)
            YOLOProcessor.detection_log.clear()

# ══════════════════════════════════════════════
# DETECTION LOG
# ══════════════════════════════════════════════
if st.session_state["detections"]:
    st.markdown("### Detection Log")
    df = pd.DataFrame(st.session_state["detections"])

    st.markdown(f"**Total detections:** {len(df)} | **Unique objects:** {df['object'].nunique()}")

    # Show identified persons
    if "identity" in df.columns:
        identified = df[df["identity"] != "-"]
        if not identified.empty:
            st.markdown(f"**Identified persons:** {', '.join(identified['identity'].unique())}")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Log as CSV", data=csv, file_name="detection_log.csv", mime="text/csv")
