"""Generate project documentation PDF for Smart Room Assistant."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def build_pdf():
    doc = SimpleDocTemplate(
        "Smart_Room_Assistant_Documentation.pdf",
        pagesize=A4,
        rightMargin=25*mm, leftMargin=25*mm,
        topMargin=25*mm, bottomMargin=25*mm
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                  fontSize=28, textColor=HexColor('#1a1a2e'),
                                  spaceAfter=10, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                     fontSize=14, textColor=HexColor('#666666'),
                                     alignment=TA_CENTER, spaceAfter=30)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
                         fontSize=20, textColor=HexColor('#16213e'),
                         spaceBefore=20, spaceAfter=12,
                         borderWidth=1, borderColor=HexColor('#4CAF50'),
                         borderPadding=5)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                         fontSize=16, textColor=HexColor('#0f3460'),
                         spaceBefore=15, spaceAfter=8)
    h3 = ParagraphStyle('H3', parent=styles['Heading3'],
                         fontSize=13, textColor=HexColor('#533483'),
                         spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle('Body', parent=styles['Normal'],
                           fontSize=11, leading=16, spaceAfter=8)
    bullet = ParagraphStyle('Bullet', parent=body,
                             leftIndent=20, bulletIndent=10,
                             spaceBefore=2, spaceAfter=2)
    code_style = ParagraphStyle('Code', parent=styles['Normal'],
                                 fontName='Courier', fontSize=9,
                                 backColor=HexColor('#f5f5f5'),
                                 leftIndent=15, spaceAfter=8, leading=13)
    highlight = ParagraphStyle('Highlight', parent=body,
                                backColor=HexColor('#e8f5e9'),
                                borderWidth=1, borderColor=HexColor('#4CAF50'),
                                borderPadding=8, spaceAfter=12)

    elements = []

    # ─── COVER PAGE ───
    elements.append(Spacer(1, 80))
    elements.append(Paragraph("Smart Room Assistant", title_style))
    elements.append(Paragraph("YOLOv8 + Face Recognition", subtitle_style))
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="60%", color=HexColor('#4CAF50'), thickness=2))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Complete Project Documentation", ParagraphStyle(
        'CoverSub', parent=body, fontSize=16, alignment=TA_CENTER, textColor=HexColor('#333'))))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Technical Flow | Architecture | How It Works", ParagraphStyle(
        'CoverSub2', parent=body, fontSize=12, alignment=TA_CENTER, textColor=HexColor('#777'))))
    elements.append(Spacer(1, 40))

    tech_data = [
        ['Technology', 'Purpose'],
        ['YOLOv8 (Ultralytics)', 'Object Detection & Tracking'],
        ['face_recognition (dlib)', 'Face Registration & Recognition'],
        ['Streamlit', 'Web UI Framework'],
        ['streamlit-webrtc', 'Real-time Webcam Streaming'],
        ['OpenCV', 'Image Processing & Preprocessing'],
        ['BoT-SORT', 'Multi-Object Tracking'],
        ['Python 3.10', 'Runtime'],
    ]
    tech_table = Table(tech_data, colWidths=[200, 250])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(tech_table)
    elements.append(PageBreak())

    # ─── TABLE OF CONTENTS ───
    elements.append(Paragraph("Table of Contents", h1))
    toc_items = [
        "1. Project Overview",
        "2. System Architecture",
        "3. How Object Detection Works (YOLOv8)",
        "4. How Face Recognition Works",
        "5. Input Modes - Image, Video, Webcam",
        "6. Preprocessing Pipeline",
        "7. Object Tracking (BoT-SORT)",
        "8. Detection Logging & Deduplication",
        "9. Webcam Optimization Techniques",
        "10. Face Registration Flow",
        "11. File Structure",
        "12. Deployment on Streamlit Cloud",
        "13. Complete Data Flow Diagram",
    ]
    for item in toc_items:
        elements.append(Paragraph(item, ParagraphStyle('TOC', parent=body, fontSize=12, spaceBefore=6, leftIndent=20)))
    elements.append(PageBreak())

    # ─── 1. PROJECT OVERVIEW ───
    elements.append(Paragraph("1. Project Overview", h1))
    elements.append(Paragraph(
        "Smart Room Assistant is a real-time object detection and face recognition web application. "
        "It uses YOLOv8 (a state-of-the-art deep learning model) to detect objects in images, videos, "
        "and live webcam feeds. It can also recognize registered faces and identify people by name.", body))
    elements.append(Paragraph("What can it do?", h3))
    for item in [
        "Detect 80+ object types (person, car, chair, laptop, phone, etc.)",
        "Track objects across video frames with unique IDs",
        "Register faces via webcam or photo upload",
        "Recognize registered faces in real-time and show names",
        "Log all detections with timestamps, confidence scores, and identity",
        "Export detection logs as CSV",
        "Generate detection heatmaps for videos",
        "Show live object counts on dashboard",
    ]:
        elements.append(Paragraph(f"&bull; {item}", bullet))
    elements.append(PageBreak())

    # ─── 2. SYSTEM ARCHITECTURE ───
    elements.append(Paragraph("2. System Architecture", h1))
    elements.append(Paragraph(
        "The app follows a single-page architecture built with Streamlit. All processing happens "
        "server-side (Python), while the UI is rendered in the browser.", body))

    arch_data = [
        ['Layer', 'Component', 'What It Does'],
        ['Frontend', 'Streamlit UI', 'Renders sidebar, buttons, image displays, tables'],
        ['Frontend', 'streamlit-webrtc', 'Streams webcam video via WebRTC protocol'],
        ['Processing', 'OpenCV', 'Image preprocessing (CLAHE, denoising, resize)'],
        ['AI - Detection', 'YOLOv8 (Ultralytics)', 'Detects objects with bounding boxes'],
        ['AI - Tracking', 'BoT-SORT', 'Assigns persistent IDs to tracked objects'],
        ['AI - Face', 'face_recognition (dlib)', 'Encodes faces into 128-dim vectors, matches them'],
        ['Storage', 'faces/face_db.json', 'Stores face embeddings as JSON'],
        ['Storage', 'Session State', 'Stores detection logs, counts, heatmap in memory'],
    ]
    arch_table = Table(arch_data, colWidths=[80, 130, 240])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(arch_table)
    elements.append(PageBreak())

    # ─── 3. HOW OBJECT DETECTION WORKS ───
    elements.append(Paragraph("3. How Object Detection Works (YOLOv8)", h1))
    elements.append(Paragraph(
        "YOLO stands for 'You Only Look Once'. It is a deep learning model that can detect "
        "multiple objects in a single image in one pass (one forward pass through the neural network).", body))

    elements.append(Paragraph("Step-by-step process:", h3))
    steps = [
        ("Input", "An image/frame is passed to the model (e.g., 640x640 pixels)."),
        ("Backbone (Feature Extraction)", "The image passes through a CNN (Convolutional Neural Network) "
         "that extracts features like edges, textures, shapes at multiple scales."),
        ("Neck (Feature Fusion)", "Features from different scales are combined using PANet/FPN "
         "so that both small and large objects can be detected."),
        ("Head (Prediction)", "The model predicts bounding boxes (x, y, width, height), "
         "confidence scores, and class probabilities for each detected object."),
        ("NMS (Non-Maximum Suppression)", "Overlapping boxes for the same object are removed. "
         "Only the box with the highest confidence is kept. Controlled by IoU threshold."),
        ("Output", "Final list of detections: class name, confidence %, bounding box coordinates."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        elements.append(Paragraph(f"<b>Step {i}: {title}</b>", body))
        elements.append(Paragraph(desc, ParagraphStyle('StepDesc', parent=body, leftIndent=20)))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Model Sizes Available:", h3))
    model_data = [
        ['Model', 'Params', 'Speed', 'Accuracy', 'Best For'],
        ['YOLOv8n', '3.2M', 'Fastest', 'Low', 'Real-time on weak hardware'],
        ['YOLOv8s', '11.2M', 'Fast', 'Medium', 'Streamlit Cloud (default)'],
        ['YOLOv8m', '25.9M', 'Medium', 'Good', 'Local with GPU'],
        ['YOLOv8l', '43.7M', 'Slow', 'High', 'High accuracy needed'],
        ['YOLOv8x', '68.2M', 'Slowest', 'Highest', 'Maximum accuracy'],
    ]
    model_table = Table(model_data, colWidths=[70, 60, 65, 65, 190])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#0f3460')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(model_table)
    elements.append(PageBreak())

    # ─── 4. HOW FACE RECOGNITION WORKS ───
    elements.append(Paragraph("4. How Face Recognition Works", h1))
    elements.append(Paragraph(
        "Face recognition uses the 'face_recognition' library, which is built on top of dlib's "
        "deep learning face recognition model. It works in 3 stages:", body))

    elements.append(Paragraph("<b>Stage 1: Face Detection (HOG)</b>", h3))
    elements.append(Paragraph(
        "HOG (Histogram of Oriented Gradients) scans the image to find rectangular regions "
        "that contain faces. It looks for patterns of light and dark that match a face shape. "
        "Returns bounding box coordinates (top, right, bottom, left) for each face.", body))

    elements.append(Paragraph("<b>Stage 2: Face Encoding (128-dimensional vector)</b>", h3))
    elements.append(Paragraph(
        "Each detected face is passed through a deep neural network that outputs a 128-dimensional "
        "vector (an array of 128 numbers). This vector is a mathematical 'fingerprint' of the face. "
        "Two photos of the same person will produce similar vectors. Different people produce different vectors.", body))

    elements.append(Paragraph("<b>Stage 3: Face Matching (Cosine Distance)</b>", h3))
    elements.append(Paragraph(
        "The encoding of a detected face is compared against all registered face encodings using "
        "Euclidean distance. If the distance is less than 0.6 (tolerance threshold), it's a match. "
        "The name of the closest match is displayed on screen.", body))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("The complete flow:", highlight))
    elements.append(Paragraph(
        "YOLO detects 'person' → Crop person region → HOG finds face in crop → "
        "Encode face to 128-dim vector → Compare with registered faces → "
        "Distance &lt; 0.6? Show name. Distance &gt; 0.6? Show 'Unknown'.", code_style))
    elements.append(PageBreak())

    # ─── 5. INPUT MODES ───
    elements.append(Paragraph("5. Input Modes - Image, Video, Webcam", h1))

    elements.append(Paragraph("<b>Image Mode</b>", h2))
    elements.append(Paragraph("Flow: Upload → Preprocess (CLAHE) → YOLOv8 Predict → Face Recognition → "
                               "Annotate → Display side-by-side (Original vs Detected) → Show object crops → Log detections", body))

    elements.append(Paragraph("<b>Video Mode</b>", h2))
    elements.append(Paragraph("Flow: Upload → Save to temp file → Read frame-by-frame → Skip frames (every Nth) → "
                               "Preprocess → YOLOv8 Track (BoT-SORT) → Face Recognition (every 5th processed frame) → "
                               "Annotate with FPS + counts → Update live display → Log unique detections → "
                               "After completion: show heatmap + full log", body))
    for item in [
        "Frame skipping: Only processes every Nth frame (configurable, default=2) for speed",
        "Tracking: Uses model.track() instead of model.predict() to assign persistent IDs",
        "Face recognition runs every 5th processed frame to save CPU (reuses last result between)",
        "Progress bar shows how far through the video",
        "Start/Stop buttons to control processing",
        "Live log updates every 30 processed frames",
    ]:
        elements.append(Paragraph(f"&bull; {item}", bullet))

    elements.append(Paragraph("<b>Webcam Mode</b>", h2))
    elements.append(Paragraph("Flow: Browser captures webcam via WebRTC → Frames sent to server → "
                               "YOLOProcessor.recv() processes each frame → Frame skip (reuse last result) → "
                               "CLAHE preprocessing → YOLO detect/track at lower resolution (320px) → "
                               "Face recognition every 10th processed frame → Annotate → Return frame to browser", body))
    for item in [
        "Uses streamlit-webrtc for browser-based camera access (no local install needed)",
        "async_processing=True for non-blocking frame handling",
        "Inference runs at 320px (configurable) instead of 640px for 4x speed boost",
        "Thread-safe detection logging via class-level shared list + threading.Lock",
        "FPS counter overlay shows real-time performance",
    ]:
        elements.append(Paragraph(f"&bull; {item}", bullet))
    elements.append(PageBreak())

    # ─── 6. PREPROCESSING ───
    elements.append(Paragraph("6. Preprocessing Pipeline", h1))
    elements.append(Paragraph(
        "Before passing a frame to YOLOv8, optional preprocessing improves detection accuracy, "
        "especially in poor lighting conditions.", body))

    elements.append(Paragraph("<b>CLAHE (Contrast Limited Adaptive Histogram Equalization)</b>", h3))
    elements.append(Paragraph(
        "Problem: In dark or unevenly lit images, objects are hard to see. "
        "Solution: CLAHE enhances local contrast without over-amplifying noise.", body))
    elements.append(Paragraph(
        "How it works: Convert BGR to LAB color space → Apply CLAHE to L (lightness) channel only → "
        "Merge back → Convert to BGR. This brightens dark areas while keeping colors natural.", body))

    elements.append(Paragraph("<b>Denoising</b>", h3))
    elements.append(Paragraph(
        "Uses cv2.fastNlMeansDenoisingColored() to reduce noise in low-light images. "
        "Slower but helps when camera produces grainy images. Disabled by default.", body))

    elements.append(Paragraph("<b>Aspect Ratio Preservation</b>", h3))
    elements.append(Paragraph(
        "Videos are resized to 960px width while maintaining aspect ratio (no stretching/distortion). "
        "This prevents objects from looking squished or elongated.", body))
    elements.append(PageBreak())

    # ─── 7. OBJECT TRACKING ───
    elements.append(Paragraph("7. Object Tracking (BoT-SORT)", h1))
    elements.append(Paragraph(
        "Without tracking, each frame is processed independently — the same person detected in "
        "frame 1 and frame 2 appears as two different 'person' detections. Tracking solves this.", body))

    elements.append(Paragraph("How BoT-SORT works:", h3))
    for item in [
        "<b>Detection:</b> YOLOv8 detects objects in the current frame",
        "<b>Feature Extraction:</b> A Re-ID (Re-Identification) model extracts appearance features for each detection",
        "<b>Motion Prediction:</b> Kalman filter predicts where each tracked object should be in the next frame",
        "<b>Association:</b> Hungarian algorithm matches current detections with existing tracks using both "
        "appearance similarity + spatial proximity",
        "<b>Track Management:</b> New tracks are created for unmatched detections. Tracks with no matches "
        "for several frames are deleted. Each track gets a unique ID (1, 2, 3...)",
    ]:
        elements.append(Paragraph(f"&bull; {item}", bullet))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "Result: Person #1 keeps ID=1 across all frames, even if they move or are briefly occluded. "
        "This enables accurate counting and deduplicated logging.", highlight))
    elements.append(PageBreak())

    # ─── 8. DETECTION LOGGING ───
    elements.append(Paragraph("8. Detection Logging & Deduplication", h1))
    elements.append(Paragraph(
        "The app logs each unique detection only once, not every frame. This is done through "
        "two deduplication strategies:", body))

    elements.append(Paragraph("<b>Strategy 1: Track ID deduplication (when tracking is ON)</b>", h3))
    elements.append(Paragraph(
        "Each tracked object has a unique ID (e.g., person ID=3). Once ID=3 is logged, "
        "it won't be logged again even if detected in 1000 more frames.", body))

    elements.append(Paragraph("<b>Strategy 2: Object+Identity combo (when tracking is OFF)</b>", h3))
    elements.append(Paragraph(
        "Without tracking, deduplication uses the combination of (object_type, identity). "
        "So 'person + Ayush' is logged once, 'person + Unknown' is logged once, 'chair + -' is logged once.", body))

    elements.append(Paragraph("Each log entry contains:", h3))
    log_data = [
        ['Field', 'Example', 'Description'],
        ['object', 'person', 'YOLO class name'],
        ['identity', 'Ayush', 'Recognized face name (or "-")'],
        ['confidence', '0.847', 'YOLO detection confidence (0-1)'],
        ['track_id', '3', 'BoT-SORT persistent ID'],
        ['bbox', '[120.5, 80.2, 450.1, 520.7]', 'Bounding box [x1, y1, x2, y2]'],
        ['timestamp', '2026-03-12 14:30:05', 'When detected'],
    ]
    log_table = Table(log_data, colWidths=[70, 140, 240])
    log_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#533483')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(log_table)
    elements.append(PageBreak())

    # ─── 9. WEBCAM OPTIMIZATION ───
    elements.append(Paragraph("9. Webcam Optimization Techniques", h1))
    elements.append(Paragraph(
        "The webcam needs to run in real-time (15-25 FPS). Several optimizations make this possible:", body))

    opt_data = [
        ['Technique', 'What It Does', 'Speed Gain'],
        ['Model Caching\n(@st.cache_resource)', 'Load YOLO model once,\nreuse across reruns', 'Eliminates 2-5s\nreload per interaction'],
        ['Frame Skipping', 'Process every 2nd frame,\nreuse last result', '2x FPS'],
        ['Lower Inference Size\n(320px vs 640px)', 'Run YOLO at smaller\nresolution for webcam', '~4x faster'],
        ['Face Recognition\nInterval (every 10th)', 'Skip face matching on\nmost frames, reuse result', 'Saves ~100ms/frame'],
        ['Async Processing', 'WebRTC frames processed\nin background thread', 'Non-blocking UI'],
        ['ONNX Export\n(optional)', 'Convert model to ONNX\noptimized runtime', '2-3x faster on CPU'],
    ]
    opt_table = Table(opt_data, colWidths=[120, 160, 170])
    opt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(opt_table)
    elements.append(PageBreak())

    # ─── 10. FACE REGISTRATION ───
    elements.append(Paragraph("10. Face Registration Flow", h1))

    elements.append(Paragraph("<b>Method 1: Upload Photos</b>", h2))
    elements.append(Paragraph(
        "User enters name → Uploads multiple photos (JPG/PNG) → Each photo is processed: "
        "HOG detects face → Validates exactly 1 face → Extracts 128-dim encoding → "
        "Saves encoding to faces/face_db.json → Saves cropped face image to faces/ folder.", body))

    elements.append(Paragraph("<b>Method 2: Webcam Capture (5 angles)</b>", h2))
    elements.append(Paragraph(
        "User enters name → App guides through 5 poses:", body))
    for i, pose in enumerate(["Look straight", "Turn slightly left", "Turn slightly right",
                               "Tilt up slightly", "Tilt down slightly"], 1):
        elements.append(Paragraph(f"&bull; Step {i}: {pose}", bullet))
    elements.append(Paragraph(
        "Each capture: Camera takes photo → HOG validates face → Stores in session → "
        "After all 5: User clicks 'Register' → All 5 encodings saved to face_db.json.", body))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Why multiple angles?</b>", h3))
    elements.append(Paragraph(
        "A single front-facing photo works only when the person faces the camera directly. "
        "By registering 5 angles, the system can recognize the person even when they turn "
        "their head, look up/down, or are viewed from the side. Each angle creates a slightly "
        "different 128-dim encoding, and the system matches against the closest one.", body))

    elements.append(Paragraph("<b>Storage Format (face_db.json):</b>", h3))
    elements.append(Paragraph(
        '{"Ayush": [[0.12, -0.05, 0.23, ...128 numbers...], [0.11, -0.04, ...]], '
        '"Rahul": [[0.08, 0.15, ...]]}', code_style))
    elements.append(PageBreak())

    # ─── 11. FILE STRUCTURE ───
    elements.append(Paragraph("11. File Structure", h1))
    file_data = [
        ['File', 'Purpose'],
        ['app2.py', 'Main application - all detection, tracking, face recognition, UI logic'],
        ['requirements.txt', 'Python dependencies (streamlit, ultralytics, face_recognition, etc.)'],
        ['packages.txt', 'System dependencies for Streamlit Cloud (cmake, libboost)'],
        ['runtime.txt', 'Python version specification (3.10)'],
        ['.streamlit/config.toml', 'Streamlit server config (upload size, CORS, theme)'],
        ['.devcontainer/devcontainer.json', 'GitHub Codespaces configuration'],
        ['.gitignore', 'Ignored files (snapshots/, *.pt, *.onnx, temp files)'],
        ['faces/face_db.json', 'Face embeddings database (created at runtime)'],
        ['faces/*.jpg', 'Registered face reference images'],
        ['snapshots/', 'Saved detection snapshots (created at runtime)'],
        ['README.md', 'Project documentation'],
    ]
    file_table = Table(file_data, colWidths=[160, 290])
    file_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(file_table)
    elements.append(PageBreak())

    # ─── 12. DEPLOYMENT ───
    elements.append(Paragraph("12. Deployment on Streamlit Cloud", h1))
    elements.append(Paragraph("Steps to deploy:", h3))
    deploy_steps = [
        "Push code to GitHub (github.com/ayushraj61/smart-room-assistant2)",
        "Go to share.streamlit.io and sign in with GitHub",
        "Click 'New app' → Select your repo → Branch: main → Main file: app2.py",
        "Streamlit Cloud auto-detects packages.txt (installs cmake, libboost) and "
        "requirements.txt (installs Python packages)",
        "App deploys and is accessible at https://ayushraj61-smart-room-assistant2.streamlit.app",
    ]
    for i, step in enumerate(deploy_steps, 1):
        elements.append(Paragraph(f"<b>{i}.</b> {step}", body))

    elements.append(Paragraph("Important notes for Streamlit Cloud:", h3))
    for item in [
        "CPU-only environment (~1GB RAM) — use model size 's' or 'n'",
        "Ephemeral filesystem — face registrations don't persist across restarts",
        "To keep faces: register locally, commit faces/ folder to repo before deploying",
        "packages.txt provides system-level dependencies (cmake for dlib compilation)",
    ]:
        elements.append(Paragraph(f"&bull; {item}", bullet))
    elements.append(PageBreak())

    # ─── 13. COMPLETE DATA FLOW ───
    elements.append(Paragraph("13. Complete Data Flow Diagram", h1))
    elements.append(Paragraph("The entire flow from input to output:", h3))
    elements.append(Spacer(1, 10))

    flow_steps = [
        ("USER INPUT", "Image upload / Video upload / Webcam stream"),
        ("   |", ""),
        ("PREPROCESSING", "CLAHE contrast enhancement + Optional denoising"),
        ("   |", ""),
        ("YOLO INFERENCE", "model.predict() or model.track() at configured size\n"
         "→ Returns: bounding boxes, class IDs, confidence scores, track IDs"),
        ("   |", ""),
        ("FACE RECOGNITION", "For each 'person' detection:\n"
         "HOG face detection → 128-dim encoding → Compare with face_db.json\n"
         "→ Match (distance < 0.6): Show name | No match: Show 'Unknown'"),
        ("   |", ""),
        ("ANNOTATION", "Draw bounding boxes + labels + FPS counter + object counts\n"
         "+ Face name labels (green=known, red=unknown)"),
        ("   |", ""),
        ("LOGGING", "Deduplicate by track_id or object+identity combo\n"
         "→ Store in session_state (main thread) or shared list (webcam thread)"),
        ("   |", ""),
        ("DISPLAY", "Show annotated frame + Live object count dashboard\n"
         "+ Detection log table + CSV download button"),
    ]

    for title, desc in flow_steps:
        if title.strip() == "|":
            elements.append(Paragraph("          ↓", ParagraphStyle('Arrow', parent=body,
                                                                      fontSize=14, alignment=TA_CENTER)))
        else:
            if desc:
                flow_data = [[title, desc]]
                ft = Table(flow_data, colWidths=[120, 330])
                ft.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, 0), HexColor('#4CAF50')),
                    ('TEXTCOLOR', (0, 0), (0, 0), HexColor('#ffffff')),
                    ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BACKGROUND', (1, 0), (1, 0), HexColor('#e8f5e9')),
                    ('BOX', (0, 0), (-1, -1), 1, HexColor('#4CAF50')),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ]))
                elements.append(ft)

    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", color=HexColor('#4CAF50'), thickness=2))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Built with YOLOv8 + face_recognition + Streamlit",
                               ParagraphStyle('Footer', parent=body, fontSize=10,
                                              alignment=TA_CENTER, textColor=HexColor('#999'))))
    elements.append(Paragraph("GitHub: github.com/ayushraj61/smart-room-assistant2",
                               ParagraphStyle('Footer2', parent=body, fontSize=10,
                                              alignment=TA_CENTER, textColor=HexColor('#999'))))

    doc.build(elements)
    print("PDF generated: Smart_Room_Assistant_Documentation.pdf")

if __name__ == "__main__":
    build_pdf()
