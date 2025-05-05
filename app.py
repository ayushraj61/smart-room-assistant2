from ultralytics import YOLO
import cv2
import pyttsx3
import threading

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure the correct model path (change if needed)

# AI voice engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def trigger_action(label):
    actions = {
        "laptop": "Launching work mode.",
        "bed": "Time to rest. Activating sleep assistant.",
        "chair": "Ergonomic check: Sit upright!",
        "tv": "Entertainment mode ready.",
        "bottle": "Stay hydrated! Drink some water.",
    }

    response = actions.get(label, f"Unrecognized object: {label}.")
    print(response)
    engine.say(response)
    engine.runAndWait()

def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects using YOLOv8
        results = model(frame)
        annotated_frame = results[0].plot()

        for result in results.pred[0]:
            label_idx = int(result[5])  # Get the predicted class index
            label = model.names[label_idx]  # Get the corresponding label
            trigger_action(label)

        # Show the annotated frame
        cv2.imshow("Smart Room Assistant", annotated_frame)
        
        # Check if the user pressed 'q' to quit the webcam feed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_detection_thread():
    """Start the detection in a separate thread to avoid UI freezing"""
    thread = threading.Thread(target=detect_from_webcam)
    thread.start()

