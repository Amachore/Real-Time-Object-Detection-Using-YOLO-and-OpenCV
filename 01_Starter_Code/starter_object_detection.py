import cv2
import time
from collections import Counter
from ultralytics import YOLO

# LAB 6: REAL-TIME OBJECT DETECTION
# STEP 1: LOAD PRETRAINED YOLO MODEL
# (Corrected from 'yololin.pt' to the standard ultralytics format)
model = YOLO("yolov8n.pt") 

# STEP 2: INITIALIZE WEBCAM
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("ERROR: Cannot access webcam.")

# STEP 3: CONFIGURATION
CONFIDENCE_THRESHOLD = 0.50
WINDOW_NAME = "LAB 6 YOLO Object Detection"
previous_time = time.perf_counter()

print("LAB 6: OBJECT DETECTION SYSTEM")
print("Press Q to exit. Press S to save a screenshot.")

# STEP 4: START REAL-TIME DETECTION
while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Failed to read webcam frame.")
        break

    # STEP 5: RUN YOLO INFERENCE
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    # STEP 6: PROCESS DETECTIONS
    annotated_frame = frame.copy()
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            detected_objects.append(class_name)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence * 100:.1f}%"
            cv2.putText(annotated_frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # STEP 7: COUNT DETECTED OBJECTS
    object_counts = Counter(detected_objects)
    total_objects = sum(object_counts.values())

    # STEP 8: CALCULATE FPS
    current_time = time.perf_counter()
    elapsed = current_time - previous_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    previous_time = current_time

    # STEP 9: DISPLAY SYSTEM INFORMATION
    # The (30, 30, 30) background acts as a clean, dark-mode HUD overlay
    cv2.rectangle(annotated_frame, (0, 0), (390, 85), (30, 30, 30), -1)
    cv2.putText(annotated_frame, "LAB 6: OBJECT DETECTION",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Objects Detected: {total_objects}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # STEP 10: DISPLAY OBJECT COUNTS
    y_position = 110
    for object_name, count in sorted(object_counts.items()):
        text = f"{object_name}: {count}"
        cv2.putText(annotated_frame, text, (10, y_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        y_position += 25

    # STEP 11: SHOW OUTPUT
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # STEP 12: KEYBOARD CONTROLS
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting object detection...")
        break
    elif key == ord("s"):
        filename = f"detection_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"Screenshot saved: {filename}")
# STEP 13: RELEASE RESOURCES
cap.release()
cv2.destroyAllWindows()
print("System terminated successfully.")
