import cv2
import time
import csv
import os
from ultralytics import YOLO

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
print("Initializing Pre-Lab 6 System...")
model = YOLO("yolov8n.pt") # Pre-trained model
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("ERROR: Cannot access webcam.")

# Read one frame to get video dimensions for zones
success, frame = cap.read()
height, width, _ = frame.shape

# System Settings (Pre-Lab 6 Requirements)
CONFIDENCE_THRESHOLD = 0.50
# Allowlist: We map the string names you tested earlier to YOLO's class IDs
ALLOWED_CLASSES = ['person', 'cell phone', 'bottle', 'laptop', 'keyboard']
allowed_class_ids = [k for k, v in model.names.items() if v in ALLOWED_CLASSES]

# Define two non-overlapping virtual zones (Zone A: Left, Zone B: Right)
ZONE_A = {'name': 'Zone A', 'x1': 10, 'y1': 150, 'x2': width//2 - 10, 'y2': height - 50, 'color': (255, 0, 0)}
ZONE_B = {'name': 'Zone B', 'x1': width//2 + 10, 'y1': 150, 'x2': width - 10, 'y2': height - 50, 'color': (0, 0, 255)}

# State Memory & Counters
session_start_time = time.time()
frame_count = 0
track_history = {} # Format: {track_id: {'zone': None, 'entry_time': 0, 'dwell_alerted': False}}
unique_ids = set()
event_counts = {'ENTER': 0, 'EXIT': 0, 'DWELL': 0}

# ==========================================
# 2. CSV LOGGING SETUP
# ==========================================
CSV_FILE = "events.csv"
csv_headers = ["timestamp", "frame_number", "track_id", "class_name", "confidence", "zone", "event_type", "elapsed_seconds"]

# Create CSV and write headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

def log_event(frame_num, t_id, cls_name, conf, zone_name, event_type, elapsed):
    """Helper function to append events to the CSV file without spamming."""
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, frame_num, t_id, cls_name, f"{conf:.2f}", zone_name, event_type, f"{elapsed:.2f}"])
        print(f">> EVENT LOGGED: {event_type} | ID:{t_id} | {zone_name} | {elapsed:.2f}s")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def point_in_zone(px, py, zone):
    """Checks if a centroid (px, py) falls inside a rectangular zone."""
    return zone['x1'] <= px <= zone['x2'] and zone['y1'] <= py <= zone['y2']

# ==========================================
# 4. MAIN REAL-TIME LOOP
# ==========================================
previous_time = time.perf_counter()
print("System Active. Press 'q' to quit, 's' to screenshot, 'r' to reset.")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    annotated_frame = frame.copy()
    current_time_abs = time.time()
    
    # RUN YOLO TRACKING (Filtering by confidence and allowlist directly in the engine)
    results = model.track(source=frame, persist=True, conf=CONFIDENCE_THRESHOLD, classes=allowed_class_ids, verbose=False)
    
    current_zone_occupancy = {'Zone A': 0, 'Zone B': 0}

    # ==========================================
    # 5. PROCESS TRACKED OBJECTS & STATE LOGIC
    # ==========================================
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, t_id, c_id, conf in zip(boxes, track_ids, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[c_id]
            unique_ids.add(t_id)
            
            # Calculate centroid
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Determine which zone the centroid is currently in
            current_zone = None
            if point_in_zone(cx, cy, ZONE_A):
                current_zone = 'Zone A'
                current_zone_occupancy['Zone A'] += 1
            elif point_in_zone(cx, cy, ZONE_B):
                current_zone = 'Zone B'
                current_zone_occupancy['Zone B'] += 1

            # Initialize memory for new track IDs
            if t_id not in track_history:
                track_history[t_id] = {'zone': None, 'entry_time': 0, 'dwell_alerted': False}

            prev_zone = track_history[t_id]['zone']

            # EVENT LOGIC: Check for Zone Transitions
            if current_zone != prev_zone:
                # 1. Exit Event (if it left a zone)
                if prev_zone is not None:
                    elapsed = current_time_abs - track_history[t_id]['entry_time']
                    log_event(frame_count, t_id, class_name, conf, prev_zone, "EXIT_ZONE", elapsed)
                    event_counts['EXIT'] += 1
                
                # 2. Enter Event (if it entered a new zone)
                if current_zone is not None:
                    log_event(frame_count, t_id, class_name, conf, current_zone, "ENTER_ZONE", 0.0)
                    event_counts['ENTER'] += 1
                    track_history[t_id]['entry_time'] = current_time_abs
                    track_history[t_id]['dwell_alerted'] = False
                
                # Update memory
                track_history[t_id]['zone'] = current_zone

            # EVENT LOGIC: Check for Dwell Alert (Continuous presence > 3.0s)
            elif current_zone is not None:
                dwell_time = current_time_abs - track_history[t_id]['entry_time']
                if dwell_time >= 3.0 and not track_history[t_id]['dwell_alerted']:
                    log_event(frame_count, t_id, class_name, conf, current_zone, "DWELL_ALERT", dwell_time)
                    event_counts['DWELL'] += 1
                    track_history[t_id]['dwell_alerted'] = True
                    
                    # Visual Indicator on screen for Dwell
                    cv2.putText(annotated_frame, "DWELL ALERT!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw bounding box and centroid
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)
            label = f"ID:{t_id} {class_name} {conf*100:.0f}%"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for zone in [ZONE_A, ZONE_B]:
        cv2.rectangle(annotated_frame, (zone['x1'], zone['y1']), (zone['x2'], zone['y2']), zone['color'], 2)
        (text_width, text_height), baseline = cv2.getTextSize(zone['name'], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_x, label_y = zone['x1'] + 3, zone['y1'] + 3
        cv2.rectangle(annotated_frame, (label_x, label_y),
                      (label_x + text_width + 16, label_y + text_height + baseline + 10), (28, 28, 28), -1)
        cv2.rectangle(annotated_frame, (label_x, label_y),
                      (label_x + text_width + 16, label_y + text_height + baseline + 10), zone['color'], 1)
        cv2.putText(annotated_frame, zone['name'], (label_x + 8, label_y + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ==========================================
    # 6. ON-SCREEN ANALYTICS HUD
    # ==========================================
    current_time = time.perf_counter()
    fps = 1.0 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
    previous_time = current_time
    run_time = current_time_abs - session_start_time

    annotated_frame = cv2.copyMakeBorder(annotated_frame, 110, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    hud_text = [
        ("ZONE TRACKING", 12, 22),
        (f"FPS: {fps:.1f}  |  Runtime: {run_time:.0f}s", width // 2, 22),
        (f"In zones: {sum(current_zone_occupancy.values())}", 12, 48),
        (f"Unique IDs: {len(unique_ids)}", width // 2, 48),
        (f"Zone A: {current_zone_occupancy['Zone A']}", 12, 70),
        (f"Zone B: {current_zone_occupancy['Zone B']}", width // 2, 70),
        (f"Events  In: {event_counts['ENTER']}  Out: {event_counts['EXIT']}  Dwell: {event_counts['DWELL']}", 12, 95)
    ]
    for text, text_x, text_y in hud_text:
        cv2.putText(annotated_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (235, 235, 235), 1, cv2.LINE_AA)

    cv2.imshow("Pre-Lab 6: Zone Tracking", annotated_frame)

    # ==========================================
    # 7. KEYBOARD CONTROLS (Q, S, R)
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting system...")
        break
    elif key == ord("s"):
        filename = f"screenshot_{int(time.time()*1000)}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f">> SCREENSHOT SAVED: {filename}")
    elif key == ord("r"):
        # Reset Session Memory
        track_history.clear()
        unique_ids.clear()
        event_counts = {'ENTER': 0, 'EXIT': 0, 'DWELL': 0}
        session_start_time = time.time()
        # Log Reset Event to CSV
        with open(CSV_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), frame_count, "-", "SYSTEM", "-", "-", "RESET", "0.00"])
        print(">> SESSION RESET. Counters cleared.")

cap.release()
cv2.destroyAllWindows()
print("System closed safely.")