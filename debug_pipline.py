# debug_pipeline.py
from ultralytics import YOLO
import cv2

model = YOLO("data/weights/yolo11n.pt")
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

cap = cv2.VideoCapture("data/samples/lane_3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.1f}s")

# Simulate one 5-second window — frames 0 to fps*5
window_frames = int(fps * 5)
frame_idx = 0
frames_processed = 0
total_vehicles = 0
seen_ids = set()

while frame_idx < window_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, conf=0.30, persist=True, tracker="bytetrack.yaml", verbose=False)
    frames_processed += 1

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            if cls not in VEHICLE_CLASSES:
                continue
            track_id = int(box.id) if box.id is not None else None
            if track_id and track_id not in seen_ids:
                seen_ids.add(track_id)
                total_vehicles += 1
                print(f"  frame {frame_idx:04d}  new vehicle id={track_id}  cls={cls}")

    frame_idx += 3  # every_n_frames

cap.release()
print(f"\nFrames processed: {frames_processed}")
print(f"Unique vehicles counted: {total_vehicles}")