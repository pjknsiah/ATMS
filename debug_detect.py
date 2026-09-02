# debug_detect.py
from ultralytics import YOLO
import cv2

model = YOLO("data/weights/yolo11n.pt")
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

cap = cv2.VideoCapture("data/samples/lane_0.mp4")
ret, frame = cap.read()
cap.release()

results = model(frame, conf=0.25, verbose=True)
for r in results:
    for box in r.boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        print(f"  {cls:<15} conf={conf:.2f}  {'✓ VEHICLE' if cls in VEHICLE_CLASSES else '✗ skip'}")