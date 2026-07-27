# ATMS — Automated Traffic Management System

Real-time, vision-based traffic signal controller using Ultralytics YOLO11 and ByteTrack.

## Quick start

```bash
# 1. Clone / open in Claude Code
# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and edit config
cp .env.example .env

# 4. Download YOLO11 weights (auto on first run)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
# Move downloaded yolo11n.pt to data/weights/

# 5. Add lane videos
# Place 4 video files in data/samples/:
#   lane_0.mp4, lane_1.mp4, lane_2.mp4, lane_3.mp4

# 6. Run
python main.py 
```

## Run tests

```bash
pytest tests/ -v
```

## Build order

See `CLAUDE.md` — follow the 5-phase build plan in sequence.

## Stack

| Component | Tool |
|---|---|
| Detection | Ultralytics YOLO11 |
| Tracking | ByteTrack (via Ultralytics) |
| Video | OpenCV + MoviePy |
| Concurrency | threading + multiprocessing |
| Tests | pytest |

## Project structure

See `CLAUDE.md` for the full annotated directory tree.
