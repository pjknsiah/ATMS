# ATMS — Automated Traffic Management System
## Claude Code Project Guide

This file is the single source of truth for this project.
Read it fully before writing any code or suggesting any changes.

---

## What this project is

A real-time, vision-based traffic signal controller.
Four lane cameras feed video into a Python pipeline that:
1. Detects and counts vehicles per lane using Ultralytics YOLO11
2. Tracks individual vehicles across frames using ByteTrack
3. Decides which lane gets the green signal based on live congestion
4. Prevents deadlock so no lane is starved indefinitely

This is a capstone project. Code must be clean, well-documented, and testable.

---

## Project structure

```
ATMS/
├── CLAUDE.md                  ← you are here
├── README.md
├── requirements.txt
├── .env.example
├── main.py                    ← entry point
│
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py        ← YOLO11 wrapper
│   │   └── tracker.py         ← ByteTrack wrapper
│   │
│   ├── signal/
│   │   ├── __init__.py
│   │   ├── controller.py      ← signal decision engine
│   │   └── deadlock.py        ← deadlock prevention logic
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── lane.py            ← per-lane processing thread
│   │   └── manager.py         ← orchestrates all lanes
│   │
│   └── utils/
│       ├── __init__.py
│       ├── video.py           ← video loading / subclip helpers
│       └── logger.py          ← structured logging
│
├── tests/
│   ├── test_detector.py
│   ├── test_controller.py
│   ├── test_deadlock.py
│   └── test_pipeline.py
│
├── data/
│   ├── samples/               ← put test .mp4 files here (one per lane)
│   └── weights/               ← YOLO11 model weights go here
│
└── docs/
    ├── architecture.md
    └── future_scope.md
```

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | |
| Detection | Ultralytics YOLO11 | Best accuracy/speed ratio, native Python API, ONNX export |
| Tracking | ByteTrack (via Ultralytics) | Per-vehicle ID tracking across frames |
| Video | OpenCV (cv2) + MoviePy | Frame extraction + subclip windowing |
| Concurrency | `threading` (I/O) + `multiprocessing` (inference) | Parallel lane processing |
| Testing | pytest | |
| Logging | Python `logging` + structlog | |

Do NOT use cvlib. Do NOT use YOLOv3. Do NOT use threading alone for inference — use multiprocessing for CPU-bound YOLO inference.

---

## Core logic — read this carefully

### Lane processing loop

Every `WINDOW_SECONDS` seconds (configurable, default 5):
1. Each lane thread extracts frames from the current video subclip
2. Frames are passed to the detector → vehicle bounding boxes returned
3. Vehicle count is accumulated (not reset until that lane gets a green)
4. Counts are posted to a shared `LaneState` object

### Signal decision (controller.py)

```
every WINDOW_SECONDS:
    winner = lane with max cumulative_count
    if winner.consecutive_greens >= DEADLOCK_THRESHOLD:
        winner = next lane that has not had a green this round
    grant green to winner
    reset winner.cumulative_count = 0
    increment winner.consecutive_greens
    reset others.consecutive_greens = 0
```

`DEADLOCK_THRESHOLD` defaults to 2. Must be configurable via `.env`.

### Vehicle classes to count

Only count these COCO classes: `car`, `motorcycle`, `bus`, `truck`.
Ignore pedestrians, cyclists, animals, and all other classes.

### Tracked vs counted

Use ByteTrack IDs to avoid double-counting vehicles that span multiple frames
within the same window. A vehicle already counted in this window should not
be counted again, even if still visible.

---

## Configuration (.env)

```
WINDOW_SECONDS=5
DEADLOCK_THRESHOLD=2
CONFIDENCE_THRESHOLD=0.45
MODEL_WEIGHTS=data/weights/yolo11n.pt
LANE_COUNT=4
VIDEO_DIR=data/samples
LOG_LEVEL=INFO
```

All magic numbers must come from config, never hardcoded.

---

## Coding rules

1. **Every public function has a docstring** — what it does, args, return value.
2. **Type hints on every function signature** — no bare `def foo(x)`.
3. **No global mutable state** — pass state explicitly or use a shared dataclass.
4. **Each module does one thing** — detector.py only detects. controller.py only decides signals. Do not mix concerns.
5. **Errors are logged, not silently swallowed** — wrap video I/O and inference in try/except with structured log output.
6. **Tests for every non-trivial function** — especially signal logic and deadlock prevention. Use pytest fixtures and mock video data.
7. **No print() statements in production code** — use the logger.

---

## Build order (do this in sequence)

Build and test each module before moving to the next.
Do not jump ahead.

### Phase 1 — Foundation
- [ ] `requirements.txt`
- [ ] `.env.example` + config loader (`src/utils/config.py`)
- [ ] `src/utils/logger.py`
- [ ] `src/utils/video.py` — load video, extract subclip, yield frames
- [ ] Test: `tests/test_video.py` — can it open a sample file and yield frames?

### Phase 2 — Detection
- [ ] `src/detection/detector.py` — YOLO11 wrapper, returns `List[Detection]`
- [ ] `src/detection/tracker.py` — ByteTrack wrapper, returns tracked IDs
- [ ] Test: `tests/test_detector.py` — mock a frame, assert Detection objects returned

### Phase 3 — Signal logic
- [ ] `src/signal/controller.py` — takes lane counts, returns winning lane index
- [ ] `src/signal/deadlock.py` — occurrence counter, returns override lane if threshold hit
- [ ] Test: `tests/test_controller.py` and `tests/test_deadlock.py`
  - Test: lane with max count wins
  - Test: deadlock triggers at threshold=2
  - Test: all lanes eventually receive green

### Phase 4 — Pipeline
- [ ] `src/pipeline/lane.py` — per-lane thread: read subclip → detect → track → post count
- [ ] `src/pipeline/manager.py` — spawn lane threads, collect counts, call controller
- [ ] Test: `tests/test_pipeline.py` — mock lanes, assert counts collected and signal issued

### Phase 5 — Entry point
- [ ] `main.py` — parse args, load config, start manager

---

## Data structures

```python
# src/detection/detector.py
@dataclass
class Detection:
    class_name: str        # "car", "truck", etc.
    confidence: float
    bbox: tuple[int,int,int,int]  # x1, y1, x2, y2
    track_id: int | None   # set by tracker, None before tracking

# src/pipeline/lane.py
@dataclass
class LaneState:
    lane_id: int
    cumulative_count: int
    consecutive_greens: int
    current_signal: str    # "green" | "red"
    last_updated: float    # time.time()
```

---

## How to run (once built)

```bash
# install deps
pip install -r requirements.txt

# download YOLO11 weights (auto-downloads on first run via Ultralytics)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# put 4 lane videos in data/samples/
# naming: lane_0.mp4, lane_1.mp4, lane_2.mp4, lane_3.mp4

# run
python main.py
```

---

## What NOT to build yet

The following are in the future scope. Do not implement them now:
- Emergency vehicle detection
- ANPR / number plate recognition
- Web dashboard
- Cloud integration
- IoT hardware control

If asked to implement these, respond: "This is future scope — not in the current build."

---

## When you are unsure

- Prefer explicit over clever
- Prefer readable over compact
- If a design decision has tradeoffs, implement the simpler option and leave a `# TODO:` comment explaining the alternative
- Always run tests before declaring a phase complete
