# ATMS — Automated Traffic Management System

Real-time, vision-based traffic signal controller. Four lane cameras feed a Python
pipeline that detects and tracks vehicles per lane using YOLO11 + ByteTrack, then
issues green signals based on live congestion — with built-in deadlock prevention so
no lane is starved indefinitely. A FastAPI + React web dashboard streams live signal
decisions over WebSocket.

> **Status:** Phases 1–5 complete · web dashboard live · 63 tests passing · verified on real video

---

## How it works

Every `WINDOW_SECONDS` (default 5 s), each lane's camera feed is sampled:

1. Extract frames from the lane video at the current offset (loops at end-of-file)
2. Run YOLO11 inference — detect cars, motorcycles, buses, and trucks only
3. Feed detections into ByteTrack — each vehicle gets a persistent ID
4. Count only **new** IDs not seen earlier in this window (no double-counting)
5. Add the window count to the lane's cumulative total

The signal controller picks the lane with the highest cumulative count, grants it
green, resets its count to zero, then moves to the next collection window.

### Deadlock prevention

If a single lane wins `DEADLOCK_THRESHOLD` consecutive rounds (default 2), the
`DeadlockGuard` forces a round-robin override: the controller picks the next lane
that hasn't yet received green this cycle, regardless of counts. Once every lane has
been served, the cycle resets and count-based selection resumes.

---

## Architecture

```
lane_0.mp4 ─┐
lane_1.mp4 ─┤  subprocess × 4
lane_2.mp4 ─┤  VehicleDetector (YOLO11)
lane_3.mp4 ─┘  └─ VehicleTracker (ByteTrack)
                    └─ (lane_id, count) → multiprocessing.Queue
                            │
                    PipelineManager (main process)
                    ├─ accumulates LaneState[]
                    │       │
                    │   SignalController + DeadlockGuard
                    │   └─ winner → green · others → red
                    │
                    └─ DashboardBroadcaster
                        └─ broadcast_sync() → WebSocket → React dashboard
```

Each lane runs in its own `multiprocessing.Process`. The YOLO model is loaded inside
each subprocess (models are not picklable across processes). The main process runs
the signal controller and streams decisions to the dashboard over WebSocket.

---

## Requirements

| Requirement   | Minimum               | Notes                                                        |
|---------------|-----------------------|--------------------------------------------------------------|
| Python        | 3.11+                 | Uses `int \| None` union syntax and modern type hints        |
| Node.js       | 18+                   | Required to build the React dashboard frontend               |
| CPU           | 4-core                | 8+ cores recommended; GPU auto-detected via CUDA if present  |
| RAM           | 4 GB                  | Each subprocess loads ~200 MB at runtime                     |
| Video files   | 4 × .mp4              | Named `lane_0.mp4` … `lane_3.mp4` in `data/samples/`        |
| YOLO weights  | yolo11n.pt            | Auto-downloaded on first run; place in `data/weights/`       |

---

## Setup

**1. Install Python dependencies**
```bash
pip install -r requirements.txt 
```

**2. Copy and edit config**
```bash
cp .env.example .env
# edit .env as needed — see Configuration table below
```

**3. Download YOLO11 weights**
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
# then move yolo11n.pt into data/weights/
```

**4. Add lane videos**
```
data/samples/
  lane_0.mp4
  lane_1.mp4
  lane_2.mp4
  lane_3.mp4
```

**5. Build the dashboard frontend**
```bash
cd dashboard-ui && npm install && npm run build
```

**6. Validate assets (optional)**
```bash
python3 main.py --dry-run
# checks all four videos and the weights file exist; exits cleanly
```

---

## Running

**Full pipeline + dashboard**
```bash
python3 main.py
```
Spawns four lane subprocesses and starts the FastAPI server on port 8000.
Open **http://localhost:8000** to view the live dashboard.
Stop with `Ctrl-C` — processes are joined cleanly.

**Custom config file**
```bash
python3 main.py --config /path/to/custom.env
```

**Single-lane demo (faster decisions on CPU-only hardware)**
```bash
LANE_COUNT=1 EVERY_N_FRAMES=10 COLLECT_TIMEOUT=60 python3 main.py
```
Running one lane is fast enough to observe live decisions in the dashboard (~15–20 s/cycle).

**Dashboard development (hot-reload)**
```bash
python3 main.py                        # backend + pipeline on :8000
cd dashboard-ui && npm run dev         # Vite dev server on :5173, proxied to :8000
```

**Run tests**
```bash
pytest tests/ -v        # all 63 tests, verbose
pytest tests/ -q        # summary only
```
No GPU, real video, or built frontend required — `tests/conftest.py` stubs the ML
stack so the full suite runs in under 3 seconds.

---

## Configuration

All settings are read from environment variables, loaded from `.env` at startup.

| Variable               | Default                      | Description                                                        |
|------------------------|------------------------------|--------------------------------------------------------------------|
| `MODEL_WEIGHTS`        | `data/weights/yolo11n.pt`    | Path to YOLO11 `.pt` weights file                                  |
| `CONFIDENCE_THRESHOLD` | `0.45`                       | Minimum detection confidence                                       |
| `VEHICLE_CLASSES`      | `car,motorcycle,bus,truck`   | COCO class names to count; others ignored                          |
| `LANE_COUNT`           | `4`                          | Number of lanes; expects `lane_0.mp4` … `lane_{N-1}.mp4`          |
| `VIDEO_DIR`            | `data/samples`               | Directory containing lane video files                              |
| `WINDOW_SECONDS`       | `5`                          | Duration of each processing window in seconds                      |
| `EVERY_N_FRAMES`       | `15`                         | Sample 1 in every N frames; higher = faster but coarser            |
| `COLLECT_TIMEOUT`      | `120`                        | Seconds to wait for all lanes before deciding; increase on slow CPUs |
| `DEADLOCK_THRESHOLD`   | `2`                          | Max consecutive greens before round-robin override activates       |
| `LOG_LEVEL`            | `INFO`                       | `DEBUG` / `INFO` / `WARNING` / `ERROR`                             |

---

## Web dashboard

The dashboard connects to the pipeline over WebSocket and updates in real time after
every signal decision. It is served by FastAPI at `http://localhost:8000`.

**Components:**
- **Signal Status** — one card per lane showing the signal dot (green glow / red),
  cumulative vehicle count, and consecutive-green badge
- **Vehicle Counts** — bar chart of current counts; winner bar highlighted in green
- **Signal History** — scrollable log of the last 20 decisions with timestamps

**Endpoints:**

| Endpoint      | Method | Description                                  |
|---------------|--------|----------------------------------------------|
| `/health`     | GET    | Liveness probe → `{"status": "ok"}`          |
| `/api/state`  | GET    | Latest lane snapshot for initial page load   |
| `/ws`         | WS     | Live stream of `signal_granted` events       |

---

## Reading the log output

Each signal decision produces a structured log line:

```
info  signal_granted  winner=2
  cumulative_counts={0: 3, 1: 4, 2: 7, 3: 3}
  window_received={0: 3, 1: 4, 2: 7, 3: 3}
  signals={0: 'red', 1: 'red', 2: 'green', 3: 'red'}
```

- `winner` — the lane receiving green this cycle
- `cumulative_counts` — each lane's accumulated count **before** the winner is reset
- `window_received` — raw counts posted by each lane process this window
- `signals` — current signal state for all lanes after the decision

When a lane wins, its `cumulative_count` is reset to 0. Losing lanes carry their
counts forward and keep accumulating until they win a future round.

---

## Demo guide

Recommended presentation sequence:

1. **Dry run** — shows asset validation, exits in < 1 s
2. **Single-lane live** — `LANE_COUNT=1 EVERY_N_FRAMES=10` while explaining the
   architecture; open `http://localhost:8000` and watch the dashboard update
3. **Pre-captured 4-lane log** — run the full system overnight and replay the log,
   grepping for winner rotations and deadlock override moments

> **CPU note:** On CPU-only machines the first 4-lane decision cycle may take
> 90–120 s. Use `LANE_COUNT=1` for live demos, or reduce `EVERY_N_FRAMES`.

---

## Project structure

```
ATMS/
├── main.py                    ← entry point: args, config, asset validation, manager
├── requirements.txt
├── .env.example               ← all configurable keys with defaults
├── CLAUDE.md                  ← project spec and build guide
│
├── src/
│   ├── dashboard/
│   │   ├── broadcaster.py     ← WebSocket broadcaster (sync↔async bridge)
│   │   └── server.py          ← FastAPI app: /health, /api/state, /ws, static files
│   │
│   ├── detection/
│   │   ├── detector.py        ← YOLO11 wrapper → List[Detection]
│   │   └── tracker.py         ← ByteTrack wrapper → track IDs per frame
│   │
│   ├── signal/
│   │   ├── controller.py      ← LaneState dataclass + SignalController.decide()
│   │   └── deadlock.py        ← DeadlockGuard: round-robin override logic
│   │
│   ├── pipeline/
│   │   ├── lane.py            ← LanePipeline: per-subprocess detect+track loop
│   │   └── manager.py         ← PipelineManager: spawn processes, collect, decide, broadcast
│   │
│   └── utils/
│       ├── config.py          ← Config dataclass + load_config()
│       ├── logger.py          ← structlog setup + get_logger()
│       └── video.py           ← iter_subclip_frames() + get_video_duration()
│
├── dashboard-ui/
│   ├── web/                   ← React source
│   │   ├── App.jsx            ← root component: WS connection, state, layout
│   │   ├── main.jsx
│   │   └── components/
│   │       ├── LaneCard.jsx      ← signal dot, count, consecutive-green badge
│   │       ├── CountChart.jsx    ← Recharts bar chart
│   │       └── SignalHistory.jsx ← last-20 decision log
│   ├── dist/                  ← built output served by FastAPI (after npm run build)
│   └── vite.config.js         ← proxy /api and /ws to :8000 in dev
│
├── tests/
│   ├── conftest.py            ← stubs ultralytics + torch; no GPU needed
│   ├── test_video.py          ← 10 tests
│   ├── test_detector.py       ← 9 tests
│   ├── test_tracker.py        ← 10 tests
│   ├── test_controller.py     ← 8 tests
│   ├── test_deadlock.py       ← 11 tests
│   └── test_pipeline.py       ← 15 tests
│
└── data/
    ├── samples/               ← lane_0.mp4 … lane_3.mp4 (not committed)
    └── weights/               ← yolo11n.pt (not committed)
```

---

## Tests

63 tests · 0 failures

| File                  | Tests | What is covered                                                |
|-----------------------|-------|----------------------------------------------------------------|
| `test_pipeline.py`    | 15    | Per-frame detect+track, queue output, multi-window accumulation|
| `test_deadlock.py`    | 11    | Override trigger, threshold edge cases, round-robin cycle      |
| `test_tracker.py`     | 10    | ByteTrack ID assignment, unique count, window reset, resilience|
| `test_video.py`       | 10    | Video loading, subclip windowing, frame sampling, error paths  |
| `test_detector.py`    | 9     | YOLO output parsing, class filtering, confidence threshold     |
| `test_controller.py`  | 8     | Winner selection, count reset, signal assignment               |

---

## Tech stack

| Layer       | Library                    | Version  | Role                                                  |
|-------------|----------------------------|----------|-------------------------------------------------------|
| Detection   | Ultralytics YOLO11         | ≥ 8.4    | Object detection; COCO-pretrained; ONNX-export capable|
| Tracking    | ByteTrack (via Ultralytics)| bundled  | Cross-frame vehicle ID assignment                     |
| Video       | OpenCV                     | ≥ 5.0    | Frame extraction and subclip windowing                |
| Concurrency | multiprocessing            | stdlib   | One `Process` per lane; `Queue` for result delivery   |
| API server  | FastAPI + uvicorn          | ≥ 0.111  | REST + WebSocket dashboard server                     |
| Frontend    | React + Vite               | 18 / 8   | Dashboard UI with hot-reload dev proxy                |
| Charts      | Recharts                   | ≥ 2.x    | Bar chart for lane vehicle counts                     |
| Config      | python-dotenv              | ≥ 1.0    | Loads `.env` at startup                               |
| Logging     | structlog                  | ≥ 26     | Structured key=value log events                       |
| Testing     | pytest + pytest-mock       | ≥ 8.0    | 63 tests; no GPU required                             |

---

## Hardware notes

If a CUDA-capable GPU is present, Ultralytics uses it automatically. On GPU,
per-frame inference drops from ~300 ms to under 10 ms.

On CPU-only machines, each subprocess is capped at `cpu_count // 4` PyTorch threads
to prevent oversubscription. Tune these together:

| Hardware                     | `EVERY_N_FRAMES` | Expected cycle time |
|------------------------------|------------------|---------------------|
| GPU (any)                    | 1–3              | 5–15 s              |
| CPU, 8+ fast cores           | 5–10             | 30–60 s             |
| CPU, 4–8 slower cores / WSL2 | 15–30            | 90–180 s            |

---

## Future scope

The following are **not implemented** in the current build:

- Emergency vehicle detection and signal preemption
- Automatic Number Plate Recognition (ANPR)
- Cloud integration or remote monitoring
- IoT hardware control (physical signal actuators)

---

*Computer Science capstone project · Python 3.11 · YOLO11 · ByteTrack · FastAPI · React*
