# ATMS — Architecture Reference

## Data flow

```
lane_0.mp4 ──┐
lane_1.mp4 ──┤──► pipeline/manager.py
lane_2.mp4 ──┤      │
lane_3.mp4 ──┘      │  spawns 4x
                     ▼
              pipeline/lane.py  (one thread per lane)
                     │
                     │  every WINDOW_SECONDS
                     ▼
              utils/video.py          ← extract subclip frames
                     │
                     ▼
              detection/detector.py   ← YOLO11 inference
                     │
                     ▼
              detection/tracker.py    ← ByteTrack IDs
                     │
                     ▼
              LaneState.cumulative_count += new_vehicles
                     │
                     │  all lanes post counts
                     ▼
              signal/controller.py    ← pick winner lane
                     │
                     ▼
              signal/deadlock.py      ← override if threshold hit
                     │
                     ▼
              GREEN signal issued → winner.count reset to 0
```

## Thread model

```
Main thread
  └── Manager
        ├── Lane-0 thread  (read → detect → track → post)
        ├── Lane-1 thread
        ├── Lane-2 thread
        └── Lane-3 thread

Shared state: List[LaneState] protected by threading.Lock
Signal decisions happen on the main thread after all lanes post.
```

## Key invariants

1. `cumulative_count` only resets to 0 when a lane receives green.
2. `consecutive_greens` resets to 0 for all non-winning lanes each round.
3. If `consecutive_greens >= DEADLOCK_THRESHOLD`, the controller skips
   the winner and grants green to the next lane without a green this round.
4. Vehicle IDs from ByteTrack prevent double-counting within a window.
