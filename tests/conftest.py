"""
Shared pytest configuration and stubs for heavy optional dependencies.

ultralytics and torch are large ML libraries that may not be present in
lightweight CI or dev environments. We stub them into sys.modules before
any test module is imported so that the production modules can be loaded
and patched normally in tests that use mocker.patch().
"""

import sys
from unittest.mock import MagicMock

# ── ultralytics stub ──────────────────────────────────────────────────────────
_ultralytics = MagicMock()
_ultralytics.YOLO = MagicMock

_trackers = MagicMock()
_trackers.BYTETracker = MagicMock

sys.modules.setdefault("ultralytics", _ultralytics)
sys.modules.setdefault("ultralytics.trackers", _trackers)
sys.modules.setdefault("ultralytics.trackers.byte_tracker", _trackers)

# ── torch stub ────────────────────────────────────────────────────────────────
# tracker.py imports torch only to build tensors inside _FakeResults.
# Tests mock BYTETracker entirely, so the real torch is never called.
_torch = MagicMock()
sys.modules.setdefault("torch", _torch)
