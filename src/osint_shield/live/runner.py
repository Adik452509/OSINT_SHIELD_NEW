"""Run pipeline cycles in the background of the API server (M10c).

The dashboard's Operations tab starts and stops a schedule, or asks for one
cycle now. Cycles run on a worker thread so API requests stay responsive; one
cycle at a time per process, and the database lease (``LivePipeline.run_cycle``)
keeps a second process - ``11_live.py run`` in another window - from running at
the same moment.

    runner = CycleRunner(lambda: LivePipeline.from_config(cfg), every_minutes=15)
    runner.start()      # a cycle now, then every 15 minutes
    runner.run_now()    # one extra cycle, unless one is running
    runner.stop()       # the cycle in progress finishes; no new one starts
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class CycleRunner:
    """Owns one pipeline (created lazily: loading it costs seconds) and its schedule."""

    def __init__(self, pipeline_factory: Callable, *, every_minutes: float = 15.0):
        self._factory = pipeline_factory
        self._pipeline = None
        self.every_minutes = float(every_minutes)
        self._cycle_lock = threading.Lock()     # one cycle at a time in this process
        self._stop = threading.Event()
        self._loop: threading.Thread | None = None
        self.last_report: dict | None = None
        self.last_error: str | None = None
        self.last_started: str | None = None
        self.last_finished: str | None = None
        self.next_run_at: str | None = None

    # ------------------------------------------------------------------ state
    @property
    def scheduled(self) -> bool:
        return self._loop is not None and self._loop.is_alive() and not self._stop.is_set()

    @property
    def busy(self) -> bool:
        return self._cycle_lock.locked()

    def status(self) -> dict:
        return {"scheduled": self.scheduled, "busy": self.busy,
                "every_minutes": self.every_minutes, "next_run_at": self.next_run_at,
                "last_started": self.last_started, "last_finished": self.last_finished,
                "last_error": self.last_error}

    # ------------------------------------------------------------------ cycles
    def _cycle(self) -> None:
        """One cycle, if none is running here. Never raises: errors land in status."""
        if not self._cycle_lock.acquire(blocking=False):
            return
        try:
            self.last_started = _iso(datetime.now(timezone.utc))
            if self._pipeline is None:
                self._pipeline = self._factory()
            self.last_report = self._pipeline.run_cycle()
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - the server must survive a bad cycle
            self.last_error = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            self.last_finished = _iso(datetime.now(timezone.utc))
            self._cycle_lock.release()

    def run_now(self) -> bool:
        """Start one cycle on a worker thread. False if one is already running."""
        if self.busy:
            return False
        threading.Thread(target=self._cycle, name="cycle-now", daemon=True).start()
        return True

    def _loop_body(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self._cycle()
            wait = max(0.0, self.every_minutes * 60 - (time.monotonic() - started))
            self.next_run_at = _iso(datetime.now(timezone.utc) + timedelta(seconds=wait))
            if self._stop.wait(wait):
                break
        self.next_run_at = None

    def start(self, every_minutes: float | None = None) -> bool:
        """Start the schedule (first cycle immediately). False if already running."""
        if self.scheduled:
            return False
        if every_minutes:
            self.every_minutes = float(every_minutes)
        self._stop.clear()
        self._loop = threading.Thread(target=self._loop_body, name="cycle-loop", daemon=True)
        self._loop.start()
        return True

    def stop(self) -> bool:
        """Stop the schedule. A cycle in progress finishes. False if not running."""
        if not self.scheduled:
            return False
        self._stop.set()
        return True

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.join(timeout)
        if self._pipeline is not None:
            self._pipeline.close()
