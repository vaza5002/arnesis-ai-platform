"""Thread-safe lifecycle management for one Arnesis processing group.

This module provides the independent lifecycle used by each real-time group.
Camera capture and CUDA inference components can be attached without changing
the public start, pause, resume, stop, or monitoring contracts.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable


class SessionState(str, Enum):
    """Supported lifecycle states for a real-time processing group."""

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class GroupSessionConfiguration:
    """Immutable runtime configuration for one processing group."""

    group_id: int
    group_code: str
    group_name: str
    gpu_index: int
    gpu_name: str
    requested_memory_mb: int
    maximum_streams: int


@dataclass(frozen=True, slots=True)
class GroupSessionSnapshot:
    """Serializable monitoring snapshot for one processing group."""

    group_id: int
    group_code: str
    group_name: str
    state: str
    cuda_device: str
    requested_memory_mb: int
    maximum_streams: int
    started_at: str | None
    updated_at: str
    uptime_seconds: float
    loop_iterations: int
    last_error: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary suitable for controller and API responses."""
        return asdict(self)


StateCallback = Callable[[int, SessionState, str | None], None]


class GroupSession:
    """Own the thread-safe execution state of one configured group."""

    def __init__(
        self,
        configuration: GroupSessionConfiguration,
        state_callback: StateCallback | None = None,
    ) -> None:
        self.configuration = configuration
        self._state_callback = state_callback
        self._state = SessionState.STOPPED
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._started_event = threading.Event()
        self._lock = threading.RLock()
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._updated_at = self._utc_now()
        self._loop_iterations = 0
        self._last_error: str | None = None

    @property
    def state(self) -> SessionState:
        """Return the current lifecycle state."""
        with self._lock:
            return self._state

    @property
    def is_alive(self) -> bool:
        """Return whether the group worker thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, timeout_seconds: float = 10.0) -> GroupSessionSnapshot:
        """Start the group session or return its existing active snapshot."""
        with self._lock:
            if self._state in {
                SessionState.STARTING,
                SessionState.RUNNING,
                SessionState.PAUSING,
                SessionState.PAUSED,
            }:
                return self.snapshot()

            if self.is_alive:
                raise RuntimeError(
                    f"Group {self.configuration.group_code} still has an "
                    "active thread."
                )

            self._stop_event.clear()
            self._pause_event.clear()
            self._started_event.clear()
            self._last_error = None
            self._loop_iterations = 0
            self._started_at = self._utc_now()
            self._stopped_at = None
            self._set_state(SessionState.STARTING)
            self._thread = threading.Thread(
                target=self._run,
                name=f"arnesis-group-{self.configuration.group_code}",
                daemon=False,
            )
            self._thread.start()

        if not self._started_event.wait(timeout_seconds):
            self.stop(timeout_seconds=timeout_seconds)
            raise TimeoutError(
                f"Group {self.configuration.group_code} did not start within "
                f"{timeout_seconds:.1f} seconds."
            )

        return self.snapshot()

    def pause(self) -> GroupSessionSnapshot:
        """Pause an active session without releasing its configuration."""
        with self._lock:
            if self._state == SessionState.PAUSED:
                return self.snapshot()

            if self._state != SessionState.RUNNING:
                raise RuntimeError(
                    f"Group {self.configuration.group_code} cannot pause from "
                    f"state {self._state.value}."
                )

            self._set_state(SessionState.PAUSING)
            self._pause_event.set()
            self._set_state(SessionState.PAUSED)
            return self.snapshot()

    def resume(self) -> GroupSessionSnapshot:
        """Resume a paused session."""
        with self._lock:
            if self._state == SessionState.RUNNING:
                return self.snapshot()

            if self._state != SessionState.PAUSED:
                raise RuntimeError(
                    f"Group {self.configuration.group_code} cannot resume from "
                    f"state {self._state.value}."
                )

            self._pause_event.clear()
            self._set_state(SessionState.RUNNING)
            return self.snapshot()

    def stop(self, timeout_seconds: float = 15.0) -> GroupSessionSnapshot:
        """Stop the group thread and preserve its final monitoring metrics."""
        with self._lock:
            if self._state == SessionState.STOPPED and not self.is_alive:
                return self.snapshot()

            if self._state != SessionState.ERROR:
                self._set_state(SessionState.STOPPING)

            self._stop_event.set()
            self._pause_event.clear()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise TimeoutError(
                    f"Group {self.configuration.group_code} did not stop "
                    f"within {timeout_seconds:.1f} seconds."
                )

        with self._lock:
            self._thread = None
            if self._stopped_at is None:
                self._stopped_at = self._utc_now()
            if self._state != SessionState.ERROR:
                self._set_state(SessionState.STOPPED)
            return self.snapshot()

    def snapshot(self) -> GroupSessionSnapshot:
        """Return a consistent snapshot of the current session state."""
        with self._lock:
            uptime = self._calculate_uptime_seconds()

            return GroupSessionSnapshot(
                group_id=self.configuration.group_id,
                group_code=self.configuration.group_code,
                group_name=self.configuration.group_name,
                state=self._state.value,
                cuda_device=(
                    f"CUDA:{self.configuration.gpu_index} - "
                    f"{self.configuration.gpu_name}"
                ),
                requested_memory_mb=(
                    self.configuration.requested_memory_mb
                ),
                maximum_streams=self.configuration.maximum_streams,
                started_at=(
                    self._started_at.isoformat(timespec="seconds")
                    if self._started_at
                    else None
                ),
                updated_at=self._updated_at.isoformat(timespec="seconds"),
                uptime_seconds=round(uptime, 3),
                loop_iterations=self._loop_iterations,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        try:
            self._set_state(SessionState.RUNNING)
            self._started_event.set()

            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    self._stop_event.wait(0.10)
                    continue

                # Camera capture and CUDA inference will replace this heartbeat
                # without changing the lifecycle contract.
                with self._lock:
                    self._loop_iterations += 1
                    self._updated_at = self._utc_now()

                self._stop_event.wait(0.10)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._stopped_at = self._utc_now()
                self._set_state(SessionState.ERROR, self._last_error)
            self._started_event.set()
        finally:
            with self._lock:
                if self._stopped_at is None:
                    self._stopped_at = self._utc_now()
                if self._state not in {
                    SessionState.ERROR,
                    SessionState.STOPPED,
                }:
                    self._set_state(SessionState.STOPPED)

    def _calculate_uptime_seconds(self) -> float:
        """Calculate live or final elapsed session time without resetting it."""
        if self._started_at is None:
            return 0.0

        end_time = self._stopped_at or self._utc_now()
        return max(0.0, (end_time - self._started_at).total_seconds())

    def _set_state(
        self,
        state: SessionState,
        error: str | None = None,
    ) -> None:
        self._state = state
        self._updated_at = self._utc_now()
        callback = self._state_callback
        if callback is not None:
            callback(self.configuration.group_id, state, error)

    @staticmethod
    def _utc_now() -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        return datetime.now(timezone.utc)
