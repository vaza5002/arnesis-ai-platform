"""Concurrent in-process registry for Arnesis group sessions."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from arnesis.processing.group_session import (
    GroupSession,
    GroupSessionConfiguration,
    GroupSessionSnapshot,
    SessionState,
    StateCallback,
)


class SessionNotFoundError(KeyError):
    """Raised when a runtime command references an unknown group session."""


class ProcessingRuntime:
    """Starts and controls multiple groups within one Arnesis instance."""

    def __init__(self, state_callback: StateCallback | None = None) -> None:
        self._state_callback = state_callback
        self._sessions: dict[int, GroupSession] = {}
        self._lock = threading.RLock()

    def start_group(
        self, configuration: GroupSessionConfiguration
    ) -> GroupSessionSnapshot:
        with self._lock:
            existing = self._sessions.get(configuration.group_id)
            if existing is not None:
                if existing.configuration != configuration and existing.is_alive:
                    raise RuntimeError(
                        "The running group configuration cannot be replaced. "
                        "Stop the group before changing its CUDA allocation."
                    )
                if existing.state in {
                    SessionState.STARTING,
                    SessionState.RUNNING,
                    SessionState.PAUSING,
                    SessionState.PAUSED,
                }:
                    return existing.snapshot()

            session = GroupSession(configuration, self._state_callback)
            self._sessions[configuration.group_id] = session

        try:
            return session.start()
        except Exception:
            with self._lock:
                if self._sessions.get(configuration.group_id) is session:
                    self._sessions.pop(configuration.group_id, None)
            raise

    def start_groups(
        self, configurations: Iterable[GroupSessionConfiguration]
    ) -> list[GroupSessionSnapshot]:
        snapshots: list[GroupSessionSnapshot] = []
        started_ids: list[int] = []
        try:
            for configuration in configurations:
                snapshot = self.start_group(configuration)
                snapshots.append(snapshot)
                started_ids.append(configuration.group_id)
            return snapshots
        except Exception:
            for group_id in reversed(started_ids):
                try:
                    self.stop_group(group_id)
                except Exception:
                    pass
            raise

    def pause_group(self, group_id: int) -> GroupSessionSnapshot:
        return self._require(group_id).pause()

    def resume_group(self, group_id: int) -> GroupSessionSnapshot:
        return self._require(group_id).resume()

    def stop_group(self, group_id: int) -> GroupSessionSnapshot:
        session = self._require(group_id)
        snapshot = session.stop()
        with self._lock:
            if not session.is_alive:
                self._sessions.pop(group_id, None)
        return snapshot

    def stop_all(self) -> list[GroupSessionSnapshot]:
        with self._lock:
            group_ids = list(self._sessions)

        results: list[GroupSessionSnapshot] = []
        errors: list[str] = []
        for group_id in group_ids:
            try:
                results.append(self.stop_group(group_id))
            except Exception as exc:
                errors.append(f"group {group_id}: {type(exc).__name__}: {exc}")

        if errors:
            raise RuntimeError("Unable to stop all groups: " + "; ".join(errors))
        return results

    def get_group(self, group_id: int) -> GroupSessionSnapshot:
        return self._require(group_id).snapshot()

    def list_groups(self) -> list[GroupSessionSnapshot]:
        with self._lock:
            sessions = list(self._sessions.values())
        return sorted(
            (session.snapshot() for session in sessions),
            key=lambda snapshot: snapshot.group_code,
        )

    def contains(self, group_id: int) -> bool:
        with self._lock:
            return group_id in self._sessions

    def _require(self, group_id: int) -> GroupSession:
        with self._lock:
            session = self._sessions.get(group_id)
        if session is None:
            raise SessionNotFoundError(
                f"No active processing session exists for group id {group_id}."
            )
        return session
