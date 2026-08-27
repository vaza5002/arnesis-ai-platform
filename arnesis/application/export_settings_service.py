"""Persistent configuration for anonymous station CSV exports."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from sqlalchemy import select

from arnesis.domain.entities import SystemSetting


@dataclass(frozen=True, slots=True)
class ExportSettings:
    enabled: bool
    output_root: str
    flush_interval_seconds: int
    encoding: str
    delimiter: str


class ExportSettingsService:
    KEY_ENABLED = "csv_export_enabled"
    KEY_ROOT = "csv_output_root"
    KEY_INTERVAL = "csv_flush_interval_seconds"
    KEY_ENCODING = "csv_encoding"
    KEY_DELIMITER = "csv_delimiter"

    def __init__(self, database) -> None:
        self.database = database

    def ensure_defaults(self) -> ExportSettings:
        defaults = {
            self.KEY_ENABLED: "true",
            self.KEY_ROOT: self._default_output_root(),
            self.KEY_INTERVAL: "60",
            self.KEY_ENCODING: "utf-8-sig",
            self.KEY_DELIMITER: ",",
        }
        with self.database.session_scope() as session:
            for key, value in defaults.items():
                if session.get(SystemSetting, key) is None:
                    session.add(SystemSetting(
                        setting_key=key,
                        setting_value=value,
                        description="Arnesis configurable CSV station metrics export.",
                    ))
        return self.get()

    def get(self) -> ExportSettings:
        with self.database.session_scope() as session:
            items = session.scalars(select(SystemSetting).where(
                SystemSetting.setting_key.in_((
                    self.KEY_ENABLED, self.KEY_ROOT, self.KEY_INTERVAL,
                    self.KEY_ENCODING, self.KEY_DELIMITER,
                )))).all()
        values = {item.setting_key: item.setting_value for item in items}
        return ExportSettings(
            enabled=values.get(self.KEY_ENABLED, "true").casefold() == "true",
            output_root=values.get(self.KEY_ROOT, self._default_output_root()),
            flush_interval_seconds=int(values.get(self.KEY_INTERVAL, "60")),
            encoding=values.get(self.KEY_ENCODING, "utf-8-sig"),
            delimiter=values.get(self.KEY_DELIMITER, ","),
        )

    def save(self, *, enabled: bool, output_root: str,
             flush_interval_seconds: int) -> ExportSettings:
        path = self.validate_output_root(output_root)
        interval = int(flush_interval_seconds)
        if interval < 5 or interval > 3600:
            raise ValueError("Flush interval must be between 5 and 3600 seconds.")
        values = {
            self.KEY_ENABLED: "true" if enabled else "false",
            self.KEY_ROOT: str(path),
            self.KEY_INTERVAL: str(interval),
            self.KEY_ENCODING: "utf-8-sig",
            self.KEY_DELIMITER: ",",
        }
        with self.database.session_scope() as session:
            for key, value in values.items():
                item = session.get(SystemSetting, key)
                if item is None:
                    item = SystemSetting(setting_key=key, setting_value=value)
                    session.add(item)
                else:
                    item.setting_value = value
        return self.get()

    def validate_output_root(self, value: str) -> Path:
        if not value.strip():
            raise ValueError("CSV output folder is required.")
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        for child in ("Occupancy", "Performance", "Combined"):
            (path / child).mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", dir=path, prefix="arnesis_write_test_",
                                suffix=".tmp", delete=False, encoding="utf-8") as stream:
            stream.write("write-test")
            temporary = Path(stream.name)
        temporary.unlink(missing_ok=True)
        return path

    @staticmethod
    def _default_output_root() -> str:
        configured = os.getenv("ARNESIS_CSV_OUTPUT_ROOT", "").strip()
        if configured:
            return str(Path(configured).expanduser().resolve())
        return str(Path("D:/Arnesis/Data/Exports").resolve())
