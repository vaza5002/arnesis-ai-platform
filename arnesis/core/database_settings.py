"""Database path and connection settings for Arnesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Provide centralized database configuration for the application."""

    project_root: Path
    database_directory_name: str = "data"
    database_file_name: str = "arnesis.db"

    @property
    def database_directory(self) -> Path:
        """Return the directory that stores application data."""
        return self.project_root / self.database_directory_name

    @property
    def database_file(self) -> Path:
        """Return the absolute SQLite database path."""
        return self.database_directory / self.database_file_name

    @property
    def sqlalchemy_url(self) -> str:
        """Return the SQLAlchemy connection URL for SQLite."""
        normalized_path = self.database_file.resolve().as_posix()
        return f"sqlite:///{normalized_path}"

    def ensure_directories(self) -> None:
        """Create the application data directory when it does not exist."""
        self.database_directory.mkdir(parents=True, exist_ok=True)


def get_default_database_settings() -> DatabaseSettings:
    """Return database settings based on the repository root."""
    project_root = Path(__file__).resolve().parents[2]
    return DatabaseSettings(project_root=project_root)
