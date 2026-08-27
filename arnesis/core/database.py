"""Database infrastructure for Arnesis.

Provides a single SQLAlchemy configuration point. SQLite is used for the
initial deployment while the schema avoids SQLite-specific column types so it
can be migrated more easily to Oracle 11g later.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Declarative base shared by every persistent entity."""


class DatabaseManager:
    """Owns the SQLAlchemy engine and session factory for the application."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or self._default_database_url()
        self.engine = self._create_engine(self.database_url)
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    @staticmethod
    def _default_database_url() -> str:
        configured_url = os.getenv("ARNESIS_DATABASE_URL", "").strip()
        if configured_url:
            return configured_url

        project_root = Path(__file__).resolve().parents[2]
        data_directory = project_root / "data"
        data_directory.mkdir(parents=True, exist_ok=True)
        database_path = (data_directory / "arnesis.db").resolve()
        return f"sqlite:///{database_path.as_posix()}"

    @staticmethod
    def _create_engine(database_url: str) -> Engine:
        url = make_url(database_url)
        options: dict[str, object] = {
            "pool_pre_ping": True,
            "future": True,
        }

        if url.get_backend_name() == "sqlite":
            options["connect_args"] = {"check_same_thread": False}

        engine = create_engine(database_url, **options)

        if url.get_backend_name() == "sqlite":
            event.listen(engine, "connect", DatabaseManager._enable_sqlite_foreign_keys)

        return engine

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    def create_schema(self) -> None:
        """Create all registered tables without deleting existing data."""
        from arnesis.domain.entities import (  # noqa: F401
            Camera,
            GpuCapacity,
            Group,
            ModelDefinition,
            ProcessingProfile,
            Roi,
            StationMetricSample,
            SystemSetting,
        )

        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transaction that commits or rolls back atomically."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Release pooled database connections."""
        self.engine.dispose()
