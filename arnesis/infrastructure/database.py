"""SQLAlchemy database infrastructure for Arnesis.

The implementation uses SQLite for the initial deployment while keeping the
application layer isolated from the database engine. Identifiers, constraints,
and schema conventions are intentionally suitable for future Oracle 11g
migration.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, event
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from arnesis.core.database_settings import (
    DatabaseSettings,
    get_default_database_settings,
)


class Base(DeclarativeBase):
    """Base class for all Arnesis ORM entities."""


class DatabaseManager:
    """Own the SQLAlchemy engine and transactional session factory."""

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self._settings = settings or get_default_database_settings()
        self._settings.ensure_directories()
        self._engine = self._create_engine()
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
            class_=Session,
        )

    @property
    def engine(self) -> Engine:
        """Return the configured SQLAlchemy engine."""
        return self._engine

    def create_schema(self) -> None:
        """Create all registered database tables."""
        Base.metadata.create_all(bind=self._engine)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Provide a transactional session with automatic rollback."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Release pooled database resources."""
        self._engine.dispose()

    def _create_engine(self) -> Engine:
        engine = create_engine(
            self._settings.sqlalchemy_url,
            future=True,
            pool_pre_ping=True,
        )

        if engine.dialect.name == "sqlite":
            event.listen(engine, "connect", self._configure_sqlite_connection)

        return engine

    @staticmethod
    def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.close()
