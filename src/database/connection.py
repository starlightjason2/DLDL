"""Engine, declarative base, and SQLite bootstrap (pattern: fastapi-sqlalchemy-postgres-template ``database/connection.py``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import NullPool

PathLike = Union[str, Path]

TRIAL_LOG_DB_FILENAME = "trials_log.db"
SCHEMA_USER_VERSION = 1
TRIALS_TABLE = "trials"
_TO, _BT = 30.0, 30_000
_ENG: dict[str, Engine] = {}


class Base(DeclarativeBase):
    pass


class SchemaError(RuntimeError):
    pass


def default_trial_db_path(trials_dir: str) -> str:
    return os.path.join(trials_dir, TRIAL_LOG_DB_FILENAME)


def trial_db_engine(p: PathLike) -> Engine:
    path = Path(p).resolve()
    k = str(path)
    if k not in _ENG:
        eng = create_engine(
            f"sqlite:///{path.as_posix()}",
            connect_args={"timeout": _TO},
            poolclass=NullPool,
            pool_pre_ping=True,
        )

        @event.listens_for(eng, "connect")
        def _pragma(dbapi_conn, _):
            c = dbapi_conn.cursor()
            c.execute(f"PRAGMA busy_timeout = {_BT}")
            for s in (
                "PRAGMA foreign_keys = ON",
                "PRAGMA journal_mode = WAL",
                "PRAGMA synchronous = NORMAL",
            ):
                c.execute(s)
            c.close()

        _ENG[k] = eng
    return _ENG[k]


def sessionmaker_for(db_path: PathLike):
    """Like the template's ``SessionLocal`` factory, keyed by DB file path."""
    return sessionmaker(
        autocommit=False, autoflush=False, bind=trial_db_engine(db_path)
    )


def connect(p: PathLike):
    path = Path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    return trial_db_engine(path).connect()


def initialize(p: PathLike) -> None:
    from . import trial_model  # noqa: F401 — register ORM mappers on ``Base.metadata``

    path = Path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    eng = trial_db_engine(path)
    with eng.connect() as conn:
        v = int(conn.execute(text("PRAGMA user_version")).scalar() or 0)
        if v > SCHEMA_USER_VERSION:
            raise SchemaError(f"{path}: user_version={v} > {SCHEMA_USER_VERSION}")
        if v < SCHEMA_USER_VERSION:
            Base.metadata.create_all(eng)
            conn.execute(text(f"PRAGMA user_version = {SCHEMA_USER_VERSION}"))
            conn.commit()
        elif not conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n LIMIT 1"),
            {"n": TRIALS_TABLE},
        ).fetchone():
            raise SchemaError(f"missing {TRIALS_TABLE!r}")
