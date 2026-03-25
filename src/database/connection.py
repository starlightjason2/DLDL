"""Engine, declarative base, and SQLite bootstrap (pattern: fastapi-sqlalchemy-postgres-template ``database/connection.py``)."""

from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Union, cast

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine


from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    encoding="utf-8",
)

_TO, _BT = 30.0, 30_000

os.makedirs(Path(os.environ.get("DB_CONNECTION")).parent, exist_ok=True)
engine = create_engine(
    os.environ.get("DB_CONNECTION"), connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(engine, "connect")
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


@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
