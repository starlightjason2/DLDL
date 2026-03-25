"""Engine, declarative base, and SQLite bootstrap (pattern: fastapi-sqlalchemy-postgres-template ``database/connection.py``)."""

from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import make_url

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    encoding="utf-8",
)

_BT = 30_000

db_connection = os.environ["DB_CONNECTION"]
url = make_url(db_connection)
if url.drivername == "sqlite" and url.database and url.database != ":memory:":
    db_path = Path(url.database)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    db_connection, connect_args={"check_same_thread": False}
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
