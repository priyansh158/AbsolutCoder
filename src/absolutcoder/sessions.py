"""SQLite session index and the LangGraph checkpoint saver."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from absolutcoder.config import data_dir

_SAVERS: dict[str, SqliteSaver] = {}


@dataclass(frozen=True)
class Session:
    """One chat thread bound to a repo and the models used for that thread."""

    id: str
    repo: str
    main_model: str
    subagent_model: str
    created_at: str
    auto_run: bool = False


def create_session(repo: Path | str, main_model: str, subagent_model: str, *, auto_run: bool = False) -> Session:
    """Start a session. Its id is the LangGraph thread id."""

    session = Session(
        id=uuid.uuid4().hex,
        repo=_repo_key(repo),
        main_model=main_model,
        subagent_model=subagent_model,
        created_at=_now(),
        auto_run=auto_run,
    )
    with _transaction() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, repo, main_model, subagent_model, auto_run, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.repo,
                session.main_model,
                session.subagent_model,
                int(session.auto_run),
                session.created_at,
            ),
        )
    return session


def list_sessions(repo: Path | str) -> list[Session]:
    """Sessions for this repo, newest first. Other repos are invisible."""

    key = _repo_key(repo)
    with _transaction() as conn:
        rows = conn.execute(
            """
            SELECT id, repo, main_model, subagent_model, auto_run, created_at
            FROM sessions
            WHERE repo = ?
            ORDER BY created_at DESC, rowid DESC
            """,
            (key,),
        ).fetchall()
    return [_session_from_row(row) for row in rows]


def load_session(session_id: str) -> Session:
    """Return the session, including the model pair saved on it."""

    with _transaction() as conn:
        row = conn.execute(
            """
            SELECT id, repo, main_model, subagent_model, auto_run, created_at
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        raise KeyError(session_id)
    return _session_from_row(row)


def update_session(
    session_id: str,
    *,
    main_model: str | None = None,
    subagent_model: str | None = None,
    auto_run: bool | None = None,
) -> Session:
    """Update the models or auto-run flag and return the stored session."""

    current = load_session(session_id)
    updated = Session(
        id=current.id,
        repo=current.repo,
        main_model=current.main_model if main_model is None else main_model,
        subagent_model=current.subagent_model if subagent_model is None else subagent_model,
        created_at=current.created_at,
        auto_run=current.auto_run if auto_run is None else auto_run,
    )
    with _transaction() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET main_model = ?, subagent_model = ?, auto_run = ?
            WHERE id = ?
            """,
            (updated.main_model, updated.subagent_model, int(updated.auto_run), updated.id),
        )
    return updated


def checkpoint_saver() -> SqliteSaver:
    """LangGraph saver for messages, tool results, and attachment paths.

    The saver lives in the same SQLite file as the session index and is reused
    for the process so a resumed thread reads the original checkpoint.
    """

    path = _db_path()
    key = str(path)
    saver = _SAVERS.get(key)
    if saver is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        saver = SqliteSaver(conn)
        saver.setup()
        _SAVERS[key] = saver
    return saver


def _db_path() -> Path:
    return data_dir() / "sessions.sqlite"


@contextmanager
def _transaction():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            main_model TEXT NOT NULL,
            subagent_model TEXT NOT NULL,
            auto_run INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        repo=row["repo"],
        main_model=row["main_model"],
        subagent_model=row["subagent_model"],
        created_at=row["created_at"],
        auto_run=bool(row["auto_run"]),
    )


def _repo_key(repo: Path | str) -> str:
    return str(Path(repo).resolve())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
