"""Engine lifecycle: start/stop/status for the Kubera Alpaca daemon.

The daemon is a DETACHED subprocess so it survives:
  - the API server restarting
  - the terminal that launched it closing
  - the Claude session ending

State is persisted in alpaca_live.sqlite::alpaca_engine_state (singleton
row id=1).

Usage from the API layer:
  state = engine_state.status()         # current row + alive-check
  engine_state.start()                  # idempotent: refuses if already running
  engine_state.stop()                   # idempotent: best-effort terminate
"""
from __future__ import annotations

import datetime as dt
import os
import signal as signal_module
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DB_PATH = "data/processed/alpaca_live.sqlite"
LOG_PATH = "logs/alpaca_engine.log"
SYNC_LOG_PATH = "logs/alpaca_sync.log"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


SCHEMA = """
CREATE TABLE IF NOT EXISTS alpaca_engine_state (
  id              INTEGER PRIMARY KEY CHECK (id = 1),
  status          TEXT NOT NULL,       -- 'stopped' | 'running' | 'error'
  pid             INTEGER,
  sync_pid        INTEGER,             -- separate pid for the sync loop
  started_at      TEXT,
  last_run_at     TEXT,
  last_run_date   TEXT,
  last_run_status TEXT,
  last_error      TEXT,
  heartbeat_at    TEXT,
  stopped_at      TEXT
);
INSERT OR IGNORE INTO alpaca_engine_state (id, status) VALUES (1, 'stopped');
"""


@dataclass
class EngineStatus:
    status: str                          # 'stopped' | 'running' | 'error'
    pid: Optional[int]
    sync_pid: Optional[int]
    started_at: Optional[str]
    last_run_at: Optional[str]
    last_run_date: Optional[str]
    last_run_status: Optional[str]
    last_error: Optional[str]
    heartbeat_at: Optional[str]
    stopped_at: Optional[str]
    engine_alive: bool                   # whether the OS process actually exists
    sync_alive: bool


def _connect() -> sqlite3.Connection:
    p = Path(DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p.as_posix(), timeout=30.0)
    con.executescript(SCHEMA)
    con.commit()
    return con


def _pid_alive(pid: Optional[int]) -> bool:
    """Cross-platform 'is this PID alive?' check.

    On Windows, os.kill(pid, 0) raises OSError for dead PIDs.
    On POSIX, os.kill(pid, 0) raises ProcessLookupError for dead PIDs.
    """
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _kill(pid: Optional[int]) -> None:
    if not pid or pid <= 0:
        return
    if not _pid_alive(pid):
        return
    if sys.platform == "win32":
        # SIGTERM on Windows maps to TerminateProcess immediately; SIGBREAK
        # is the closer analog to graceful Ctrl-C. Try SIGBREAK first, then
        # fall through to a hard taskkill.
        try:
            os.kill(pid, signal_module.SIGBREAK)  # type: ignore[attr-defined]
            time.sleep(1.5)
        except Exception:
            pass
        if _pid_alive(pid):
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, check=False)
    else:
        try:
            os.kill(pid, signal_module.SIGTERM)
            time.sleep(1.5)
        except Exception:
            pass
        if _pid_alive(pid):
            try:
                os.kill(pid, signal_module.SIGKILL)
            except Exception:
                pass


def _spawn_detached(module: str, log_path: str) -> int:
    """Launch `python -m <module>` as a fully detached process. Returns PID."""
    log_p = Path(log_path)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_p, "a", encoding="utf-8")

    # Use the same Python interpreter as the parent (the venv).
    cmd = [sys.executable, "-m", module]

    kwargs: dict = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
    }
    if sys.platform == "win32":
        # DETACHED_PROCESS (0x08) and CREATE_NEW_PROCESS_GROUP (0x200) so the
        # child outlives this Python process AND its terminal session.
        DETACHED = 0x00000008
        NEW_PGRP = 0x00000200
        NO_WIN  = 0x08000000  # CREATE_NO_WINDOW
        kwargs["creationflags"] = DETACHED | NEW_PGRP | NO_WIN
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    return proc.pid


def status() -> EngineStatus:
    con = _connect()
    try:
        row = con.execute(
            """SELECT status, pid, sync_pid, started_at, last_run_at, last_run_date,
                       last_run_status, last_error, heartbeat_at, stopped_at
                FROM alpaca_engine_state WHERE id = 1"""
        ).fetchone()
        if row is None:
            row = ("stopped", None, None, None, None, None, None, None, None, None)
        pid = row[1]
        sync_pid = row[2]
        engine_alive = _pid_alive(pid)
        sync_alive = _pid_alive(sync_pid)

        # If we think we're running but the pid is dead, reconcile to 'stopped'.
        new_status = row[0]
        if new_status == "running" and not engine_alive:
            con.execute(
                "UPDATE alpaca_engine_state SET status='stopped', stopped_at=? WHERE id=1",
                (dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",),
            )
            con.commit()
            new_status = "stopped"

        return EngineStatus(
            status=new_status, pid=pid, sync_pid=sync_pid,
            started_at=row[3], last_run_at=row[4], last_run_date=row[5],
            last_run_status=row[6], last_error=row[7],
            heartbeat_at=row[8], stopped_at=row[9],
            engine_alive=engine_alive, sync_alive=sync_alive,
        )
    finally:
        con.close()


def start() -> EngineStatus:
    """Start the engine + sync loop as detached processes.

    Refuses if either is already alive.
    """
    cur = status()
    if cur.engine_alive:
        raise RuntimeError(
            f"engine already running (pid={cur.pid}). Stop it first."
        )

    # Kill stale sync if record is dead
    sync_pid: Optional[int] = cur.sync_pid if cur.sync_alive else None
    if sync_pid is None:
        sync_pid = _spawn_detached("services.alpaca.sync", SYNC_LOG_PATH)

    engine_pid = _spawn_detached("services.alpaca.engine", LOG_PATH)
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    con = _connect()
    try:
        con.execute(
            """UPDATE alpaca_engine_state SET
                 status = 'running',
                 pid = ?, sync_pid = ?, started_at = ?,
                 stopped_at = NULL, last_error = NULL
               WHERE id = 1""",
            (engine_pid, sync_pid, now),
        )
        con.commit()
    finally:
        con.close()

    # Tiny pause so the child has time to write its first log line.
    time.sleep(0.5)
    return status()


def stop() -> EngineStatus:
    """Stop the engine + sync loop. Idempotent."""
    cur = status()
    _kill(cur.pid)
    _kill(cur.sync_pid)
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    con = _connect()
    try:
        con.execute(
            """UPDATE alpaca_engine_state SET
                 status = 'stopped', stopped_at = ?,
                 pid = NULL, sync_pid = NULL
               WHERE id = 1""",
            (now,),
        )
        con.commit()
    finally:
        con.close()
    return status()
