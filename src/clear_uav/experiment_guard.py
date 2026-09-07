"""Small, cross-platform guards for independent experiment runners.

Locks deliberately do not expire: a PID may belong to another host sharing the
same output directory. An abandoned lock must be inspected, not stolen.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import uuid


def assert_unlocked(path: Path, label: str) -> None:
    if path.exists():
        try:
            owner = path.read_text(encoding="utf-8")
        except OSError:
            owner = "owner record is not readable"
        raise BlockingIOError(
            f"{label} is locked: {path}. Another runner may still be active. "
            f"Do not start a duplicate or remove this lock until its owner has "
            f"been confirmed stopped on the recorded host. Owner: {owner}"
        )


@contextmanager
def stage_lock(path: Path, label: str):
    """Own one stage from its final preflight through all child processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    owner = {"stage": label, "host": socket.gethostname(), "pid": os.getpid(),
             "started_utc": datetime.now(timezone.utc).isoformat(), "token": token}
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        assert_unlocked(path, label)
        # The previous owner could release between open() and the diagnostic.
        raise BlockingIOError(f"{label} lock changed; rerun the same command safely.")
    try:
        with stream:
            json.dump(owner, stream)
        yield
    finally:
        # Never delete a replacement lock belonging to a different owner.
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("token") == token:
                path.unlink()
        except (FileNotFoundError, json.JSONDecodeError):
            pass


def artifact_identity(path: Path, root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    resolved = path.resolve()
    try:
        location = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        location = resolved.as_posix()
    return {"path": location, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def verify_artifact(path: Path, identity: dict, root: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Recorded artifact is missing: {path}")
    if artifact_identity(path, root) != identity:
        raise ValueError(f"Artifact differs from its completion receipt: {path}. "
                         "Preserve both files and investigate; do not silently reuse it.")


def write_json_atomic(path: Path, payload: dict) -> None:
    """Replace a runner-owned receipt without exposing partially written JSON."""
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
