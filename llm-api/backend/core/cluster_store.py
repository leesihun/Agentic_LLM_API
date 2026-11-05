"""
Persistent cluster registry and task queue.

The master LLM API owns this store. Slaves register, heartbeat, lease work,
stream events, and complete tasks through the /api/cluster routes.

Storage layout (under config.CLUSTER_DIR):
    nodes.json                     node registry (atomic rewrite)
    cluster.lock                   lock for nodes.json
    tasks/{task_id}.json           task record — the single source of truth
    tasks/{task_id}.lock           per-task lock
    tasks/{task_id}.events.jsonl   append-only task event log
    index/queued/{key}__{task_id}  hint: task is queued (key = created_at, µs)
    index/leased/{task_id}         hint: task is leased (scanned for expiry)
    artifacts/{task_id}/...        files uploaded by slaves

The index files are hints only. Every state transition re-reads the task
record under its lock, so a stale/missing marker can never cause a double
lease; it only costs a skipped or extra read. A periodic reconcile pass adds
markers for any task the index missed (crash between writes, legacy data).

All writes are atomic (tmp file + fsync + os.replace, retried on Windows
sharing violations), so readers never see a torn record and lock-free reads
are safe. Unparseable records are quarantined (renamed *.corrupt-<ts>) and
logged instead of silently disappearing.

Every method here is synchronous and may block on file locks — the routes
call them via asyncio.to_thread so the master event loop (which also serves
every chat stream) is never blocked.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout as FileLockTimeout

import config


# Mirrored by the slave-side upload limits in hoonbot/core/cluster_worker.py.
MAX_ARTIFACTS_PER_TASK = 10
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024

_TERMINAL = {"completed", "failed", "cancelled"}
_LOCK_TIMEOUT = 10            # seconds; normal per-task / registry operations
_CLAIM_LOCK_TIMEOUT = 2       # seconds; a busy candidate is skipped during lease
_EXPIRY_SCAN_INTERVAL = 5.0   # seconds between expired-lease scans (per process)
_RECONCILE_INTERVAL = 600.0   # seconds between full index reconcile passes
_CACHE_MAX_RECORD_BYTES = 256 * 1024  # larger task records are not kept in the read cache
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class LeaseConflictError(Exception):
    """A worker acted on a task it no longer holds (re-leased elsewhere, or
    already finished by another node). Routes map this to HTTP 409."""


# Re-exported so the routes can map lock contention to 503 without importing
# filelock themselves.
StoreBusyError = FileLockTimeout


def _log(message: str) -> None:
    print(f"[Cluster] {message}", flush=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _valid_task_id(task_id: Any) -> bool:
    return isinstance(task_id, str) and bool(_TASK_ID_RE.match(task_id))


def _sort_key(created_at: str | None) -> str:
    """Fixed-width, lexically sortable key from created_at (µs since epoch)."""
    dt = _parse_iso(created_at) or _now()
    micros = max(0, (dt - _EPOCH) // timedelta(microseconds=1))
    return f"{micros:020d}"


def _replace_with_retry(src: str, dst: str, attempts: int = 8) -> None:
    """os.replace, retried on Windows sharing violations (a reader holding the
    destination open). On POSIX the first attempt either works or raises."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _fsync_dir(path: Path) -> None:
    if sys.platform.startswith("win"):
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(str(tmp), str(path))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def _touch(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8"):
            pass
    except OSError as exc:
        _log(f"WARNING: could not create index marker {path.name}: {exc}")


def _unlink(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        _log(f"WARNING: could not remove {path.name}: {exc}")


class ClusterStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir or config.CLUSTER_DIR)
        self.tasks_dir = self.base_dir / "tasks"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.index_dir = self.base_dir / "index"
        self.queued_dir = self.index_dir / "queued"
        self.leased_dir = self.index_dir / "leased"
        self.nodes_file = self.base_dir / "nodes.json"
        self.lock_file = self.base_dir / "cluster.lock"
        for d in (self.base_dir, self.tasks_dir, self.artifacts_dir, self.queued_dir, self.leased_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Per-process bookkeeping (routes run in a threadpool → guard it).
        self._state_lock = threading.Lock()
        self._last_reconcile = 0.0
        self._last_expiry_scan = 0.0
        # Lock-free read cache: task_id -> ((ino, mtime_ns, size), task dict).
        # Atomic replace gives every version a new inode, so the key changes
        # on every write; used only for listings and lease pre-filtering.
        self._read_cache: dict[str, tuple[tuple[int, int, int], dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Node registry
    # ------------------------------------------------------------------

    def register_node(self, payload: dict[str, Any]) -> dict[str, Any]:
        node_name = str(payload.get("node_name") or "").strip()
        if not node_name:
            raise ValueError("node_name is required")

        now = _iso()
        with FileLock(self.lock_file, timeout=_LOCK_TIMEOUT):
            nodes = self._read_nodes_unlocked()
            existing = nodes.get(node_name, {})
            node = {
                **existing,
                **payload,
                "node_name": node_name,
                "status": "online",
                "registered_at": existing.get("registered_at") or now,
                "last_seen_at": now,
            }
            nodes[node_name] = node
            self._write_nodes_unlocked(nodes)
            return node

    def heartbeat_node(self, node_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        node_name = (node_name or "").strip()
        if not node_name:
            raise ValueError("node_name is required")

        with FileLock(self.lock_file, timeout=_LOCK_TIMEOUT):
            nodes = self._read_nodes_unlocked()
            node = nodes.get(node_name, {"node_name": node_name, "registered_at": _iso()})
            node.update(payload)
            node["node_name"] = node_name
            node["status"] = "online"
            node["last_seen_at"] = _iso()
            nodes[node_name] = node
            self._write_nodes_unlocked(nodes)
            return node

    def reconcile_node_leases(self, node_name: str, task_ids: list[Any]) -> dict[str, list[str]]:
        """Heartbeat-side lease validation (optional; used when a slave's
        heartbeat lists the task ids it is running as ``active_task_ids``).

        Tasks still leased to *node_name* get their lease extended (a cheap
        extra renewal); tasks the node no longer holds — re-leased to another
        node, cancelled, or unknown — are reported as ``revoked`` so the slave
        can abort the duplicate run."""
        node_name = (node_name or "").strip()
        renewed: list[str] = []
        revoked: list[str] = []
        for raw in task_ids or []:
            task_id = str(raw or "").strip()
            if not task_id:
                continue
            try:
                task = self.renew_lease(node_name, task_id)
            except (LeaseConflictError, FileNotFoundError):
                revoked.append(task_id)
                continue
            except OSError:
                # Busy lock (FileLockTimeout) or transient I/O error — neither
                # renewed nor revoked; retry next beat.
                continue
            if task.get("status") == "leased":
                renewed.append(task_id)
            else:
                # Already terminal (e.g. cancelled) while the node still
                # believes it is running it.
                revoked.append(task_id)
        return {"renewed": renewed, "revoked": revoked}

    def list_nodes(self) -> list[dict[str, Any]]:
        with FileLock(self.lock_file, timeout=_LOCK_TIMEOUT):
            nodes = list(self._read_nodes_unlocked().values())
        return [self._with_health(node) for node in nodes]

    def status(self) -> dict[str, Any]:
        nodes = self.list_nodes()
        tasks = self.list_tasks(include_completed=False)
        return {
            "node_name": config.NODE_NAME,
            "role": config.CLUSTER_ROLE,
            "nodes": nodes,
            "task_counts": self._task_counts(tasks),
            "stale_after_seconds": config.CLUSTER_NODE_STALE_SECONDS,
        }

    # ------------------------------------------------------------------
    # Task queue
    # ------------------------------------------------------------------

    def create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")

        task_id = str(payload.get("task_id") or uuid.uuid4())
        if not _valid_task_id(task_id):
            raise ValueError("task_id must be 1-128 chars of [A-Za-z0-9._-] and start with a letter or digit")
        task = {
            "task_id": task_id,
            "status": "queued",
            "prompt": prompt,
            "source": payload.get("source") or "api",
            "created_by": payload.get("created_by") or "unknown",
            "target_node": payload.get("target_node") or None,
            "required_capabilities": list(payload.get("required_capabilities") or []),
            "required_tags": list(payload.get("required_tags") or []),
            "metadata": dict(payload.get("metadata") or {}),
            "created_at": _iso(),
            "leased_by": None,
            "leased_at": None,
            "lease_expires_at": None,
            "lease_epoch": 0,
            "completed_at": None,
            "result": None,
            "error": None,
        }
        with FileLock(self._task_lock(task_id), timeout=_LOCK_TIMEOUT):
            existing = self._read_task_unlocked(task_id)
            if existing:
                # Idempotent create: a retried POST with an explicit task_id
                # must never clobber a task that may already be running.
                return existing
            self._write_task_unlocked(task)
            self._mark_queued(task)
        return task

    def lease_task(self, node_name: str, capabilities: list[str], tags: list[str]) -> dict[str, Any] | None:
        node_name = (node_name or "").strip()
        if not node_name:
            raise ValueError("node_name is required")
        caps = set(capabilities or [])
        node_tags = set(tags or [])

        self._maybe_reconcile()
        self._maybe_requeue_expired()

        for marker in sorted(self._safe_listdir(self.queued_dir)):
            _, sep, task_id = marker.partition("__")
            marker_path = self.queued_dir / marker
            if not sep or not _valid_task_id(task_id):
                _unlink(marker_path)
                continue

            # Lock-free pre-filter: routing attributes never change after
            # create, so a non-matching task can be skipped without locking.
            # (Non-queued snapshots fall through to the locked path, which
            # drops the stale marker.)
            snapshot = self._read_task_cached(task_id)
            if snapshot is None and not self._task_file(task_id).exists():
                _unlink(marker_path)  # task deleted (retention cleanup)
                continue
            if (
                snapshot is not None
                and snapshot.get("status") == "queued"
                and not self._matches_node(snapshot, node_name, caps, node_tags)
            ):
                continue

            now = _now()
            try:
                with FileLock(self._task_lock(task_id), timeout=_CLAIM_LOCK_TIMEOUT):
                    task = self._read_task_unlocked(task_id)
                    if not task:
                        _unlink(marker_path)
                        continue
                    if task.get("status") != "queued":
                        # Stale hint (already leased/finished) — drop it; never
                        # rewrite a task we don't change.
                        _unlink(marker_path)
                        if task.get("status") == "leased":
                            _touch(self.leased_dir / task_id)
                        continue
                    if not self._matches_node(task, node_name, caps, node_tags):
                        continue

                    expires = now + timedelta(seconds=config.CLUSTER_TASK_LEASE_SECONDS)
                    task["status"] = "leased"
                    task["leased_by"] = node_name
                    task["leased_at"] = _iso(now)
                    task["lease_expires_at"] = _iso(expires)
                    task["lease_epoch"] = int(task.get("lease_epoch") or 0) + 1
                    self._write_task_unlocked(task)
                    _touch(self.leased_dir / task_id)
                    _unlink(marker_path)
                    self._unmark_queued(task)
            except FileLockTimeout:
                continue  # another node is claiming it right now
            except OSError as exc:
                # Transient read/write failure on this record — skip it this
                # pass (marker kept) rather than failing the whole lease call.
                _log(f"WARNING: lease skipped task {task_id}: {exc}")
                continue

            self._safe_append_event(task_id, {
                "type": "leased",
                "node_name": node_name,
                "message": f"Task leased by {node_name}",
                "data": {"lease_epoch": task["lease_epoch"]},
            })
            return task
        return None

    def renew_lease(self, node_name: str, task_id: str, lease_epoch: Any = None) -> dict[str, Any]:
        """Extend a running task's lease so a long execution isn't re-queued
        (and duplicated on another node) while it is still in progress.

        Only the current lease-holder may renew. Outcomes:
        - leased to *node_name* → lease extended, task returned.
        - terminal and last held by *node_name* → returned unchanged (late
          renew racing completion is harmless).
        - lease expired and re-queued but not yet re-leased, and the caller
          was the last holder → the caller reclaims it (no duplicate run).
        - held by another node, or ``lease_epoch`` given and different
          → LeaseConflictError (HTTP 409): the caller must stop.
        """
        node_name = (node_name or "").strip()
        if not _valid_task_id(task_id):
            raise FileNotFoundError(task_id)
        reclaimed = False
        with FileLock(self._task_lock(task_id), timeout=_LOCK_TIMEOUT):
            task = self._read_task_unlocked(task_id)
            if not task:
                raise FileNotFoundError(task_id)
            if not node_name:
                return task  # anonymous renew is a no-op (as before)
            status = task.get("status")
            self._check_epoch(task, lease_epoch)

            if status == "leased":
                holder = task.get("leased_by")
                if holder and holder != node_name:
                    raise LeaseConflictError(f"task {task_id} is leased by {holder}, not {node_name}")
                now = _now()
                task["leased_at"] = task.get("leased_at") or _iso(now)
                task["lease_expires_at"] = _iso(now + timedelta(seconds=config.CLUSTER_TASK_LEASE_SECONDS))
                self._write_task_unlocked(task)
                _touch(self.leased_dir / task_id)
                return task

            if status in _TERMINAL:
                if self._held_by_other(task, node_name):
                    holder = task.get("completed_by") or task.get("leased_by")
                    raise LeaseConflictError(f"task {task_id} was {status} by {holder}")
                return task

            # queued: the lease expired and the task was re-queued.
            if task.get("last_leased_by") == node_name:
                now = _now()
                task["status"] = "leased"
                task["leased_by"] = node_name
                task["leased_at"] = _iso(now)
                task["lease_expires_at"] = _iso(now + timedelta(seconds=config.CLUSTER_TASK_LEASE_SECONDS))
                self._write_task_unlocked(task)
                _touch(self.leased_dir / task_id)
                self._unmark_queued(task)
                reclaimed = True
            else:
                raise LeaseConflictError(f"task {task_id} is no longer leased by {node_name}")

        if reclaimed:
            self._safe_append_event(task_id, {
                "type": "lease_reclaimed",
                "node_name": node_name,
                "message": f"Expired lease reclaimed by {node_name}",
            })
        return task

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        if not _valid_task_id(task_id):
            return None
        task = self._read_task_cached(task_id)
        if task is None and self._task_file(task_id).exists():
            # Unparseable — re-read under the lock (quarantines if corrupt).
            with FileLock(self._task_lock(task_id), timeout=_LOCK_TIMEOUT):
                task = self._read_task_unlocked(task_id)
        return task

    def list_tasks(self, include_completed: bool = True) -> list[dict[str, Any]]:
        if include_completed:
            task_ids = [p.stem for p in self.tasks_dir.glob("*.json")]
        else:
            # Active tasks only: served from the index instead of every record.
            self._maybe_reconcile()
            ids = set(self._safe_listdir(self.leased_dir))
            for marker in self._safe_listdir(self.queued_dir):
                _, sep, tid = marker.partition("__")
                if sep:
                    ids.add(tid)
            task_ids = [tid for tid in ids if _valid_task_id(tid)]

        tasks: list[dict[str, Any]] = []
        for task_id in task_ids:
            task = self._read_task_cached(task_id)
            if not task:
                continue
            if include_completed or task.get("status") not in _TERMINAL:
                tasks.append(task)
        self._prune_cache(set(task_ids) if include_completed else None)
        # Newest first by creation time (mtime changes on every write and
        # would reorder the list on each lease/renew).
        tasks.sort(key=lambda t: (_sort_key(t.get("created_at")), t.get("task_id") or ""), reverse=True)
        return [self._strip_task(t) for t in tasks]

    def append_event(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not _valid_task_id(task_id):
            raise FileNotFoundError(task_id)
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            data = {"value": data}
        event = {
            "at": _iso(),
            "type": payload.get("type") or "event",
            "node_name": payload.get("node_name"),
            "message": payload.get("message") or "",
            "data": data,
        }
        events_path = self._task_events_file(task_id)
        with FileLock(self._task_lock(task_id), timeout=_LOCK_TIMEOUT):
            node = str(event.get("node_name") or "").strip()
            if node:
                task = self._read_task_unlocked(task_id)
                if task and self._held_by_other(task, node):
                    # Informational only — keep it, but flag it so a stale
                    # worker's chatter isn't mistaken for the holder's.
                    event["data"] = {**data, "stale_worker": True}
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False))
                f.write("\n")
        return event

    def complete_task(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        status = payload.get("status") or ("failed" if payload.get("error") else "completed")
        if not isinstance(status, str) or status not in _TERMINAL:
            raise ValueError("status must be completed, failed, or cancelled")
        if not _valid_task_id(task_id):
            raise FileNotFoundError(task_id)
        node_name = str(payload.get("node_name") or "").strip()
        with FileLock(self._task_lock(task_id), timeout=_LOCK_TIMEOUT):
            task = self._read_task_unlocked(task_id)
            if not task:
                raise FileNotFoundError(task_id)
            current = task.get("status")

            if current in _TERMINAL:
                holder = task.get("completed_by") or task.get("leased_by")
                if current == status and (not node_name or not holder or holder == node_name):
                    # Idempotent retry of a completion that already landed.
                    return task
                raise LeaseConflictError(f"task {task_id} is already {current}")

            self._check_epoch(task, payload.get("lease_epoch"))
            if node_name:
                if current == "leased":
                    holder = task.get("leased_by")
                    if holder and holder != node_name:
                        raise LeaseConflictError(f"task {task_id} is leased by {holder}, not {node_name}")
                else:
                    # queued after an expired lease: only the former holder
                    # may still report (its work is valid; nobody re-ran it).
                    if task.get("last_leased_by") != node_name:
                        raise LeaseConflictError(f"task {task_id} is not leased by {node_name}")
                    task["leased_by"] = node_name
            # No node_name → master/admin action (e.g. cancel); always allowed.

            task["status"] = status
            task["completed_at"] = _iso()
            task["completed_by"] = node_name or None
            task["result"] = payload.get("result")
            task["error"] = payload.get("error")
            self._write_task_unlocked(task)
            if current == "queued":
                self._unmark_queued(task)
            _unlink(self.leased_dir / task_id)
        self._safe_append_event(task_id, {
            "type": status,
            "node_name": node_name or None,
            "message": payload.get("error") or "Task completed",
        })
        return task

    def save_artifact(self, task_id: str, node_name: str, filename: str, content: bytes) -> dict[str, Any]:
        """Store a file delivered by a slave for *task_id* and record it on the task."""
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
        if not _valid_task_id(task_id):
            raise FileNotFoundError(task_id)
        node_name = (node_name or "").strip()
        # Basename only — no path traversal from uploaded names.
        safe = Path(str(filename).replace("\\", "/")).name.strip()
        if not safe or safe in {".", ".."}:
            safe = f"artifact-{uuid.uuid4().hex[:8]}"

        with FileLock(self._task_lock(task_id), timeout=_LOCK_TIMEOUT):
            task = self._read_task_unlocked(task_id)
            if not task:
                raise FileNotFoundError(task_id)
            if node_name and self._held_by_other(task, node_name):
                raise LeaseConflictError(f"task {task_id} is not held by {node_name}")
            artifacts = list(task.get("artifacts") or [])
            if len(artifacts) >= MAX_ARTIFACTS_PER_TASK:
                raise ValueError(f"artifact limit reached ({MAX_ARTIFACTS_PER_TASK} per task)")

            task_dir = self.artifacts_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            dest = task_dir / safe
            counter = 1
            while dest.exists():
                dest = task_dir / f"{counter}-{safe}"
                counter += 1
            _atomic_write_bytes(dest, content)

            artifact = {
                "name": dest.name,
                "path": str(dest.resolve()),
                "size": len(content),
                "uploaded_by": node_name or None,
                "uploaded_at": _iso(),
            }
            task["artifacts"] = artifacts + [artifact]
            self._write_task_unlocked(task)
        self._safe_append_event(task_id, {
            "type": "artifact",
            "node_name": node_name,
            "message": f"Received artifact {dest.name} ({len(content)} bytes)",
        })
        return artifact

    def load_events(self, task_id: str) -> list[dict[str, Any]]:
        if not _valid_task_id(task_id):
            return []
        path = self._task_events_file(task_id)
        if not path.exists():
            return []
        events = []
        skipped = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    skipped += 1  # torn append (crash mid-write)
                    continue
                if isinstance(event, dict):
                    events.append(event)
        if skipped:
            _log(f"WARNING: skipped {skipped} unparseable event line(s) for task {task_id}")
        return events

    # ------------------------------------------------------------------
    # Lease maintenance
    # ------------------------------------------------------------------

    def _maybe_requeue_expired(self, force: bool = False) -> None:
        """Re-queue tasks whose lease expired. Only the leased index is
        scanned (bounded by in-flight work), and only on an interval."""
        with self._state_lock:
            mono = time.monotonic()
            if not force and mono - self._last_expiry_scan < _EXPIRY_SCAN_INTERVAL:
                return
            self._last_expiry_scan = mono

        for task_id in self._safe_listdir(self.leased_dir):
            marker = self.leased_dir / task_id
            if not _valid_task_id(task_id):
                _unlink(marker)
                continue
            snapshot = self._read_task_cached(task_id)
            if snapshot is None and not self._task_file(task_id).exists():
                _unlink(marker)  # task deleted (retention cleanup)
                continue
            if snapshot is not None and snapshot.get("status") == "leased":
                expires = _parse_iso(snapshot.get("lease_expires_at"))
                if expires and expires >= _now():
                    continue  # still valid — no lock needed
            previous = None
            try:
                with FileLock(self._task_lock(task_id), timeout=_CLAIM_LOCK_TIMEOUT):
                    task = self._read_task_unlocked(task_id)
                    if not task or task.get("status") != "leased":
                        _unlink(marker)
                        if task and task.get("status") == "queued":
                            self._mark_queued(task)
                        continue
                    expires = _parse_iso(task.get("lease_expires_at"))
                    if expires and expires >= _now():
                        continue
                    previous = task.get("leased_by")
                    task["status"] = "queued"
                    task["last_leased_by"] = previous
                    task["leased_by"] = None
                    task["leased_at"] = None
                    task["lease_expires_at"] = None
                    self._write_task_unlocked(task)
                    self._mark_queued(task)
                    _unlink(marker)
            except OSError as exc:  # includes FileLockTimeout (busy → next scan)
                if not isinstance(exc, FileLockTimeout):
                    _log(f"WARNING: expiry scan skipped task {task_id}: {exc}")
                continue
            _log(f"Lease on task {task_id} held by {previous} expired; re-queued")
            self._safe_append_event(task_id, {
                "type": "requeued",
                "node_name": previous,
                "message": f"Lease held by {previous} expired; task re-queued",
            })

    def _maybe_reconcile(self, force: bool = False) -> None:
        """Add index markers for any queued/leased task the index lacks
        (legacy data from before the index existed, or a crash between the
        record write and the marker write). Markers are only ever *added*
        here; stale markers are dropped by the lock-holding paths."""
        with self._state_lock:
            mono = time.monotonic()
            if not force and self._last_reconcile and mono - self._last_reconcile < _RECONCILE_INTERVAL:
                return
            self._last_reconcile = mono

        queued_ids = set()
        for marker in self._safe_listdir(self.queued_dir):
            _, sep, tid = marker.partition("__")
            if sep:
                queued_ids.add(tid)
        leased_ids = set(self._safe_listdir(self.leased_dir))
        added = 0
        seen: set[str] = set()
        for path in self.tasks_dir.glob("*.json"):
            task_id = path.stem
            seen.add(task_id)
            if not _valid_task_id(task_id):
                continue
            task = self._read_task_cached(task_id)
            if task is None:
                if path.exists():
                    # Unparseable: re-read under lock so it gets quarantined.
                    try:
                        with FileLock(self._task_lock(task_id), timeout=_CLAIM_LOCK_TIMEOUT):
                            self._read_task_unlocked(task_id)
                    except OSError:  # busy lock or transient read error
                        pass
                continue
            status = task.get("status")
            if status == "queued" and task_id not in queued_ids:
                self._mark_queued(task)
                added += 1
            elif status == "leased" and task_id not in leased_ids:
                _touch(self.leased_dir / task_id)
                added += 1
        self._prune_cache(seen)
        if added:
            _log(f"Index reconcile: added {added} missing queue/lease marker(s)")

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _read_nodes_unlocked(self) -> dict[str, Any]:
        if not self.nodes_file.exists():
            return {}
        try:
            data = json.loads(self.nodes_file.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._quarantine(self.nodes_file, exc)
            return {}
        except OSError as exc:
            _log(f"ERROR: could not read {self.nodes_file}: {exc}")
            return {}
        if not isinstance(data, dict):
            self._quarantine(self.nodes_file, TypeError("nodes.json is not an object"))
            return {}
        return data

    def _write_nodes_unlocked(self, nodes: dict[str, Any]) -> None:
        _atomic_write_json(self.nodes_file, nodes)

    def _task_file(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def _task_lock(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.lock"

    def _task_events_file(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.events.jsonl"

    def _read_task_unlocked(self, task_id: str) -> dict[str, Any] | None:
        """Authoritative read; caller holds the task lock. A corrupt record is
        quarantined (and logged) instead of silently vanishing.

        A transient I/O error (EMFILE, EIO, Windows sharing violation) is
        re-raised rather than reported as "missing": treating it as None would
        turn a slave's completion into a permanent 404 and drop the task's
        queue marker. Routes map it to 503 (retryable)."""
        path = self._task_file(task_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            _log(f"ERROR: could not read task {task_id}: {exc}")
            raise
        try:
            task = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._quarantine(path, exc)
            return None
        if not isinstance(task, dict) or not task.get("task_id"):
            self._quarantine(path, TypeError("task record is not an object with task_id"))
            return None
        return task

    def _read_task_cached(self, task_id: str) -> dict[str, Any] | None:
        """Lock-free read (safe: writes are atomic replaces). Returns a copy,
        or None if missing/unparseable — never quarantines."""
        path = self._task_file(task_id)
        try:
            st = path.stat()
        except OSError:
            self._read_cache.pop(task_id, None)
            return None
        key = (int(getattr(st, "st_ino", 0) or 0), int(st.st_mtime_ns), int(st.st_size))
        cached = self._read_cache.get(task_id)
        if cached and cached[0] == key:
            return copy.deepcopy(cached[1])
        try:
            task = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        if not isinstance(task, dict) or not task.get("task_id"):
            return None
        if key[2] <= _CACHE_MAX_RECORD_BYTES:
            self._read_cache[task_id] = (key, task)
            return copy.deepcopy(task)
        # Huge records (long delegated results) are re-read on demand so the
        # cache can't pin hundreds of MB on the master.
        self._read_cache.pop(task_id, None)
        return task

    def _prune_cache(self, live_ids: set[str] | None) -> None:
        if live_ids is None:
            return
        for task_id in list(self._read_cache.keys()):
            if task_id not in live_ids:
                self._read_cache.pop(task_id, None)

    def _write_task_unlocked(self, task: dict[str, Any]) -> None:
        _atomic_write_json(self._task_file(task["task_id"]), task)

    def _mark_queued(self, task: dict[str, Any]) -> None:
        task_id = task["task_id"]
        marker = self.queued_dir / f"{_sort_key(task.get('created_at'))}__{task_id}"
        if not marker.exists():
            _touch(marker)

    def _unmark_queued(self, task: dict[str, Any]) -> None:
        task_id = task["task_id"]
        if _parse_iso(task.get("created_at")):
            _unlink(self.queued_dir / f"{_sort_key(task.get('created_at'))}__{task_id}")
            return
        # Legacy record without created_at: its marker name isn't derivable.
        for name in self._safe_listdir(self.queued_dir):
            if name.partition("__")[2] == task_id:
                _unlink(self.queued_dir / name)

    @staticmethod
    def _safe_listdir(path: Path) -> list[str]:
        try:
            return [n for n in os.listdir(path) if not n.startswith(".")]
        except OSError:
            return []

    @staticmethod
    def _quarantine(path: Path, exc: BaseException) -> None:
        stamp = _now().strftime("%Y%m%dT%H%M%S")
        dest = path.with_name(f"{path.name}.corrupt-{stamp}-{uuid.uuid4().hex[:6]}")
        try:
            _replace_with_retry(str(path), str(dest))
            _log(f"ERROR: unparseable {path.name} ({exc}); quarantined as {dest.name}")
        except OSError as move_exc:
            _log(f"ERROR: unparseable {path.name} ({exc}); quarantine failed: {move_exc}")

    def _safe_append_event(self, task_id: str, payload: dict[str, Any]) -> None:
        """Bookkeeping events must never fail the state change that already
        committed (e.g. a lease the slave would otherwise never learn about)."""
        try:
            self.append_event(task_id, payload)
        except Exception as exc:
            _log(f"WARNING: could not append '{payload.get('type')}' event for {task_id}: {exc}")

    # ------------------------------------------------------------------
    # Matching and summaries
    # ------------------------------------------------------------------

    @staticmethod
    def _check_epoch(task: dict[str, Any], lease_epoch: Any) -> None:
        """Optional fencing token: a caller that passes the lease_epoch it was
        handed at lease time is rejected once the task was re-leased."""
        if lease_epoch is None or lease_epoch == "":
            return
        try:
            supplied = int(lease_epoch)
        except (TypeError, ValueError):
            return
        current = int(task.get("lease_epoch") or 0)
        if supplied != current:
            raise LeaseConflictError(
                f"stale lease for task {task.get('task_id')} (epoch {supplied}, current {current})"
            )

    @staticmethod
    def _held_by_other(task: dict[str, Any], node_name: str) -> bool:
        if not node_name:
            return False
        status = task.get("status")
        if status == "leased":
            holder = task.get("leased_by")
        elif status in _TERMINAL:
            holder = task.get("completed_by") or task.get("leased_by")
        else:
            holder = task.get("last_leased_by")
        return bool(holder) and holder != node_name

    def _with_health(self, node: dict[str, Any]) -> dict[str, Any]:
        last_seen = _parse_iso(node.get("last_seen_at"))
        stale = True
        age = None
        if last_seen:
            age = max(0, int((_now() - last_seen).total_seconds()))
            stale = age > config.CLUSTER_NODE_STALE_SECONDS
        return {
            **node,
            "healthy": not stale,
            "seconds_since_seen": age,
            "status": "stale" if stale else node.get("status", "online"),
        }

    @staticmethod
    def _matches_node(task: dict[str, Any], node_name: str, caps: set[str], tags: set[str]) -> bool:
        target = task.get("target_node")
        if target and target != node_name:
            return False
        required_caps = set(task.get("required_capabilities") or [])
        required_tags = set(task.get("required_tags") or [])
        return required_caps.issubset(caps) and required_tags.issubset(tags)

    @staticmethod
    def _strip_task(task: dict[str, Any]) -> dict[str, Any]:
        summary = dict(task)
        prompt = summary.get("prompt") or ""
        if len(prompt) > 300:
            summary["prompt"] = prompt[:300] + "..."
        result = summary.get("result")
        if isinstance(result, str) and len(result) > 300:
            summary["result"] = result[:300] + "..."
        return summary

    @staticmethod
    def _task_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in tasks:
            status = task.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts


cluster_store = ClusterStore()
