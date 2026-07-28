"""
Persistent cluster registry and task queue.

The master LLM API owns this store. Slaves register, heartbeat, lease work,
stream events, and complete tasks through the /api/cluster routes.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

import config


# Mirrored by the slave-side upload limits in hoonbot/core/cluster_worker.py.
MAX_ARTIFACTS_PER_TASK = 10
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class ClusterStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or config.CLUSTER_DIR
        self.tasks_dir = self.base_dir / "tasks"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.nodes_file = self.base_dir / "nodes.json"
        self.lock_file = self.base_dir / "cluster.lock"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Node registry
    # ------------------------------------------------------------------

    def register_node(self, payload: dict[str, Any]) -> dict[str, Any]:
        node_name = str(payload.get("node_name") or "").strip()
        if not node_name:
            raise ValueError("node_name is required")

        now = _iso()
        with FileLock(self.lock_file, timeout=10):
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
        node_name = node_name.strip()
        if not node_name:
            raise ValueError("node_name is required")

        with FileLock(self.lock_file, timeout=10):
            nodes = self._read_nodes_unlocked()
            node = nodes.get(node_name, {"node_name": node_name, "registered_at": _iso()})
            node.update(payload)
            node["node_name"] = node_name
            node["status"] = "online"
            node["last_seen_at"] = _iso()
            nodes[node_name] = node
            self._write_nodes_unlocked(nodes)
            return node

    def list_nodes(self) -> list[dict[str, Any]]:
        with FileLock(self.lock_file, timeout=10):
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
            "completed_at": None,
            "result": None,
            "error": None,
        }
        with FileLock(self._task_lock(task_id), timeout=10):
            self._write_task_unlocked(task)
        return task

    def lease_task(self, node_name: str, capabilities: list[str], tags: list[str]) -> dict[str, Any] | None:
        node_name = node_name.strip()
        if not node_name:
            raise ValueError("node_name is required")
        now = _now()
        expires = now + timedelta(seconds=config.CLUSTER_TASK_LEASE_SECONDS)
        caps = set(capabilities or [])
        node_tags = set(tags or [])

        for path in sorted(self.tasks_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            task_id = path.stem
            with FileLock(self._task_lock(task_id), timeout=10):
                task = self._read_task_unlocked(task_id)
                if not task:
                    continue
                task = self._recover_expired_lease(task, now)
                if task.get("status") != "queued":
                    self._write_task_unlocked(task)
                    continue
                if not self._matches_node(task, node_name, caps, node_tags):
                    self._write_task_unlocked(task)
                    continue

                task["status"] = "leased"
                task["leased_by"] = node_name
                task["leased_at"] = _iso(now)
                task["lease_expires_at"] = _iso(expires)
                self._write_task_unlocked(task)
            self.append_event(task_id, {
                "type": "leased",
                "node_name": node_name,
                "message": f"Task leased by {node_name}",
            })
            return task
        return None

    def renew_lease(self, node_name: str, task_id: str) -> dict[str, Any] | None:
        """Extend a running task's lease so a long execution isn't re-queued
        (and duplicated on another node) while it is still in progress.

        Only the current lease-holder may renew; a task that is terminal or has
        already been re-leased to someone else is returned unchanged. The slave
        calls this on an interval derived from the lease length, so a task can
        run for as long as it needs without the lease expiring under it.
        """
        node_name = (node_name or "").strip()
        now = _now()
        expires = now + timedelta(seconds=config.CLUSTER_TASK_LEASE_SECONDS)
        with FileLock(self._task_lock(task_id), timeout=10):
            task = self._read_task_unlocked(task_id)
            if not task:
                raise FileNotFoundError(task_id)
            if task.get("status") == "leased" and task.get("leased_by") == node_name:
                task["leased_at"] = task.get("leased_at") or _iso(now)
                task["lease_expires_at"] = _iso(expires)
                self._write_task_unlocked(task)
            return task

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        with FileLock(self._task_lock(task_id), timeout=10):
            return self._read_task_unlocked(task_id)

    def list_tasks(self, include_completed: bool = True) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for path in sorted(self.tasks_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            task = self.load_task(path.stem)
            if not task:
                continue
            if include_completed or task.get("status") not in {"completed", "failed", "cancelled"}:
                tasks.append(self._strip_task(task))
        return tasks

    def append_event(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "at": _iso(),
            "type": payload.get("type") or "event",
            "node_name": payload.get("node_name"),
            "message": payload.get("message") or "",
            "data": payload.get("data") or {},
        }
        events_path = self._task_events_file(task_id)
        with FileLock(self._task_lock(task_id), timeout=10):
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False))
                f.write("\n")
        return event

    def complete_task(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        status = payload.get("status") or ("failed" if payload.get("error") else "completed")
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("status must be completed, failed, or cancelled")
        with FileLock(self._task_lock(task_id), timeout=10):
            task = self._read_task_unlocked(task_id)
            if not task:
                raise FileNotFoundError(task_id)
            task["status"] = status
            task["completed_at"] = _iso()
            task["result"] = payload.get("result")
            task["error"] = payload.get("error")
            self._write_task_unlocked(task)
        self.append_event(task_id, {
            "type": status,
            "node_name": payload.get("node_name"),
            "message": payload.get("error") or "Task completed",
        })
        return task

    def save_artifact(self, task_id: str, node_name: str, filename: str, content: bytes) -> dict[str, Any]:
        """Store a file delivered by a slave for *task_id* and record it on the task."""
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
        # Basename only — no path traversal from uploaded names.
        safe = Path(str(filename).replace("\\", "/")).name.strip() or f"artifact-{uuid.uuid4().hex[:8]}"

        with FileLock(self._task_lock(task_id), timeout=10):
            task = self._read_task_unlocked(task_id)
            if not task:
                raise FileNotFoundError(task_id)
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
            dest.write_bytes(content)

            artifact = {
                "name": dest.name,
                "path": str(dest.resolve()),
                "size": len(content),
                "uploaded_by": node_name or None,
                "uploaded_at": _iso(),
            }
            task["artifacts"] = artifacts + [artifact]
            self._write_task_unlocked(task)
        self.append_event(task_id, {
            "type": "artifact",
            "node_name": node_name,
            "message": f"Received artifact {dest.name} ({len(content)} bytes)",
        })
        return artifact

    def load_events(self, task_id: str) -> list[dict[str, Any]]:
        path = self._task_events_file(task_id)
        if not path.exists():
            return []
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _read_nodes_unlocked(self) -> dict[str, Any]:
        if not self.nodes_file.exists():
            return {}
        try:
            return json.loads(self.nodes_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_nodes_unlocked(self, nodes: dict[str, Any]) -> None:
        self.nodes_file.write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")

    def _task_file(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def _task_lock(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.lock"

    def _task_events_file(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.events.jsonl"

    def _read_task_unlocked(self, task_id: str) -> dict[str, Any] | None:
        path = self._task_file(task_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_task_unlocked(self, task: dict[str, Any]) -> None:
        self._task_file(task["task_id"]).write_text(
            json.dumps(task, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Matching and summaries
    # ------------------------------------------------------------------

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
    def _recover_expired_lease(task: dict[str, Any], now: datetime) -> dict[str, Any]:
        if task.get("status") != "leased":
            return task
        expires = _parse_iso(task.get("lease_expires_at"))
        if expires and expires < now:
            task["status"] = "queued"
            task["leased_by"] = None
            task["leased_at"] = None
            task["lease_expires_at"] = None
        return task

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
