"""
Simple SQLite database for users and sessions.
Append-only JSONL storage for conversation history.
"""
import asyncio
import os
import sqlite3
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from filelock import FileLock, Timeout

import config


class Database:
    """Simple SQLite database wrapper"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        # Generous busy timeout: several workers/threads may write session
        # counters at once; waiting beats failing a turn with "database is locked".
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Sessions table (lightweight metadata only)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    FOREIGN KEY (username) REFERENCES users(username)
                )
            """)

            # Add title column if it doesn't exist (migration for existing DBs)
            cols = [row[1] for row in cursor.execute("PRAGMA table_info(sessions)").fetchall()]
            if "title" not in cols:
                cursor.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
            if "workspace_dir" not in cols:
                cursor.execute("ALTER TABLE sessions ADD COLUMN workspace_dir TEXT")

            # Create default admin user
            self._create_default_admin()

    def _create_default_admin(self):
        """Create default admin user if not exists"""
        # Import here to avoid circular dependency
        from passlib.context import CryptContext

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Validate password length before hashing (bcrypt limit: 72 bytes)
                password_bytes = config.DEFAULT_ADMIN_PASSWORD.encode('utf-8')
                password_to_hash = config.DEFAULT_ADMIN_PASSWORD
                
                if len(password_bytes) > 72:
                    print(f"Warning: DEFAULT_ADMIN_PASSWORD exceeds 72 bytes ({len(password_bytes)} bytes).")
                    print("Please update config.DEFAULT_ADMIN_PASSWORD to be 72 bytes or less.")
                    # Truncate to 72 bytes as a fallback
                    password_to_hash = config.DEFAULT_ADMIN_PASSWORD.encode('utf-8')[:72].decode('utf-8', errors='ignore')
                
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                password_hash = pwd_context.hash(password_to_hash)
                
                cursor.execute(
                    "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (
                        config.DEFAULT_ADMIN_USERNAME,
                        password_hash,
                        "admin"
                    )
                )
        except Exception as e:
            print(f"Warning: Could not create default admin: {e}")

    # ========================================================================
    # User operations
    # ========================================================================
    def create_user(self, username: str, password_hash: str, role: str = "user") -> bool:
        """Create a new user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, password_hash, role)
                )
                return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========================================================================
    # Session operations
    # ========================================================================
    def create_session(self, session_id: str, username: str) -> bool:
        """Create a new session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sessions (id, username) VALUES (?, ?)",
                    (session_id, username)
                )
                return True
        except sqlite3.IntegrityError:
            return False

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def increment_session_message_count(self, session_id: str, delta: int):
        """Increment message count for a session."""
        if delta == 0:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET message_count = COALESCE(message_count, 0) + ? WHERE id = ?",
                (delta, session_id)
            )

    def list_user_sessions(self, username: str) -> List[Dict[str, Any]]:
        """List all sessions for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE username = ? ORDER BY created_at DESC",
                (username,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_session_title(self, session_id: str, title: str):
        """Set or update the title for a session."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id)
            )

    def update_session_workspace(self, session_id: str, workspace_dir: Optional[str]):
        """Set or clear the workspace directory for a session.

        Stored verbatim (absolute path); resolution/validation lives in the
        chat route, not the DB layer.
        """
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET workspace_dir = ? WHERE id = ?",
                (workspace_dir, session_id)
            )

    def search_sessions(self, username: str, query: str) -> List[Dict[str, Any]]:
        """Search sessions by title or session ID for a user."""
        pattern = f"%{query}%"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE username = ? AND (title LIKE ? OR id LIKE ?) ORDER BY created_at DESC",
                (username, pattern, pattern)
            )
            return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# Durable file helpers (atomic replace, Windows-tolerant)
# ============================================================================

def _fsync_quietly(f) -> None:
    """Flush Python + OS buffers for an open file; never raises."""
    try:
        f.flush()
        os.fsync(f.fileno())
    except (OSError, ValueError):
        pass


def _replace_with_retry(src: Path, dst: Path, attempts: int = 5, delay: float = 0.05) -> None:
    """os.replace with a short retry loop.

    On Windows os.replace raises PermissionError while another process (an
    antivirus scanner, an editor, a concurrent reader without
    FILE_SHARE_DELETE) has the destination open. Retry briefly before giving up.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write *text* to *path* atomically (tmp file + fsync + os.replace).

    Readers see either the old or the new content, never a torn file — even
    when the process dies mid-write.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            _fsync_quietly(f)
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# ============================================================================
# Per-session turn lease
# ============================================================================

class SessionBusyError(Exception):
    """Another turn (or a manual compaction) currently holds the session lease."""


class TurnLease:
    """Cross-process exclusive lease on one session's turn.

    Backed by a FileLock on data/sessions/{id}.turn.lock, so it serializes
    turns across uvicorn workers as well as within one process (each lease
    opens its own file descriptor, and flock/msvcrt locks conflict per fd).
    `release()` is idempotent and safe to call from any thread (the lock is
    created with thread_local=False). Dropping the object also releases the
    OS lock (FileLock.__del__), so a lease can never outlive its owner.
    """

    def __init__(self, session_id: str, lock: FileLock):
        self.session_id = session_id
        self._lock = lock
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._lock.release(force=True)
        except Exception as exc:
            print(f"[SESSION] Failed to release turn lease for {self.session_id}: {exc}")


def _make_file_lock(path: Path) -> FileLock:
    try:
        return FileLock(str(path), thread_local=False)
    except TypeError:  # very old filelock without the thread_local kwarg
        return FileLock(str(path))


class ConversationStore:
    """
    Store conversations as append-only JSONL plus a small recent-message cache.

    Files:
      - data/sessions/{session_id}.jsonl
      - data/sessions/{session_id}.recent.json
      - data/sessions/{session_id}.lock        (short-lived I/O lock)
      - data/sessions/{session_id}.turn.lock   (per-turn lease, see TurnLease)

    Durability rules:
      - Full rewrites and the recent cache are written atomically
        (tmp + fsync + os.replace).
      - Appends are fsynced; a torn trailing line from a crash is fenced off
        with a newline so it never merges with the next record.
      - Reads skip torn/invalid JSONL lines with a warning instead of failing
        the whole session.
      - `.recent.json` records the budget it was trimmed with and the JSONL
        size it reflects; it is rebuilt from the log when invalid, when the
        log changed behind its back, or when the configured budget changed.

    Legacy sessions stored as data/sessions/{session_id}.json are migrated on first access.
    """

    # Bumped when the .recent.json payload shape changes (forces a rebuild).
    _RECENT_FORMAT = 2

    def __init__(self, sessions_dir: str | Path | None = None):
        self.sessions_dir = Path(sessions_dir or config.SESSIONS_DIR)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_log_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.jsonl"

    def _get_recent_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.recent.json"

    def _get_legacy_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _get_lock_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.lock"

    def _get_turn_lock_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.turn.lock"

    def _recent_window(self) -> int:
        return max(1, int(getattr(config, "MAX_CONVERSATION_HISTORY", 400)))

    def _recent_max_tokens(self) -> int:
        return int(getattr(config, "MAX_CONVERSATION_TOKENS", 300_000))

    def _recent_budget(self) -> Dict[str, int]:
        return {
            "format": self._RECENT_FORMAT,
            "max_messages": self._recent_window(),
            "max_tokens": self._recent_max_tokens(),
        }

    def _trim_recent(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Bound the recent-history cache by both message count and token budget."""
        from backend.utils.tokens import trim_to_token_budget
        return trim_to_token_budget(
            messages, self._recent_max_tokens(), max_messages=self._recent_window()
        )

    def _session_exists(self, session_id: str) -> bool:
        return (
            self._get_session_log_file(session_id).exists()
            or self._get_recent_file(session_id).exists()
            or self._get_legacy_file(session_id).exists()
        )

    def _log_size(self, session_id: str) -> int:
        try:
            return self._get_session_log_file(session_id).stat().st_size
        except OSError:
            return 0

    # ------------------------------------------------------------------
    # Turn lease
    # ------------------------------------------------------------------

    async def acquire_turn_lease(self, session_id: str, timeout: float = 2.0) -> TurnLease:
        """Acquire the per-session turn lease, waiting up to *timeout* seconds.

        Polls a non-blocking lock attempt so the event loop is never blocked.
        Raises SessionBusyError when another turn still holds it.
        """
        lock = _make_file_lock(self._get_turn_lock_file(session_id))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout))
        while True:
            try:
                lock.acquire(timeout=0)
                return TurnLease(session_id, lock)
            except Timeout:
                pass
            if loop.time() >= deadline:
                raise SessionBusyError(f"session {session_id} is busy")
            await asyncio.sleep(0.1)

    def is_turn_active(self, session_id: str) -> bool:
        """Best-effort probe: True if some turn currently holds the session lease."""
        lock = _make_file_lock(self._get_turn_lock_file(session_id))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return True
        except Exception:
            return False
        try:
            lock.release(force=True)
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_conversation(self, session_id: str, messages: List[Dict[str, Any]]):
        """Rewrite the full conversation history (atomically)."""
        with FileLock(self._get_lock_file(session_id), timeout=10):
            self._write_full_conversation_unlocked(session_id, messages)

    def append_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """Append new messages and refresh the bounded hot-history cache."""
        if not messages:
            return

        with FileLock(self._get_lock_file(session_id), timeout=10):
            self._ensure_migrated_unlocked(session_id)

            # Read the cache BEFORE appending: its validity check compares the
            # JSONL size it was built against with the current size.
            recent_messages = self._load_recent_unlocked(session_id)

            self._append_log_unlocked(session_id, messages)

            if recent_messages is None:
                # Missing/invalid/stale cache — rebuild from the full log,
                # which already contains the messages just appended.
                recent_messages = self._read_log_unlocked(session_id)
            else:
                recent_messages.extend(messages)
            self._write_recent_unlocked(session_id, self._trim_recent(recent_messages))

    def load_conversation(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load the full conversation history.

        Returns None when the session has no stored history. Raises
        filelock.Timeout when the session's I/O lock is held too long, so
        callers can report "busy" instead of silently dropping history.
        """
        if not self._session_exists(session_id):
            return None

        try:
            with FileLock(self._get_lock_file(session_id), timeout=10):
                self._ensure_migrated_unlocked(session_id)
                return self._read_log_unlocked(session_id)
        except Timeout:
            raise
        except Exception as exc:
            print(f"[SESSION] WARNING: failed to load conversation {session_id}: {exc}")
            return None

    def load_recent_conversation(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Load the recent conversation cache used on the hot request path.

        Raises filelock.Timeout when the session's I/O lock is held too long.
        """
        if not self._session_exists(session_id):
            return None

        try:
            with FileLock(self._get_lock_file(session_id), timeout=10):
                self._ensure_migrated_unlocked(session_id)
                recent_messages = self._load_recent_unlocked(session_id)
                if recent_messages is None:
                    recent_messages = self._trim_recent(self._read_log_unlocked(session_id))
                    try:
                        self._write_recent_unlocked(session_id, recent_messages)
                    except Exception as exc:
                        print(f"[SESSION] WARNING: could not rewrite recent cache for {session_id}: {exc}")
                if limit is not None and limit >= 0:
                    if limit == 0:
                        return []
                    return recent_messages[-limit:]
                return recent_messages
        except Timeout:
            raise
        except Exception as exc:
            print(f"[SESSION] WARNING: failed to load recent conversation {session_id}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internals (caller holds the session I/O lock)
    # ------------------------------------------------------------------

    def _ensure_migrated_unlocked(self, session_id: str):
        log_file = self._get_session_log_file(session_id)
        if log_file.exists():
            return

        legacy_file = self._get_legacy_file(session_id)
        if not legacy_file.exists():
            return

        try:
            with open(legacy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[SESSION] WARNING: unreadable legacy session file {legacy_file.name}: {exc}")
            return

        messages = data.get("messages", []) if isinstance(data, dict) else []
        messages = [m for m in messages if isinstance(m, dict) and m.get("role")]
        self._write_full_conversation_unlocked(session_id, messages)
        if legacy_file.exists():
            try:
                legacy_file.unlink()
            except OSError:
                pass

    @staticmethod
    def _serialize_messages(messages: List[Dict[str, Any]]) -> str:
        return "".join(
            json.dumps(message, ensure_ascii=False, default=str) + "\n"
            for message in messages
        )

    def _append_log_unlocked(self, session_id: str, messages: List[Dict[str, Any]]):
        log_file = self._get_session_log_file(session_id)
        # Fence off a torn trailing line (crash mid-append) so the next record
        # starts on its own line instead of merging into the garbage.
        needs_newline = False
        try:
            if log_file.stat().st_size > 0:
                with open(log_file, 'rb') as rf:
                    rf.seek(-1, os.SEEK_END)
                    needs_newline = rf.read(1) != b"\n"
        except OSError:
            needs_newline = False

        payload = self._serialize_messages(messages)
        if needs_newline:
            payload = "\n" + payload
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(payload)
            _fsync_quietly(f)

    def _write_full_conversation_unlocked(self, session_id: str, messages: List[Dict[str, Any]]):
        log_file = self._get_session_log_file(session_id)
        atomic_write_text(log_file, self._serialize_messages(messages))

        self._write_recent_unlocked(session_id, self._trim_recent(list(messages)))

        legacy_file = self._get_legacy_file(session_id)
        if legacy_file.exists():
            try:
                legacy_file.unlink()
            except OSError:
                pass

    def _read_log_unlocked(self, session_id: str) -> List[Dict[str, Any]]:
        log_file = self._get_session_log_file(session_id)
        if not log_file.exists():
            return []

        messages: List[Dict[str, Any]] = []
        skipped = 0
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    skipped += 1
                    continue
                if not isinstance(obj, dict) or not obj.get("role"):
                    skipped += 1
                    continue
                messages.append(obj)
        if skipped:
            print(
                f"[SESSION] WARNING: skipped {skipped} torn/invalid line(s) "
                f"in {log_file.name}"
            )
        return messages

    def _load_recent_unlocked(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Return the cached recent messages, or None when the cache must be rebuilt.

        None means: missing, unreadable/invalid, built with a different
        budget (e.g. MAX_CONVERSATION_* was raised), or out of sync with the
        JSONL log (a crash between the log append and the cache write).
        """
        recent_file = self._get_recent_file(session_id)
        if not recent_file.exists():
            return None

        try:
            with open(recent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[SESSION] WARNING: invalid recent cache {recent_file.name} ({exc}); rebuilding from log")
            return None

        if not isinstance(data, dict):
            return None
        messages = data.get("messages")
        if not isinstance(messages, list):
            return None
        messages = [m for m in messages if isinstance(m, dict) and m.get("role")]
        if not self._get_session_log_file(session_id).exists():
            # Orphan cache with no JSONL log (pre-JSONL data): it is the only
            # copy of the history, so trust it rather than "rebuild" to empty.
            return self._trim_recent(messages)
        if data.get("budget") != self._recent_budget():
            return None
        log_size = data.get("log_size")
        if not isinstance(log_size, int) or log_size != self._log_size(session_id):
            return None
        return messages

    def _write_recent_unlocked(self, session_id: str, messages: List[Dict[str, Any]]):
        recent_file = self._get_recent_file(session_id)
        payload = {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "budget": self._recent_budget(),
            "log_size": self._log_size(session_id),
            "messages": messages,
        }
        atomic_write_text(
            recent_file,
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        )


# Global instances
db = Database()
conversation_store = ConversationStore()
