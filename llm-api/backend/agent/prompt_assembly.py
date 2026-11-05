"""PromptMixin: system prompt, dynamic context, RAG/memo/file formatting, and tool schema helpers."""
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import config
from backend.agent._cache import (
    _CACHED_SYSTEM_PROMPT,
    _CACHED_TOOL_SCHEMAS,
    _load_memo_cached,
    _rag_collections_cache,
    _RAG_CACHE_TTL,
)

# Module-level caches for environment context (avoids spawning subprocesses every iteration)
_env_git_cache: Dict[str, Any] = {}
_env_files_cache: Dict[str, Any] = {}
_repo_doc_cache: Dict[str, Any] = {}  # resolved path -> {"mtime", "size", "text"}
_ENV_GIT_TTL = 30   # seconds
_ENV_FILES_TTL = 60 # seconds

# Git context: per-command timeout and a total budget across the three calls,
# so a huge / network-mounted repo can't stall request start-up.
_GIT_CMD_TIMEOUT = 3.0
_GIT_TOTAL_BUDGET = 5.0

# Recent-files walk bounds. The old unbounded rglob walked node_modules/data
# trees before filtering them out — seconds of blocking IO on a large workspace.
_WALK_PRUNE = {
    ".git", "__pycache__", "node_modules", "data", ".venv", "venv", "dist",
    "dist-web", "build", ".mypy_cache", ".pytest_cache", ".tox", ".idea",
    ".cache", "models", "offline_models",
}
_WALK_MAX_DEPTH = 4
_WALK_MAX_ENTRIES = 5000
_WALK_TIME_BUDGET = 1.5  # seconds


def _get_git_context(cwd: str) -> str:
    """Return a brief git block for cwd. Cached 30s. Returns '' if not a git repo or git absent."""
    cached = _env_git_cache.get(cwd)
    if cached and time.time() < cached["expires_at"]:
        return cached["text"]

    lines = []
    git_dir = Path(cwd) / ".git"
    try:
        is_repo = git_dir.exists()
    except OSError:
        is_repo = False
    if not is_repo:
        text = ""
        _env_git_cache[cwd] = {"text": text, "expires_at": time.time() + _ENV_GIT_TTL}
        return text

    deadline = time.monotonic() + _GIT_TOTAL_BUDGET
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"      # never take index.lock for a read
    env["GIT_TERMINAL_PROMPT"] = "0"     # never block on credentials

    def _run(args):
        remaining = deadline - time.monotonic()
        if remaining <= 0.2:
            return ""
        try:
            r = subprocess.run(
                args, cwd=cwd, capture_output=True, text=True,
                timeout=min(_GIT_CMD_TIMEOUT, remaining),
                encoding="utf-8", errors="replace", env=env,
                stdin=subprocess.DEVNULL,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        lines.append(f"Branch: {branch}")
    status = _run(["git", "status", "--porcelain"])
    if status:
        # Limit to first 20 lines
        status_lines = status.splitlines()[:20]
        lines.append("Git status:\n" + "\n".join(status_lines))
    log = _run(["git", "log", "-5", "--oneline"])
    if log:
        lines.append("Recent commits:\n" + log)

    text = "\n".join(lines)
    _env_git_cache[cwd] = {"text": text, "expires_at": time.time() + _ENV_GIT_TTL}
    return text


def _get_recent_files(cwd: str) -> str:
    """Return recently-modified files (last 24h). Cached 60s.

    Bounded walk: prunes heavy/vendored directories BEFORE descending, stops at
    _WALK_MAX_DEPTH, _WALK_MAX_ENTRIES files, or _WALK_TIME_BUDGET seconds.
    """
    cached = _env_files_cache.get(cwd)
    if cached and time.time() < cached["expires_at"]:
        return cached["text"]

    now = time.time()
    cutoff = now - 86400  # 24h
    deadline = time.monotonic() + _WALK_TIME_BUDGET
    recent = []
    seen = 0
    try:
        root_depth = len(Path(cwd).parts)
        for dirpath, dirnames, filenames in os.walk(cwd, topdown=True, onerror=lambda e: None):
            depth = len(Path(dirpath).parts) - root_depth
            # Prune in place so os.walk never descends into excluded trees.
            if depth >= _WALK_MAX_DEPTH:
                dirnames[:] = []
            else:
                dirnames[:] = [
                    d for d in dirnames
                    if d not in _WALK_PRUNE and not d.startswith(".")
                ]
            stop = False
            for fname in filenames:
                seen += 1
                if seen > _WALK_MAX_ENTRIES or time.monotonic() > deadline:
                    stop = True
                    break
                full = os.path.join(dirpath, fname)
                try:
                    mtime = os.stat(full).st_mtime
                except OSError:
                    continue
                if mtime >= cutoff:
                    try:
                        rel = os.path.relpath(full, cwd)
                    except ValueError:
                        rel = full
                    recent.append((mtime, rel.replace("\\", "/")))
            if stop:
                break
    except Exception:
        pass

    recent.sort(reverse=True)
    top = [name for _, name in recent[:10]]
    text = "Recently modified:\n" + "\n".join(top) if top else ""
    _env_files_cache[cwd] = {"text": text, "expires_at": time.time() + _ENV_FILES_TTL}
    return text


def _read_repo_doc_cached(path: Path) -> Optional[str]:
    """Read a repo instruction doc, cached on (mtime, size). None if unreadable."""
    key = str(path)
    try:
        st = path.stat()
    except OSError:
        _repo_doc_cache.pop(key, None)
        return None
    cached = _repo_doc_cache.get(key)
    if cached and cached["mtime"] == st.st_mtime and cached["size"] == st.st_size:
        return cached["text"]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
    except Exception:
        return None
    _repo_doc_cache[key] = {"mtime": st.st_mtime, "size": st.st_size, "text": text}
    return text


def _format_environment_context(workspace_dir: Optional[Path] = None) -> str:
    """Build a compact <env> block injected before each turn. Cross-platform.

    When *workspace_dir* is set, it is reported as the working directory and
    used as the base for git/recent-files context. Otherwise the process CWD
    is used (legacy behaviour).
    """
    if workspace_dir:
        try:
            cwd = str(Path(workspace_dir).resolve())
        except Exception:
            cwd = os.getcwd()
    else:
        default_root = getattr(config, "AGENT_DEFAULT_WORKSPACE", None)
        cwd = str(default_root) if default_root else os.getcwd()
    sys_name = platform.system()          # 'Windows', 'Linux', 'Darwin'
    sys_ver = platform.release()

    if sys_name == "Windows":
        shell = "PowerShell 5.1"
        script_ext = ".ps1"
    elif sys_name == "Darwin":
        shell = "zsh"
        script_ext = ".sh"
    else:
        shell = "bash"
        script_ext = ".sh"

    today = datetime.now().strftime("%Y-%m-%d (%A)")

    lines = [
        f"Working directory: {cwd}",
    ]
    if workspace_dir:
        lines.append("(this is the user-selected session workspace; relative "
                     "paths in file/shell tools resolve here first)")
    lines.extend([
        f"Platform: {sys_name} {sys_ver}",
        f"Shell: {shell}  Script extension: {script_ext}",
        f"Today: {today}",
    ])

    git_ctx = _get_git_context(cwd)
    if git_ctx:
        lines.append(git_ctx)

    files_ctx = _get_recent_files(cwd)
    if files_ctx:
        lines.append(files_ctx)

    block = "\n".join(lines)
    # Hard cap: keep under ~1.2 KB so it doesn't eat the dynamic context budget
    if len(block) > 1200:
        block = block[:1200] + "\n...[env truncated]"
    return f"## ENVIRONMENT\n{block}"


class PromptMixin:
    """Builds the static system prompt and per-request dynamic context."""

    def _build_system_prompt(self) -> str:
        """Return the STATIC system prompt (byte-stable for KV cache reuse)."""
        return _CACHED_SYSTEM_PROMPT

    def _build_dynamic_context(self, attached_files: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Return per-request dynamic context (RAG, memo, files). Separate from
        system prompt so the static prefix stays byte-identical for cache_prompt."""
        parts = []
        ws = getattr(self, "workspace_dir", None)
        # Environment block first — gives the model platform/shell/cwd/git before any other context
        env_ctx = _format_environment_context(ws)
        if env_ctx:
            parts.append(env_ctx)
        repo_docs = self._format_repo_instructions_context()
        if repo_docs:
            parts.append(repo_docs)
        rag_ctx = self._format_rag_collections_context()
        if rag_ctx:
            parts.append(rag_ctx)
        if self.username and "memo" in self.enabled_tools:
            memo_ctx = _load_memo_cached(self.username)
            if memo_ctx:
                memo_cap = getattr(config, "AGENT_MEMO_MAX_CHARS", 2000)
                if len(memo_ctx) > memo_cap:
                    memo_ctx = memo_ctx[:memo_cap] + "\n...[memo context truncated]"
                parts.append(memo_ctx)
        if self._session_todos:
            parts.append(self._format_todos())
        if attached_files:
            parts.append(self._format_attached_files(attached_files))
            inlined = self._format_attached_file_contents(attached_files)
            if inlined:
                parts.append(inlined)
        if not parts:
            return None
        dynamic_ctx = "\n".join(parts)
        dynamic_cap = getattr(config, "AGENT_DYNAMIC_CONTEXT_MAX_CHARS", 6000)
        if len(dynamic_ctx) > dynamic_cap:
            dynamic_ctx = dynamic_ctx[:dynamic_cap] + "\n...[dynamic context truncated]"
        return dynamic_ctx

    def _repo_doc_candidates(self) -> List[Path]:
        """Return repository instruction docs in precedence/read order.

        When a session workspace is set, scan it first for the standard
        agent-onboarding files. The API's own repo docs are appended as a
        fallback only when no workspace is set — pointing the agent at the
        API's `AGENTS.md` while the user is working on a different project
        is actively harmful.
        """
        candidates: List[Path] = []
        ws = getattr(self, "workspace_dir", None)
        if ws:
            ws_root = Path(ws).resolve()
            candidates.extend([
                ws_root / "AGENTS.md",
                ws_root / "CLAUDE.md",
                ws_root / ".claude" / "CLAUDE.md",
                ws_root / "README.md",
                ws_root / "docs" / "AGENTS.md",
                ws_root / "docs" / "README.md",
            ])
        else:
            repo_root = config.APP_DIR.parent.resolve()
            candidates.extend([
                repo_root / "AGENTS.md",
                repo_root / "CLAUDE.md",
                repo_root / "README.md",
                config.APP_DIR / "README.md",
                repo_root / "hoonbot" / "README.md",
                repo_root / "messenger" / "README.md",
            ])
        seen = set()
        unique = []
        for path in candidates:
            try:
                resolved = path.resolve()
            except Exception:
                continue
            if resolved not in seen:
                unique.append(resolved)
                seen.add(resolved)
        return unique

    def _format_repo_instructions_context(self) -> str:
        """Inject discovered repo docs before the model plans or edits."""
        cap = getattr(config, "AGENT_REPO_DOC_CONTEXT_MAX_CHARS", 12000)
        if cap <= 0:
            return ""

        sections = []
        remaining = cap
        ws = getattr(self, "workspace_dir", None)
        bases: List[Path] = []
        if ws:
            try:
                bases.append(Path(ws).resolve())
            except Exception:
                pass
        try:
            bases.append(config.APP_DIR.parent.resolve())
        except Exception:
            pass
        for path in self._repo_doc_candidates():
            if remaining <= 0:
                break
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue
            text = _read_repo_doc_cached(path)
            if not text or not text.strip():
                continue

            # Workspace docs live outside the API repo: relative_to() would raise
            # ValueError there. Prefer the workspace base, then the API repo,
            # then the absolute path.
            label = None
            for base in bases:
                try:
                    label = path.relative_to(base).as_posix()
                    break
                except ValueError:
                    continue
            if label is None:
                label = path.as_posix()
            header = f"\n### {label}\n"
            budget = remaining - len(header)
            if budget <= 0:
                break
            body = text[:budget]
            if len(text) > budget:
                body += "\n...[repo doc truncated]"
            sections.append(header + body)
            remaining -= len(header) + len(body)

        if not sections:
            return ""

        return (
            "\n\n## REPOSITORY INSTRUCTIONS DISCOVERED BEFORE WORK\n"
            "Follow these as repository guidance after higher-priority instructions. "
            "Use them before planning, editing, or verification.\n"
            + "\n".join(sections)
        )

    def _refresh_available_rag_collections(self):
        """Load available RAG collections for the current user (60s module-level TTL cache)."""
        self._available_rag_collections = []

        if "rag" not in self.enabled_tools or not self.username:
            return

        # Check module-level cache first
        cached = _rag_collections_cache.get(self.username)
        if cached and time.time() < cached["expires_at"]:
            self._available_rag_collections = cached["collections"]
            return

        try:
            from tools.rag import RAGTool
            tool = RAGTool(username=self.username)
            result = tool.list_collections()
            if not result.get("success"):
                return

            collections = result.get("collections", [])
            names = sorted({
                c.get("name")
                for c in collections
                if isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("name")
            })
            self._available_rag_collections = names
            _rag_collections_cache[self.username] = {
                "collections": names,
                "expires_at": time.time() + _RAG_CACHE_TTL,
            }
        except Exception as e:
            print(f"[RAG] Failed to load available collections for prompt context: {e}")

    def _get_available_rag_collections(self) -> List[str]:
        if self._available_rag_collections is None:
            self._refresh_available_rag_collections()
        return self._available_rag_collections or []

    def _format_rag_collections_context(self) -> str:
        if "rag" not in self.enabled_tools:
            return ""

        available = self._get_available_rag_collections()
        lines = ["\n\n## RAG COLLECTIONS"]
        lines.append("Use only existing collection_name values from this list when calling the rag tool.")
        if available:
            lines.append(f"Available collection_name values: {json.dumps(available, ensure_ascii=False)}")
        else:
            lines.append("Available collection_name values: []")
            lines.append("No collection exists yet. Ask the user to create a collection before using rag.")
        return "\n".join(lines)

    def _format_todos(self) -> str:
        """Format session todos for injection into dynamic context."""
        status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
        lines = ["\n\n## CURRENT TASKS"]
        lines.append("Track progress here. Mark tasks completed immediately when done.")
        for t in self._session_todos:
            icon = status_icon.get(t.get("status", "pending"), "[ ]")
            priority = t.get("priority", "medium")
            lines.append(f"  {icon} {t['id']}: {t['content']} (priority: {priority})")
        return "\n".join(lines)

    def _format_attached_file_contents(self, attached_files: List[Dict[str, Any]]) -> str:
        """Inline the full contents of attached text files when the total size
        is small enough to fit in the budget. Above the budget, return ""
        (the caller still emits a metadata preview separately).

        Skips files with errors, images, and anything not obviously text.
        """
        if not attached_files:
            return ""
        budget = max(0, getattr(config, "AGENT_ATTACHED_FILE_INLINE_BUDGET", 0))
        if budget <= 0:
            return ""
        text_categories = {"text", "code", "data"}
        eligible: List[Dict[str, Any]] = []
        total = 0
        for f in attached_files:
            if "error" in f or not f.get("path"):
                continue
            cat = f.get("category", "")
            if cat not in text_categories:
                continue
            size = f.get("size") or 0
            if size <= 0:
                continue
            total += size
            eligible.append(f)
        if not eligible or total > budget:
            return ""
        sections: List[str] = []
        remaining = budget
        for f in eligible:
            if remaining <= 0:
                break
            try:
                path = Path(f["path"])
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
            except Exception:
                continue
            name = f.get("name") or path.name
            header = f"\n### {name}\n"
            budget_for_body = remaining - len(header)
            if budget_for_body <= 0:
                break
            body = text[:budget_for_body]
            if len(text) > budget_for_body:
                body += "\n...[file truncated]"
            sections.append(header + body)
            remaining -= len(header) + len(body)
        if not sections:
            return ""
        return (
            "\n\n## ATTACHED FILE CONTENTS\n"
            "Full contents of small attached files, inlined so you can use them "
            "without re-reading. Trust this content as if you had run "
            "file_reader.\n"
            + "".join(sections)
        )

    def _format_attached_files(self, attached_files: List[Dict[str, Any]]) -> str:
        if not attached_files:
            return ""
        lines = ["\n\n## ATTACHED FILES"]
        lines.append(f"The user has attached {len(attached_files)} file(s).\n")
        for idx, f in enumerate(attached_files, 1):
            if "error" in f:
                lines.append(f"{idx}. {f['name']} - ERROR: {f['error']}")
                continue
            size_kb = f.get('size', 0) / 1024
            lines.append(f"{idx}. {f['name']} ({f.get('type', '?')}, {size_kb:.1f} KB)")
            if 'headers' in f:
                lines.append(f"   Columns: {', '.join(f['headers'])}")
                lines.append(f"   Rows: {f.get('rows', '?')}")
            if 'structure' in f:
                lines.append(f"   Structure: {f['structure']}")
                if f.get('keys'):
                    lines.append(f"   Keys: {', '.join(f['keys'][:10])}")
            if 'lines' in f:
                lines.append(f"   Lines: {f['lines']}")
                if f.get('definitions'):
                    lines.append(f"   Definitions: {', '.join(f['definitions'][:5])}")
            if 'preview' in f:
                preview_cap = getattr(config, "AGENT_FILE_PREVIEW_MAX_CHARS", 120)
                lines.append(f"   Preview: {str(f['preview'])[:preview_cap]}...")
        return "\n".join(lines)

    def _get_tool_schemas(self) -> Optional[List[Dict[str, Any]]]:
        if not self.enabled_tools:
            return None
        if self._filtered_tool_schemas is None:
            enabled = set(self.enabled_tools)
            self._filtered_tool_schemas = [
                s for s in _CACHED_TOOL_SCHEMAS
                if s["function"]["name"] in enabled
            ]
        return self._filtered_tool_schemas if self._filtered_tool_schemas else None
