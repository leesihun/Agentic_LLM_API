"""Live tool / turn registry and supervisor helpers (round 4: visibility + backup).

Two jobs live here, both deliberately dependency-light and thread-safe:

1. A process-wide registry of the tool calls and turns that are running right
   now (`register_tool`, `note_turn`, `snapshot_running`, `snapshot_turn`).
   The HTTP edge (GET /sessions/{id}/activity) and hoonbot's @status read it;
   the agent loop writes it. Every call is O(1)/O(n_running) under one lock.

2. Pure helpers for the supervisor ("watchdog") that judges long tool calls and
   long turns: `describe_call` (short human summary of a tool call),
   `trim_processes`, the fixed supervisor prompts, and tolerant JSON parsing of
   the supervisor's verdict. The LLM calls themselves are made by the agent
   loop (see DispatchMixin._supervise_tool / AgentLoop._maybe_review_turn).

Everything here fails open: a broken snapshot or an unparseable verdict never
raises into the agent loop.
"""
import json
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # P2 owns this module; every use below is defensive.
    from backend.utils import proc_lifecycle as _proc_lifecycle
except Exception:  # pragma: no cover - module always exists in this repo
    _proc_lifecycle = None


_CHILD_MARK = "__sub_"

_LOCK = threading.Lock()
# tool_call_id -> {tool_name, tool_call_id, session_id, detail, started (monotonic), started_at (epoch)}
_TOOLS: Dict[str, Dict[str, Any]] = {}
# session_id -> {started, started_at, iteration, state, last_activity, ...extra primitives}
_TURNS: Dict[str, Dict[str, Any]] = {}
# Leak guards: a crashed run that never unregistered must not grow forever.
_MAX_TOOLS = 1000
_MAX_TURNS = 1000


def _belongs(entry_session: Optional[str], wanted: str, include_children: bool = True) -> bool:
    if entry_session is None:
        return False
    if entry_session == wanted:
        return True
    return include_children and entry_session.startswith(wanted + _CHILD_MARK)


# --------------------------------------------------------------------------
# K3 registry
# --------------------------------------------------------------------------

def _tool_key(session_id: Optional[str], tool_call_id: str) -> str:
    # Keyed by (session, call id): a parser that reuses ids (e.g. "call_0") in two
    # concurrent sessions must not make one session's call hide the other's.
    return f"{session_id or ''}\x00{tool_call_id}"


def register_tool(session_id: Optional[str], tool_call_id: str, tool_name: str, detail: str = "") -> None:
    """Record a tool call as running. Never raises."""
    if not tool_call_id:
        return
    try:
        now = time.monotonic()
        with _LOCK:
            if len(_TOOLS) >= _MAX_TOOLS:
                oldest = min(_TOOLS.items(), key=lambda kv: kv[1].get("started", 0.0))[0]
                _TOOLS.pop(oldest, None)
            _TOOLS[_tool_key(session_id, tool_call_id)] = {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "session_id": session_id,
                "detail": detail or "",
                "started": now,
                "started_at": time.time(),
            }
    except Exception:
        pass


def unregister_tool(tool_call_id: str, session_id: Optional[str] = None) -> None:
    try:
        with _LOCK:
            _TOOLS.pop(_tool_key(session_id, tool_call_id), None)
    except Exception:
        pass


def note_turn(session_id: Optional[str], **fields: Any) -> None:
    """Create/refresh the turn record of *session_id* (start time is set on the
    first call). Recognised fields: iteration, state ("llm"|"tools"|"final"),
    plus any other primitive-valued extras. Always bumps last_activity."""
    if not session_id:
        return
    try:
        now = time.monotonic()
        with _LOCK:
            turn = _TURNS.get(session_id)
            if turn is None:
                if len(_TURNS) >= _MAX_TURNS:
                    oldest = min(_TURNS.items(), key=lambda kv: kv[1].get("started", 0.0))[0]
                    _TURNS.pop(oldest, None)
                turn = {
                    "started": now,
                    "started_at": time.time(),
                    "iteration": 0,
                    "state": "llm",
                }
                _TURNS[session_id] = turn
            for key, value in fields.items():
                if key in ("started", "started_at", "last_activity"):
                    continue
                if value is None or isinstance(value, (str, int, float, bool)):
                    turn[key] = value
            turn["last_activity"] = now
    except Exception:
        pass


def clear_turn(session_id: Optional[str]) -> None:
    if not session_id:
        return
    try:
        with _LOCK:
            _TURNS.pop(session_id, None)
    except Exception:
        pass


def snapshot_turn(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """{started_elapsed, iteration, state, idle_seconds, ...} or None."""
    if not session_id:
        return None
    try:
        now = time.monotonic()
        with _LOCK:
            turn = _TURNS.get(session_id)
            if turn is None:
                return None
            snap = dict(turn)
        started = snap.pop("started", now)
        last = snap.pop("last_activity", now)
        snap["started_elapsed"] = round(max(0.0, now - started), 1)
        snap["idle_seconds"] = round(max(0.0, now - last), 1)
        return snap
    except Exception:
        return None


def snapshot_running(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Running tool calls: [{tool_name, tool_call_id, session_id, detail,
    elapsed, processes}]. With *session_id*, child sessions are included."""
    try:
        now = time.monotonic()
        with _LOCK:
            entries = [
                dict(e) for e in _TOOLS.values()
                if session_id is None or _belongs(e.get("session_id"), session_id)
            ]
    except Exception:
        return []
    entries.sort(key=lambda e: e.get("started", 0.0))
    out: List[Dict[str, Any]] = []
    for e in entries:
        out.append({
            "tool_name": e.get("tool_name", ""),
            "tool_call_id": e.get("tool_call_id", ""),
            "session_id": e.get("session_id"),
            "detail": e.get("detail", ""),
            "elapsed": round(max(0.0, now - e.get("started", now)), 1),
            "processes": trim_processes(live_processes(
                e.get("tool_call_id"), session_id=e.get("session_id"), include_children=False,
            )),
        })
    return out


def tool_elapsed(tool_call_id: str, session_id: Optional[str] = None) -> Optional[float]:
    """Seconds a registered tool call has been running (None if unknown)."""
    try:
        with _LOCK:
            e = _TOOLS.get(_tool_key(session_id, tool_call_id))
            started = e.get("started") if e else None
        return None if started is None else max(0.0, time.monotonic() - started)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Process snapshots (proc_lifecycle K2, defensive)
# --------------------------------------------------------------------------

def live_processes(tool_call_id: Optional[str] = None, session_id: Optional[str] = None,
                   include_children: bool = True) -> List[Dict[str, Any]]:
    """proc_lifecycle.snapshot_processes with every failure mapped to []."""
    fn = getattr(_proc_lifecycle, "snapshot_processes", None) if _proc_lifecycle else None
    if fn is None:
        return []
    try:
        procs = fn(session_id=session_id, tool_call_id=tool_call_id,
                   include_children_sessions=include_children)
        return list(procs or [])
    except Exception:
        return []


def _last_lines(text: str, cap: int = 160) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    joined = " ⏎ ".join(lines[-2:])
    return joined[-cap:]


def trim_processes(procs: Iterable[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    """K1 shape: [{pid, command, elapsed, idle, tail}] (<=limit, command<=100,
    tail<=160). Dead entries are skipped."""
    out: List[Dict[str, Any]] = []
    try:
        for p in procs or ():
            if not isinstance(p, dict) or p.get("alive") is False:
                continue
            out.append({
                "pid": p.get("pid"),
                "command": _oneline(p.get("command"), 100),
                "elapsed": round(float(p.get("elapsed") or 0.0), 1),
                "idle": round(float(p.get("idle_seconds") or 0.0), 1),
                "tail": _last_lines(p.get("tail") or "", 160),
            })
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


def stalled_process(procs: Iterable[Dict[str, Any]], stall_seconds: float) -> Optional[Dict[str, Any]]:
    """First live FOREGROUND process silent for >= stall_seconds (process_monitor
    background processes are never judged)."""
    try:
        for p in procs or ():
            if not isinstance(p, dict) or p.get("alive") is False:
                continue
            if p.get("tool") == "process_monitor":
                continue
            idle = float(p.get("idle_seconds") or 0.0)
            elapsed = float(p.get("elapsed") or 0.0)
            if idle >= stall_seconds and elapsed >= stall_seconds:
                return p
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# describe_call: short human summary of a tool call
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_SECRET_RES = (
    (re.compile(r"(?i)(--?(?:token|password|passwd|secret|api[-_]?key|apikey)[ =]+)\S+"), r"\1***"),
    (re.compile(r"(?i)(authorization:\s*(?:bearer|basic)?\s*)\S+"), r"\1***"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1***"),
    (re.compile(r"(?i)\b((?:token|password|passwd|secret|api[-_]?key|apikey)=)[^\s&;\"']+"), r"\1***"),
)
_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Delete|Update) File:\s*(.+?)\s*$", re.MULTILINE)


def mask_text(value: Any) -> str:
    """Best-effort credential masking for text that leaves the agent loop as
    display / supervisor-prompt data. Uses proc_lifecycle.mask_secrets (the
    stronger rule set: `*_TOKEN=`/`*_PASSWORD=` assignments, `-u user:pw`, URL
    userinfo, well-known key shapes) plus this module's own rules."""
    text = str(value if value is not None else "")
    fn = getattr(_proc_lifecycle, "mask_secrets", None) if _proc_lifecycle else None
    if fn is not None:
        try:
            text = fn(text)
        except Exception:
            pass
    for rx, repl in _SECRET_RES:
        text = rx.sub(repl, text)
    return text


def _oneline(value: Any, cap: int) -> str:
    # Mask BEFORE truncating, so a secret cut in half is never half-shown.
    text = _WS_RE.sub(" ", mask_text(value)).strip()
    if len(text) > cap:
        text = text[: max(1, cap - 1)] + "…"
    return text


def describe_call(name: str, arguments: Any, cap: int = 120) -> str:
    """<=cap-char single-line summary of what a tool call does, e.g.
    "list /srv/data" (file_navigator) or the shell command (shell_exec).
    Never raises."""
    try:
        a: Dict[str, Any] = arguments if isinstance(arguments, dict) else {}

        def s(key: str) -> str:
            v = a.get(key)
            return "" if v in (None, "") else str(v)

        if name == "file_navigator":
            parts = [s("operation"), s("path"), s("pattern")]
            text = " ".join(p for p in parts if p)
        elif name == "shell_exec":
            text = s("command")
        elif name == "code_exec":
            code = s("code")
            first = next((ln.strip() for ln in code.splitlines() if ln.strip()), "")
            text = first
        elif name == "grep":
            pat, path = s("pattern"), s("path")
            text = f"{pat} in {path}" if path else pat
        elif name in ("file_reader", "file_writer", "file_edit"):
            text = s("path")
        elif name == "apply_patch":
            files = _PATCH_FILE_RE.findall(s("patch"))
            text = ", ".join(files[:2]) + (f" +{len(files) - 2}" if len(files) > 2 else "")
        elif name == "agent":
            text = s("description") or " ".join(
                p for p in (s("subagent_type"), s("prompt")[:80]) if p
            )
        elif name in ("rag", "websearch"):
            text = s("query")
        elif name == "shell_lint":
            text = s("path")
        elif name == "memo":
            text = " ".join(p for p in (s("operation"), s("key")) if p)
        elif name == "process_monitor":
            text = " ".join(p for p in (s("operation"), s("handle"), s("command")) if p)
        elif name == "todo_write":
            todos = a.get("todos")
            text = f"{len(todos)} items" if isinstance(todos, list) else ""
        else:
            text = ""
            for v in a.values():
                if isinstance(v, str) and v.strip():
                    text = v
                    break
        return _oneline(text, cap)
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Supervisor prompts + tolerant parsing
# --------------------------------------------------------------------------

TOOL_REVIEW_SYSTEM = (
    "You are a strict but fair supervisor of an autonomous coding agent. One of the "
    "agent's tool calls has been running for a long time. Decide whether it is still "
    "reasonable to keep waiting. Answer \"stop\" ONLY on positive evidence that the call is "
    "hung or unreasonable: the last output shows it is waiting for interactive input, an "
    "endless loop or the same error/output repeating, or work clearly unrelated to the "
    "user's request. Silence ALONE is NOT evidence: builds, installs, training, sleeps and "
    "commands that redirect their output to a file are quiet by nature, and the tool "
    "enforces its own declared timeout (an explicit timeout was the agent's deliberate "
    "choice). If there is any sign of progress, or a long runtime is plausible for this "
    "kind of task, answer \"continue\".\n"
    "Reply with ONLY one JSON object, no prose: "
    "{\"action\":\"continue\"|\"stop\",\"reason\":\"<one short sentence, in the user's language>\"}\n"
    "(감독자 지침: 확실히 멈췄거나 비합리적일 때만 stop, 그 외에는 continue.)"
)

TURN_REVIEW_SYSTEM = (
    "You are a supervisor of an autonomous coding agent that has been working on one "
    "user request for a long time. From the summary of its work so far, judge whether "
    "it is on track. Verdicts: \"continue\" (making real progress toward the request), "
    "\"replan\" (going in circles, doing sequential work that could be split, or the plan "
    "is too broad), \"wrap_up\" (the request is essentially satisfied or further work has "
    "diminishing returns — it should deliver now). Prefer \"continue\" unless there is "
    "clear evidence otherwise.\n"
    "Reply with ONLY one JSON object, no prose: "
    "{\"verdict\":\"continue\"|\"replan\"|\"wrap_up\",\"note\":\"<one or two short sentences, in the user's language>\"}\n"
    "(감독자 지침: 진행이 정상이면 continue.)"
)


def parse_supervisor_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Tolerant extraction of the first JSON object in *text* (code fences,
    leading prose and trailing junk are ignored). None when nothing parses."""
    if not text:
        return None
    raw = str(text)
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw)
    if "</think>" in raw:  # orphan close: the reasoning parser consumed the opener
        raw = raw.rsplit("</think>", 1)[1]
    if "<think>" in raw:  # never closed: nothing usable after it
        raw = raw.split("<think>", 1)[0]
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
        start = raw.find("{", start + 1)
    # Last resort: pull the verdict keys out with regexes.
    out: Dict[str, Any] = {}
    m = re.search(r'"?action"?\s*[:=]\s*"?(continue|stop)', raw, re.IGNORECASE)
    if m:
        out["action"] = m.group(1).lower()
    m = re.search(r'"?verdict"?\s*[:=]\s*"?(continue|replan|wrap_up)', raw, re.IGNORECASE)
    if m:
        out["verdict"] = m.group(1).lower()
    m = re.search(r'"?(?:reason|note)"?\s*[:=]\s*"([^"]{1,300})', raw)
    if m:
        out["reason"] = out["note"] = m.group(1)
    return out or None


def parse_tool_verdict(text: Optional[str]) -> Tuple[str, str]:
    """(action, reason). Any failure -> ("continue", "")."""
    data = parse_supervisor_json(text)
    if not isinstance(data, dict):
        return "continue", ""
    action = str(data.get("action", "continue")).strip().lower()
    if action != "stop":
        action = "continue"
    reason = _oneline(data.get("reason") or data.get("note") or "", 200)
    return action, reason


def parse_turn_verdict(text: Optional[str]) -> Tuple[str, str]:
    """(verdict, note). Any failure -> ("continue", "")."""
    data = parse_supervisor_json(text)
    if not isinstance(data, dict):
        return "continue", ""
    verdict = str(data.get("verdict", "continue")).strip().lower().replace("-", "_").replace(" ", "_")
    if verdict not in ("continue", "replan", "wrap_up"):
        verdict = "continue"
    note = _oneline(data.get("note") or data.get("reason") or "", 300)
    return verdict, note


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 90:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 90:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def build_tool_review_prompt(
    tool_name: str, detail: str, elapsed: float, checks: int, request: str,
    procs: List[Dict[str, Any]],
    declared_timeout: Optional[float] = None,
    explicit_timeout: bool = False,
) -> str:
    lines = [
        f"User request (headline): {_oneline(request, 700) or '(unknown)'}",
        f"Tool: {tool_name}",
        f"Call: {detail or '(no summary)'}",
        f"Running for: {fmt_duration(elapsed)} (supervisor check #{checks})",
    ]
    if declared_timeout:
        lines.append(
            f"Declared timeout: {fmt_duration(declared_timeout)} "
            f"({'set explicitly by the agent' if explicit_timeout else 'default'}); "
            f"the tool stops itself then."
        )
    if procs:
        lines.append("Live OS processes of this call:")
        for p in procs[:3]:
            tail = _oneline(p.get("tail") or "", 300) or "(no output yet)"
            lines.append(
                f"- pid {p.get('pid')} | {_oneline(p.get('command'), 160)} | "
                f"running {fmt_duration(float(p.get('elapsed') or 0))} | "
                f"silent for {fmt_duration(float(p.get('idle_seconds') or 0))} | last output: {tail}"
            )
    else:
        lines.append("Live OS processes of this call: none (in-process tool; no output signal available).")
    lines.append("Should the agent keep waiting? Answer with the JSON object only.")
    return "\n".join(lines)
