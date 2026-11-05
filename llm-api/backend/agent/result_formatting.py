"""FormattingMixin: message builders and tool result microcompaction."""
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple

import config
from backend.core.llm_backend import ToolCall


# Status keys are emitted first so the model (and anything that truncates the
# tail) always sees whether the call worked before the payload.
_STATUS_KEYS = (
    "success", "exit_code", "returncode", "timed_out", "status",
    "error", "hint", "needs_clarification", "question",
)

# Per-tool primary payload field: it gets the FULL per-tool budget (other string
# fields share a smaller one). Tools not listed fall back to their largest
# top-level string field.
_PRIMARY_FIELD = {
    "file_reader": "content",
    "shell_exec": "stdout",
    "code_exec": "stdout",
    "agent": "result",
    "process_monitor": "stdout",
    "shell_lint": "output",
}

# Fields that are process output: truncate head+tail (the end of a log — the
# traceback, the final summary line — is usually what matters most).
_HEAD_TAIL_FIELDS = {"stdout", "stderr", "output", "result", "final_text"}

_HEAD_FRACTION = 0.3


def _head_tail(text: str, cap: int, head_fraction: float = _HEAD_FRACTION) -> str:
    """Keep the first ~30% and last ~70% of *cap* chars with an omission marker."""
    if cap <= 0 or len(text) <= cap:
        return text
    head = max(0, int(cap * head_fraction))
    tail = max(0, cap - head)
    omitted = len(text) - head - tail
    tail_text = text[-tail:] if tail else ""
    return f"{text[:head]}\n[… {omitted} chars omitted …]\n{tail_text}"


def _pageable_spill_text(result: Any) -> str:
    """Render a raw tool result as line-oriented text for a disk spill.

    String fields keep their real newlines between `===== key =====` markers;
    other fields are pretty-printed JSON. One-line JSON would defeat
    file_reader's line paging (offset/limit) and its per-read byte cap."""
    if not isinstance(result, dict):
        try:
            return json.dumps(result, ensure_ascii=False, default=str, indent=2)
        except Exception:
            return str(result)
    parts: List[str] = []
    for key, value in result.items():
        if isinstance(value, str):
            parts.append(f"===== {key} ({len(value)} chars) =====")
            parts.append(value)
            parts.append(f"===== end {key} =====")
        else:
            try:
                rendered = json.dumps(value, ensure_ascii=False, default=str, indent=2)
            except Exception:
                rendered = str(value)
            parts.append(f"===== {key} =====")
            parts.append(rendered)
    return "\n".join(parts) + "\n"


def _tool_budget(tool_name: str) -> int:
    """Per-tool char budget: TOOL_RESULT_BUDGETS (C12), else legacy TOOL_RESULT_BUDGET."""
    budgets = getattr(config, "TOOL_RESULT_BUDGETS", None)
    if not isinstance(budgets, dict):
        budgets = getattr(config, "TOOL_RESULT_BUDGET", None)
    if not isinstance(budgets, dict):
        budgets = {}
    default = budgets.get("default", getattr(config, "TOOL_RESULT_DEFAULT_BUDGET", 8000))
    try:
        return max(500, int(budgets.get(tool_name, default)))
    except (TypeError, ValueError):
        return 8000


class FormattingMixin:
    """Builds assistant/tool messages and applies per-tool result budgets."""

    def _build_assistant_tool_msg(self, tool_calls: List[ToolCall],
                                  content: Optional[str] = None) -> Dict[str, Any]:
        # Always send content as a string. Some vLLM chat templates
        # (Qwen, GLM, certain Llama-3 tool variants) reject content=null on
        # assistant messages with tool_calls and return 400 Bad Request
        # during chat-template rendering — manifests as 0-duration, 0-token
        # failures with no useful error.
        calls = []
        for tc in tool_calls:
            args = tc.function.arguments
            if not isinstance(args, dict):
                args = {}
            try:
                args_json = json.dumps(args, ensure_ascii=False, default=str)
            except Exception:
                args_json = "{}"
            calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": args_json,
                },
            })
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": calls,
        }

    def _build_tool_result_msg(self, tool_call: ToolCall, result: Dict[str, Any]) -> Dict[str, Any]:
        name = tool_call.function.name
        if not isinstance(result, dict):
            result = {"success": False, "error": f"Tool returned {type(result).__name__}", "output": result}
        budget = _tool_budget(name)

        # Spill the RAW result to disk FIRST (before any summarizing) whenever it
        # won't fit the budget, so the model can recover every byte with
        # file_reader via the absolute path referenced in the preview.
        # file_reader is skipped: its preview already pages the source file via
        # next_offset, and spilling a read would just create another file to read.
        spill_path: Optional[str] = None
        try:
            raw_json = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            raw_json = str(result)
        if len(raw_json) > budget and self.session_id and name != "file_reader":
            # Written as line-oriented text (not one-line JSON) so file_reader's
            # line-based offset/limit paging and byte cap can actually page it.
            spill_path = self._save_tool_result_to_disk(
                tool_call.id, _pageable_spill_text(result), suffix=".txt",
            )

        args = tool_call.function.arguments if isinstance(tool_call.function.arguments, dict) else {}
        preview = self._build_tool_result_preview(
            result, name, arguments=args, spill_path=spill_path,
        )
        # The raw JSON fit the budget but the preview still abbreviated a field
        # (a secondary string, a nested stream like process_monitor's
        # stdout.lines): spill anyway so the omitted part stays recoverable.
        if (
            spill_path is None
            and self.session_id
            and name != "file_reader"
            and isinstance(preview, dict)
            and preview.get("full_result_note")
        ):
            spill_path = self._save_tool_result_to_disk(
                tool_call.id, _pageable_spill_text(result), suffix=".txt",
            )
            if spill_path:
                preview = self._build_tool_result_preview(
                    result, name, arguments=args, spill_path=spill_path,
                )
        content = self._serialize_within_budget(preview, result, name, budget, args, spill_path)
        return {
            "role": "tool",
            "name": name,
            "content": content,
            "tool_call_id": tool_call.id,
        }

    def _serialize_within_budget(
        self,
        preview: Any,
        result: Dict[str, Any],
        tool_name: str,
        budget: int,
        arguments: Dict[str, Any],
        spill_path: Optional[str],
    ) -> str:
        """Serialize *preview* to JSON, shrinking caps until it fits a hard
        ceiling. Never slices serialized JSON — the content stays valid JSON."""
        # Primary field may use the whole budget; secondary fields, keys and
        # the spill note add overhead, so allow some headroom before shrinking.
        ceiling = int(budget * 1.5) + 512
        content = json.dumps(preview, ensure_ascii=False, default=str)
        scale = 1.0
        attempts = 0
        while len(content) > ceiling and attempts < 6:
            attempts += 1
            scale *= 0.6
            preview = self._build_tool_result_preview(
                result, tool_name, arguments=arguments,
                spill_path=spill_path, scale=scale,
            )
            content = json.dumps(preview, ensure_ascii=False, default=str)
        if len(content) > ceiling:
            # Pathological structure (thousands of tiny keys, etc.): fall back to
            # a minimal status-only object that is still valid JSON.
            minimal: Dict[str, Any] = {
                k: result.get(k) for k in _STATUS_KEYS if k in result
            }
            minimal["note"] = (
                f"Result too large to show ({len(content)} chars after summarizing)."
                + (f" Full result saved at {spill_path} — read it with file_reader "
                   f"(offset/limit)." if spill_path else "")
            )
            if spill_path:
                minimal["full_result_path"] = spill_path
            content = json.dumps(minimal, ensure_ascii=False, default=str)
        if spill_path:
            self._log(
                f"  [MICROCOMPACT] {tool_name} result summarized to {len(content)} chars; "
                f"raw result spilled to {spill_path}"
            )
        return content

    def _primary_field(self, result: Dict[str, Any], tool_name: str) -> Optional[str]:
        field = _PRIMARY_FIELD.get(tool_name)
        if field and isinstance(result.get(field), str):
            return field
        best, best_len = None, 0
        for key, val in result.items():
            if key in _STATUS_KEYS:
                continue
            if isinstance(val, str) and len(val) > best_len:
                best, best_len = key, len(val)
        return best

    def _build_tool_result_preview(
        self,
        result: Any,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        spill_path: Optional[str] = None,
        scale: float = 1.0,
    ) -> Any:
        """Create a bounded preview before serialising large tool payloads.

        Status keys come first, then the payload. The primary payload field
        (file_reader.content, shell stdout, ...) gets the full per-tool budget;
        other strings get a quarter of it. stdout/stderr-like fields are cut
        head+tail. file_reader truncation is line-aligned and reports
        `truncated: true` + `next_offset` so the model can page on.
        """
        budget = max(200, int(_tool_budget(tool_name) * scale))
        if not isinstance(result, dict):
            return self._summarize_tool_value(result, text_cap=budget, list_cap=6, depth=0)

        primary = self._primary_field(result, tool_name)
        secondary_cap = max(200, budget // 4)
        ordered: Dict[str, Any] = {}
        any_truncated = False
        # Set by _summarize_tool_value when it shortens a nested value.
        trunc_flag: List[bool] = [False]

        for key in _STATUS_KEYS:
            if key in result:
                val = result[key]
                if isinstance(val, str) and len(val) > secondary_cap:
                    val = _head_tail(val, secondary_cap)
                    any_truncated = True
                ordered[key] = val

        rest = [(k, v) for k, v in result.items() if k not in _STATUS_KEYS]
        item_cap = 24
        reader_extra: Dict[str, Any] = {}
        for key, val in rest[:item_cap]:
            if key == primary and isinstance(val, str):
                if tool_name == "file_reader" and key == "content":
                    new_val, reader_extra = self._truncate_file_reader_content(
                        val, budget, result, arguments or {},
                    )
                    ordered[key] = new_val
                    if reader_extra:
                        any_truncated = True
                    continue
                if len(val) > budget:
                    any_truncated = True
                    ordered[key] = (
                        _head_tail(val, budget) if key in _HEAD_TAIL_FIELDS or tool_name in _PRIMARY_FIELD
                        else val[:budget] + f"... [{len(val)} chars total]"
                    )
                else:
                    ordered[key] = val
                continue
            if isinstance(val, str):
                if len(val) > secondary_cap:
                    any_truncated = True
                    ordered[key] = (
                        _head_tail(val, secondary_cap) if key in _HEAD_TAIL_FIELDS
                        else val[:secondary_cap] + f"... [{len(val)} chars total]"
                    )
                else:
                    ordered[key] = val
                continue
            if isinstance(val, dict) and (key in _HEAD_TAIL_FIELDS or key == _PRIMARY_FIELD.get(tool_name)):
                # Structured stream output (process_monitor read_output:
                # stdout/stderr = {"lines": str, "next_offset": ...}). Its text
                # is process output: cut head+tail (the newest lines — errors,
                # progress, tracebacks — are what matter), with the full budget
                # when it is the tool's primary field.
                cap = budget if key == _PRIMARY_FIELD.get(tool_name) else secondary_cap
                nested: Dict[str, Any] = {}
                for k2, v2 in list(val.items())[:item_cap]:
                    if isinstance(v2, str):
                        if len(v2) > cap:
                            nested[k2] = _head_tail(v2, cap)
                            any_truncated = True
                        else:
                            nested[k2] = v2
                    else:
                        nested[k2] = self._summarize_tool_value(
                            v2, text_cap=secondary_cap, list_cap=6, depth=2, flag=trunc_flag,
                        )
                if len(val) > item_cap:
                    nested["_truncated_keys"] = len(val) - item_cap
                    any_truncated = True
                ordered[key] = nested
                continue
            summarized = self._summarize_tool_value(
                val, text_cap=secondary_cap, list_cap=6, depth=1, flag=trunc_flag,
            )
            ordered[key] = summarized
        if trunc_flag[0]:
            any_truncated = True
        if len(rest) > item_cap:
            ordered["_truncated_keys"] = len(rest) - item_cap
            any_truncated = True
        # Applied after the payload loop so the reader's own `truncated: false`
        # can't overwrite our truncated/next_offset.
        if reader_extra:
            ordered.update(reader_extra)

        if spill_path:
            ordered["full_result_path"] = spill_path
            ordered["full_result_note"] = (
                "Output above is abbreviated. The complete raw result is saved at "
                "full_result_path (absolute) — read it with file_reader using "
                "offset/limit if you need the omitted part."
            )
        elif any_truncated and "full_result_note" not in ordered:
            ordered["full_result_note"] = "Output above is abbreviated."
        return ordered

    def _truncate_file_reader_content(
        self,
        content: str,
        budget: int,
        result: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Line-aligned head truncation for file_reader content.

        Returns (content, extra_fields). extra_fields carries truncated=True and
        next_offset (1-based line to pass as `offset` on the next read) when the
        view stops before the end of the file — whether because of this budget
        or the reader's own byte cap.
        """
        try:
            start_line = max(1, int(arguments.get("offset") or 1))
        except (TypeError, ValueError):
            start_line = 1
        total_lines = result.get("total_lines")
        lines_returned = result.get("lines_returned")
        extra: Dict[str, Any] = {}

        if len(content) > budget:
            cut = content.rfind("\n", 0, budget)
            if cut <= 0:
                cut = budget
                kept = content[:cut]
                kept_lines = kept.count("\n")  # last line is partial: re-read it
            else:
                kept = content[:cut + 1]
                kept_lines = kept.count("\n")
            next_offset = start_line + max(1, kept_lines) if kept_lines else start_line + 1
            extra["truncated"] = True
            extra["next_offset"] = next_offset
            extra["truncation_note"] = (
                f"Content cut at {len(kept)} of {len(content)} chars (lines "
                f"{start_line}-{next_offset - 1} shown). Continue with "
                f"file_reader(offset={next_offset}). Do NOT rewrite this file "
                "from this partial view."
            )
            return kept, extra

        if result.get("truncated") and isinstance(lines_returned, int):
            next_offset = start_line + max(1, lines_returned - 1)
            if not isinstance(total_lines, int) or next_offset <= total_lines:
                extra["next_offset"] = next_offset
                extra["truncation_note"] = (
                    f"Reader byte cap reached. Continue with file_reader(offset={next_offset})."
                )
        return content, extra

    def _summarize_tool_value(
        self,
        value: Any,
        text_cap: int,
        list_cap: int,
        depth: int,
        flag: Optional[List[bool]] = None,
    ) -> Any:
        """Bounded copy of *value*. Sets flag[0] = True when anything was cut,
        so the caller can mark the result abbreviated (and spill it)."""
        if isinstance(value, str):
            if len(value) <= text_cap:
                return value
            if flag is not None:
                flag[0] = True
            # Head+tail: the end of nested text (a log tail, a final status)
            # is usually what matters.
            return _head_tail(value, text_cap)

        if isinstance(value, list):
            item_cap = list_cap if depth == 0 else max(2, list_cap - 2)
            summarized = [
                self._summarize_tool_value(
                    item,
                    text_cap=max(80, text_cap // 2),
                    list_cap=max(2, list_cap - 1),
                    depth=depth + 1,
                    flag=flag,
                )
                for item in value[:item_cap]
            ]
            if len(value) > item_cap:
                summarized.append(f"... [{len(value) - item_cap} more items]")
                if flag is not None:
                    flag[0] = True
            return summarized

        if isinstance(value, dict):
            item_cap = 24 if depth == 0 else 10
            summarized = {
                key: self._summarize_tool_value(
                    item,
                    text_cap=max(80, text_cap // 2),
                    list_cap=max(2, list_cap - 1),
                    depth=depth + 1,
                    flag=flag,
                )
                for key, item in list(value.items())[:item_cap]
            }
            if len(value) > item_cap:
                summarized["_truncated_keys"] = len(value) - item_cap
                if flag is not None:
                    flag[0] = True
            return summarized

        return value

    def _truncate_tool_result(self, tool_name: str, content: str, call_id: str = "") -> str:
        """Bound an already-serialized tool result string to its per-tool budget.

        Kept for callers that hold a plain string. The FULL content is saved to
        disk first and the marker carries the ABSOLUTE path so file_reader can
        open it from any workspace. (Structured results go through
        _build_tool_result_msg, which keeps JSON valid.)"""
        budget = _tool_budget(tool_name)
        if len(content) <= budget:
            return content

        self._log(f"  [MICROCOMPACT] {tool_name} result truncated: {len(content)} -> {budget} chars")

        disk_hint = ""
        if self.session_id:
            abs_path = self._save_tool_result_to_disk(call_id, content)
            if abs_path:
                disk_hint = f" — full result at {abs_path} (read with file_reader)"

        return _head_tail(content, budget) + f"\n...[truncated to {budget}/{len(content)} chars{disk_hint}]"

    def _save_tool_result_to_disk(self, call_id: str, content: str,
                                  suffix: str = ".txt") -> Optional[str]:
        """Persist an oversized raw tool result to disk; return its ABSOLUTE path
        (or None on failure). Relative paths were unresolvable by file_reader,
        which resolves against the workspace, not llm-api's APP_DIR."""
        safe_id = "".join(
            c if (c.isalnum() or c in "-_.") else "_" for c in (call_id or "")
        )[:64].strip(".") or f"call_{uuid.uuid4().hex[:8]}"
        session_dir = config.TOOL_RESULTS_DIR / (self.session_id or "default")
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / f"{safe_id}{suffix}"
            if path.exists():
                # Never clobber an earlier spill that a prior marker points at.
                path = session_dir / f"{safe_id}_{uuid.uuid4().hex[:6]}{suffix}"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            abs_path = str(path.resolve())
        except Exception as e:
            print(f"[MICROCOMPACT] Failed to save tool result to disk: {e}")
            return None
        spills = getattr(self, "_spill_paths", None)
        if spills is None:
            try:
                self._spill_paths = {}
                spills = self._spill_paths
            except Exception:
                spills = None
        if spills is not None and call_id:
            spills[call_id] = abs_path
        return abs_path
