"""LoggingMixin: prompts.log write helpers for agent-level events."""
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import config
from backend.core.llm_backend import ToolCall
from backend.utils.agent_log_banner import print_agent_log_banner_once
from backend.utils.prompts_log_append import append_capped_prompts_log


class LoggingMixin:
    """Provides _log* methods that write structured agent events to prompts.log."""

    def _summary_logging_enabled(self) -> bool:
        return self._log_level in {"summary", "debug"}

    def _debug_logging_enabled(self) -> bool:
        return self._log_level == "debug"

    def _write_log_sync(self, message: str):
        path = Path(getattr(config, "AGENT_LOG_PATH", config.PROMPTS_LOG_PATH))
        print_agent_log_banner_once(path)
        append_capped_prompts_log(
            message if message.endswith("\n") else message + "\n",
            path=path,
        )

    def _log(self, message: str):
        """Append a line to prompts.log."""
        if not self._summary_logging_enabled():
            return
        try:
            if getattr(config, "AGENT_LOG_ASYNC", True):
                try:
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(None, self._write_log_sync, message)
                    return
                except RuntimeError:
                    pass
            self._write_log_sync(message)
        except Exception as e:
            print(f"[AGENT-LOG] Failed to write to prompts.log: {e}")

    def _log_block(self, lines: List[str]):
        """Write multiple lines to prompts.log as a single block."""
        self._log('\n'.join(lines))

    def _log_iteration_start(self, iteration: int, streaming: bool = False):
        tag = "STREAM " if streaming else ""
        self._log_block([
            "",
            "~" * 80,
            f">>> AGENT {tag}ITERATION {iteration + 1}/{self.max_iterations}",
            "~" * 80,
            f"  Session:     {self.session_id or 'N/A'}",
            f"  Username:    {self.username or 'N/A'}",
            f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "~" * 80,
            "",
        ])

    def _log_tool_calls_requested(self, tool_calls: List[ToolCall], iteration: int):
        if not self._summary_logging_enabled():
            return
        lines = [
            "",
            "-" * 80,
            f">>> LLM REQUESTED TOOL CALLS (Iteration {iteration + 1})",
            "-" * 80,
            f"  Tool Count:  {len(tool_calls)}",
            f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        for i, tc in enumerate(tool_calls, 1):
            lines.append(f"  [{i}] {tc.function.name} (id: {tc.id})")
            arg_cap = 500 if self._debug_logging_enabled() else 280
            args = tc.function.arguments
            parse_error = getattr(tc.function, "parse_error", None)
            if parse_error:
                lines.append(f"      PARSE ERROR: {str(parse_error)[:arg_cap]}")
            if isinstance(args, dict):
                arg_items = list(args.items())
            else:
                arg_items = [("(non-dict arguments)", args)]
            for k, v in arg_items:
                sv = str(v)
                if len(sv) > arg_cap:
                    sv = sv[:arg_cap] + f"... [{len(sv)} chars total]"
                lines.append(f"      {k}: {sv}")
            lines.append("")
        lines.append("-" * 80)
        lines.append("")
        self._log_block(lines)

    def _log_tool_result(self, tool_name: str, tool_call_id: str,
                         result: Dict[str, Any], duration: float):
        if not self._summary_logging_enabled():
            return
        success = result.get("success", True)
        lines = [
            "",
            "." * 80,
            f"<<< TOOL RESULT: {tool_name} [{tool_call_id or 'N/A'}]",
            "." * 80,
            f"  Status:      {'SUCCESS' if success else 'FAILED'}",
            f"  Duration:    {duration:.2f}s",
        ]
        if not success and result.get("error"):
            lines.append(f"  Error:       {str(result['error'])[:500]}")
        if self._debug_logging_enabled():
            result_str = json.dumps(result, ensure_ascii=False, default=str)
            if len(result_str) > 1500:
                lines.append(f"  Result:      {result_str[:1500]}")
                lines.append(f"               ... [{len(result_str)} chars total]")
            else:
                lines.append(f"  Result:      {result_str}")
        else:
            if "executed" in result:
                lines.append(f"  Executed:    {result['executed']}")
            if "returncode" in result:
                lines.append(f"  Return Code: {result['returncode']}")
            preview_cap = 450
            for key in ("stdout", "stderr", "output", "message"):
                chunk = result.get(key)
                if chunk is None:
                    continue
                s = str(chunk).strip()
                if not s:
                    continue
                label = key.upper()
                if len(s) > preview_cap:
                    s = s[:preview_cap] + f"... [{len(str(chunk))} chars total]"
                lines.append(f"  {label}:       {s}")
        lines.append("." * 80)
        lines.append("")
        self._log_block(lines)

    def _log_execution_summary(self, tool_calls: List[ToolCall],
                               results: List[Dict[str, Any]],
                               durations: List[float], iteration: int):
        succeeded = sum(1 for r in results if r.get("success", True))
        failed = len(results) - succeeded
        wall_time = max(durations) if durations else 0
        lines = [
            "",
            "-" * 80,
            f">>> TOOL EXECUTION SUMMARY (Iteration {iteration + 1})",
            "-" * 80,
            f"  Tools Run:   {len(results)}",
            f"  Succeeded:   {succeeded}",
            f"  Failed:      {failed}",
            f"  Wall Time:   {wall_time:.2f}s (parallel execution)",
            "",
        ]
        for i, (tc, res, dur) in enumerate(zip(tool_calls, results, durations), 1):
            status = "SUCCESS" if res.get("success", True) else "FAILED"
            lines.append(f"  [{i}] {tc.function.name:<20s} -- {status} ({dur:.2f}s)")
        lines.extend(["", "-" * 80, ""])
        self._log_block(lines)

    def _log_agent_complete(self, reason: str, iterations_used: int):
        self._log_block([
            "",
            "~" * 80,
            ">>> AGENT COMPLETE",
            "~" * 80,
            f"  Reason:      {reason}",
            f"  Iterations:  {iterations_used}",
            f"  Tool Calls:  {len(self.tool_calls_log)}",
            f"  Session:     {self.session_id or 'N/A'}",
            f"  Username:    {self.username or 'N/A'}",
            f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "~" * 80,
            "",
        ])
