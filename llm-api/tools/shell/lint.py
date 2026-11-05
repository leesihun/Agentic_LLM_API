"""ShellLintTool: static analysis for .ps1 (PSScriptAnalyzer) and .sh (shellcheck/bash -n)."""
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


_MAX_FINDINGS = 50
_TIMEOUT = 30
_PS_UTF8 = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $ProgressPreference='SilentlyContinue'; "
_NO_PSSA_EXIT = 97  # sentinel: PSScriptAnalyzer module not installed


def _ps_quote(value: str) -> str:
    """Single-quoted PowerShell literal ('' escapes a quote)."""
    return "'" + str(value).replace("'", "''") + "'"


def _run(argv: List[str]) -> subprocess.CompletedProcess:
    """Run a linter non-interactively with UTF-8 decoding and a hard timeout."""
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_TIMEOUT,
    )


class ShellLintTool:
    """Run PSScriptAnalyzer (Windows) or shellcheck/bash -n (Unix) on a shell script."""

    def __init__(self, workspace_dir: Optional[str] = None):
        # Relative paths resolve like the other file tools: session workspace,
        # else AGENT_DEFAULT_WORKSPACE.
        base = workspace_dir or getattr(config, "AGENT_DEFAULT_WORKSPACE", None) or Path.cwd()
        self.base_dir = Path(base)

    def _resolve(self, path: str) -> Path:
        p = Path(str(path)).expanduser()
        if p.is_absolute():
            return p
        candidate = (self.base_dir / p)
        if candidate.exists():
            return candidate.resolve()
        return p.resolve()

    def lint(self, path: str) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "path is required."}
        p = self._resolve(path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if not p.is_file():
            return {"success": False, "error": f"Not a file: {p}"}

        ext = p.suffix.lower()
        is_windows = platform.system() == "Windows"

        if ext == ".ps1":
            return self._lint_ps1(p, is_windows)
        elif ext in (".sh", ".bash"):
            return self._lint_sh(p, is_windows)
        else:
            # Generic: try bash -n on unix; PSParser on windows
            if is_windows:
                return self._lint_ps1(p, is_windows)
            return self._lint_sh(p, is_windows)

    # ------------------------------------------------------------------ #
    # PowerShell linting
    # ------------------------------------------------------------------ #

    def _lint_ps1(self, path: Path, is_windows: bool) -> Dict[str, Any]:
        # Prefer PSScriptAnalyzer (gives rule names + severity)
        psa_result = self._try_psscriptanalyzer(path, is_windows)
        if psa_result is not None:
            return psa_result

        # Fallback: PSParser syntax-only check (always available if pwsh/powershell exists)
        return self._try_psparser(path, is_windows)

    def _try_psscriptanalyzer(self, path: Path, is_windows: bool) -> Optional[Dict[str, Any]]:
        shell_exe = "powershell.exe" if is_windows else "pwsh"
        # Without the explicit module check a missing PSScriptAnalyzer looked
        # like a clean run (error swallowed, exit 0) and PSParser never ran.
        ps_code = (
            _PS_UTF8
            + f"if (-not (Get-Command Invoke-ScriptAnalyzer -ErrorAction SilentlyContinue)) "
            f"{{ exit {_NO_PSSA_EXIT} }}; "
            f"$results = Invoke-ScriptAnalyzer -Path {_ps_quote(path)} -Severity Error,Warning 2>$null; "
            f"$results | ForEach-Object {{ "
            f"\"$($_.ScriptName):$($_.Line):$($_.Severity):$($_.RuleName):$($_.Message)\" }}"
        )
        try:
            proc = _run([shell_exe, "-NoProfile", "-NonInteractive", "-Command", ps_code])
            if proc.returncode == _NO_PSSA_EXIT:
                return None
            if proc.returncode not in (0, 1):
                return None
            lines = [l for l in (proc.stdout or "").splitlines() if l.strip()]
            if not lines and proc.returncode != 0:
                return None
            return self._format_result(path, lines, "PSScriptAnalyzer")
        except (OSError, subprocess.TimeoutExpired):
            return None

    def _try_psparser(self, path: Path, is_windows: bool) -> Dict[str, Any]:
        shell_exe = "powershell.exe" if is_windows else "pwsh"
        quoted = _ps_quote(path)
        ps_code = (
            _PS_UTF8
            + f"$errors = $null; "
            f"[void][System.Management.Automation.PSParser]::Tokenize("
            f"(Get-Content -Raw {quoted}), [ref]$errors); "
            f"$errors | ForEach-Object {{ "
            f"{quoted} + ':' + $_.Token.StartLine + ':Error:Syntax:' + $_.Message }}"
        )
        try:
            proc = _run([shell_exe, "-NoProfile", "-NonInteractive", "-Command", ps_code])
            lines = [l for l in (proc.stdout or "").splitlines() if l.strip()]
            return self._format_result(path, lines, "PSParser")
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"success": False, "error": f"No PowerShell linter available: {e}"}

    # ------------------------------------------------------------------ #
    # Bash / sh linting
    # ------------------------------------------------------------------ #

    def _lint_sh(self, path: Path, is_windows: bool) -> Dict[str, Any]:
        # Prefer shellcheck — richer output
        sc_result = self._try_shellcheck(path)
        if sc_result is not None:
            return sc_result
        # Fallback: bash -n (syntax only)
        return self._try_bash_n(path)

    def _try_shellcheck(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            proc = _run(["shellcheck", "-f", "json", str(path)])
            # 0 = clean, 1 = findings; anything else is a shellcheck failure
            # (unreadable file, bad option) — don't report that as "clean".
            if proc.returncode not in (0, 1) and not (proc.stdout or "").strip():
                return None
            data = json.loads(proc.stdout or "[]")
            if not isinstance(data, list):
                return None
            lines = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                severity = str(item.get("level", "warning")).capitalize()
                rule = item.get("code", "")
                msg = item.get("message", "")
                line_no = item.get("line", 0)
                lines.append(f"{path}:{line_no}:{severity}:SC{rule}:{msg}")
            return self._format_result(path, lines, "shellcheck")
        except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
            return None

    def _try_bash_n(self, path: Path) -> Dict[str, Any]:
        try:
            proc = _run(["bash", "-n", str(path)])
            lines = [l for l in (proc.stderr or "").splitlines() if l.strip()]
            return self._format_result(path, lines, "bash -n")
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"success": False, "error": f"No shell linter available: {e}"}

    # ------------------------------------------------------------------ #
    # Format helper
    # ------------------------------------------------------------------ #

    def _format_result(self, path: Path, findings: list, linter: str) -> Dict[str, Any]:
        capped = findings[:_MAX_FINDINGS]
        if not capped:
            return {
                "success": True,
                "clean": True,
                "linter": linter,
                "message": f"No issues found in {path.name}",
                "findings": [],
            }
        return {
            "success": True,
            "clean": False,
            "linter": linter,
            "findings_count": len(findings),
            "findings": capped,
            "message": (
                f"{len(findings)} issue(s) found. Fix all, then run shell_lint again "
                f"to confirm clean before shell_exec."
            ),
        }
