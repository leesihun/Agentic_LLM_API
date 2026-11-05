"""
Tool schemas for the LLM agent (OpenAI function-calling format).

Consumed by backend/agent.py at module load time to build the tool list
sent to vLLM. Parameter names must exactly match what _dispatch_tool()
passes to each tool's execute/read/write/navigate/edit/search method.

Descriptions are built once at import from config (deterministic f-strings),
so the serialized tool list stays byte-stable for vLLM prefix caching.
"""

try:
    import config as _config
except Exception:  # pragma: no cover - schemas must import even without config
    _config = None


def _cfg_int(name: str, default: int) -> int:
    try:
        return int(getattr(_config, name, default))
    except (TypeError, ValueError):
        return default


_AIRGAPPED = bool(getattr(_config, "AIRGAPPED", True))
_SUBAGENT_MAX_PER_TURN = _cfg_int("SUBAGENT_MAX_PER_TURN", 16)
_SHELL_DEFAULT_TIMEOUT = _cfg_int("SHELL_DEFAULT_TIMEOUT_SECONDS", 300)
_SHELL_MAX_TIMEOUT = _cfg_int("SHELL_EXEC_HARD_CAP_SECONDS", 3600)
_CODE_DEFAULT_TIMEOUT = _cfg_int("CODE_EXEC_DEFAULT_TIMEOUT_SECONDS", 300)
_CODE_MAX_TIMEOUT = _cfg_int("CODE_EXEC_HARD_CAP_SECONDS", 1800)

_OFFLINE_NOTE = (
    " This deployment is airgapped: there is no internet access (no pip/npm "
    "install from an index, no curl/wget to external hosts)."
    if _AIRGAPPED else ""
)
_EXPLORE_TOOL_LIST = (
    "file_reader, grep, file_navigator, rag, shell_lint"
    if _AIRGAPPED else
    "file_reader, grep, file_navigator, rag, shell_lint, websearch"
)

TOOL_SCHEMAS: dict = {

    # ================================================================
    # SHELL EXECUTION
    # ================================================================

    "shell_exec": {
        "name": "shell_exec",
        "description": (
            "Execute a shell command. State does not persist between calls — pass "
            "working_directory when cwd matters. Check the ## ENVIRONMENT block for "
            "the active shell (on Linux servers it is bash: `&&`, `$VAR`; PowerShell "
            "on Windows uses `;` and `$env:VAR`). For long-running servers/watchers "
            "or jobs that may outlive the timeout use process_monitor instead. For "
            "reading/searching files use file_reader, grep, and file_navigator — not "
            f"cat/grep/find.{_OFFLINE_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Maximum seconds to wait (default: {_SHELL_DEFAULT_TIMEOUT}; omit or 0 = "
                        f"the default, NOT unlimited; values above {_SHELL_MAX_TIMEOUT} are capped). "
                        "The process tree is killed on timeout."
                    ),
                },
                "working_directory": {
                    "type": "string",
                    "description": (
                        "Working directory for the command. Relative paths resolve against the "
                        "session workspace (the default agent workspace when none is set). "
                        "Defaults to the session workspace."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Clear description of what this command does (5-10 words for simple commands, "
                        "more detail for piped/complex operations). Example: 'Install numpy via pip' or "
                        "'Find and delete all .tmp files recursively'."
                    ),
                },
            },
            "required": ["command"],
        },
    },

    # ================================================================
    # FILE OPERATIONS
    # ================================================================

    "file_reader": {
        "name": "file_reader",
        "description": (
            "Read a file from the local filesystem. Absolute paths read directly; "
            "relative paths resolve against the session workspace (server CWD by default), "
            "then uploads. Use offset/limit for large files. Prefer this over `cat`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the file. Absolute paths (e.g. /home/user/file.py) are read directly. "
                        "Relative paths resolve against the session workspace, then user uploads, then server CWD."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based, default: 1).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 2000).",
                },
            },
            "required": ["path"],
        },
    },

    "file_edit": {
        "name": "file_edit",
        "description": (
            "Exact string replacement in a file. Read the file first so old_string "
            "matches byte-for-byte (preserve indentation). For multi-line edits, "
            ".ps1 or .sh files, use apply_patch instead. Set replace_all=true to "
            "rename a symbol across the file.\n\n"
            "If exact match fails, the harness retries with progressively looser "
            "whitespace-normalised matching (line-trimmed -> indent-flexible -> "
            "whitespace-collapsed). The result's `strategy` field reports which "
            "match succeeded. If all strategies fail or produce ambiguous matches, "
            "re-read the file with file_reader (offset/limit to scope) and retry "
            "with a smaller, more precisely-anchored old_string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or session-relative path to the file to modify.",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "The exact text to replace (must match character-for-character including "
                        "whitespace and indentation). Must be unique in the file unless replace_all=true."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with. Must be different from old_string.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences of old_string (default false).",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },

    "apply_patch": {
        "name": "apply_patch",
        "description": (
            "Apply a V4A context-anchored patch to one or more files. Preferred for "
            "all multi-line edits and shell scripts (handles CRLF/LF and whitespace).\n\n"
            "Format:\n"
            "    *** Begin Patch\n"
            "    *** Update File: path/to/file\n"
            "    @@ nearby unique line\n"
            "     unchanged context\n"
            "    -removed\n"
            "    +added\n"
            "    *** End Patch\n\n"
            "Directives: *** Add File / *** Delete File / *** Update File / *** Move to. "
            "Multiple files per envelope. On failure the error names the unmatched "
            "context line — re-read with file_reader, then reissue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "V4A patch text starting with '*** Begin Patch' and ending with '*** End Patch'.",
                },
                "persist": {
                    "type": "boolean",
                    "description": (
                        "If true/omitted (default), files newly added by this patch are kept. "
                        "Pass false only for throwaway scratch, which flags added files as "
                        "temporary so they get cleaned up. Updates/deletes/moves are unaffected."
                    ),
                },
            },
            "required": ["patch"],
        },
    },

    "file_writer": {
        "name": "file_writer",
        "description": (
            "Write a file (overwrites). Use ONLY for new files or full rewrites — "
            "for any modification of an existing file, use file_edit or apply_patch. "
            "New files are KEPT by default; pass persist=false only for genuine "
            "throwaway scratch (prefer code_exec for that). "
            "Don't create docs/README files unless the user asked."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path, or relative path resolved against the session "
                        "workspace (the server CWD when no workspace is set)."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write to the file.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["write", "append"],
                    "description": "'write' overwrites the file (default); 'append' adds to the end.",
                },
                "persist": {
                    "type": "boolean",
                    "description": (
                        "If true/omitted (default), a newly-created file is kept. "
                        "Pass false only for throwaway scratch, which is treated as "
                        "temporary and gets cleaned up. Editing an existing file is unaffected."
                    ),
                },
            },
            "required": ["path", "content"],
        },
    },

    "file_navigator": {
        "name": "file_navigator",
        "description": (
            "List or find files by name/glob. Operations: list (directory), "
            "search (glob like '**/*.py'), tree (recursive). For searching file "
            "*contents* use grep instead. If you know the path, call file_reader."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "search", "tree"],
                    "description": (
                        "'list': list files in a directory. "
                        "'search': find files matching a glob pattern. "
                        "'tree': show directory tree structure."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory path. Absolute paths are used directly. "
                        "Relative paths resolve against the session workspace "
                        "(the server CWD when no workspace is set)."
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for search operation (e.g. '*.csv', '**/*.py').",
                },
            },
            "required": ["operation"],
        },
    },

    # ================================================================
    # CONTENT SEARCH
    # ================================================================

    "grep": {
        "name": "grep",
        "description": (
            "ripgrep content search. Supports full regex, glob filters, and file-type "
            "filters. Output modes: 'files_with_matches' (default, just paths), "
            "'content' (matching lines, supports -A/-B/-C), 'count'. For multi-round "
            "open-ended exploration of several areas, fan out parallel `agent` calls "
            "with subagent_type='explore'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (rg PATH). Defaults to current working directory.",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.js', '*.{ts,tsx}') — maps to rg --glob.",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "Output mode: 'content' shows matching lines (supports -A/-B/-C context, "
                        "-n line numbers), 'files_with_matches' shows file paths (default), "
                        "'count' shows match counts."
                    ),
                },
                "-A": {
                    "type": "integer",
                    "description": "Number of lines to show after each match (rg -A). Requires output_mode: 'content'.",
                },
                "-B": {
                    "type": "integer",
                    "description": "Number of lines to show before each match (rg -B). Requires output_mode: 'content'.",
                },
                "-C": {
                    "type": "integer",
                    "description": "Alias for context — lines before AND after each match.",
                },
                "context": {
                    "type": "integer",
                    "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: 'content'.",
                },
                "-n": {
                    "type": "boolean",
                    "description": "Show line numbers in output (rg -n). Requires output_mode: 'content'. Defaults to true.",
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case insensitive search (rg -i).",
                },
                "type": {
                    "type": "string",
                    "description": "File type to search (rg --type). Common types: js, py, rust, go, java, ts.",
                },
                "head_limit": {
                    "type": "integer",
                    "description": (
                        "Limit output to first N lines/entries, equivalent to '| head -N'. "
                        "Works across all output modes. Defaults to 250. Pass 0 for unlimited."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N lines/entries before applying head_limit. Defaults to 0.",
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline mode where . matches newlines and patterns can span lines. Default: false.",
                },
            },
            "required": ["pattern"],
        },
    },

    # ================================================================
    # CODE EXECUTION
    # ================================================================

    "code_exec": {
        "name": "code_exec",
        "description": (
            "Run a complete Python script. Returns stdout/stderr/returncode. "
            "Use for data processing, calculations, library calls, plotting, ML. "
            "Prefer this over shell_exec for Python work. Check returncode before "
            "declaring success. The script runs with cwd = the session workspace "
            "(also exported as WORKSPACE_DIR), so relative paths read/write project "
            "files; the script file itself lives in a temp dir. Write outputs you "
            "want to keep to explicit paths. For jobs longer than "
            f"{_CODE_MAX_TIMEOUT}s use process_monitor.{_OFFLINE_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python source code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Maximum seconds for execution (default: {_CODE_DEFAULT_TIMEOUT}; omit or 0 = "
                        f"the default, NOT unlimited; values above {_CODE_MAX_TIMEOUT} are capped)."
                    ),
                },
            },
            "required": ["code"],
        },
    },

    # ================================================================
    # TASK TRACKING
    # ================================================================

    "todo_write": {
        "name": "todo_write",
        "description": (
            "Track progress for multi-step tasks (3+ distinct steps). Each call "
            "replaces the full list. Exactly one task in 'in_progress' at a time. "
            "Mark 'completed' only when actually done. Skip for trivial or "
            "single-step requests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete updated todo list (replaces previous list).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Stable identifier (e.g. 'task_1', 'fix_auth').",
                            },
                            "content": {
                                "type": "string",
                                "description": "Imperative form: what needs to be done (e.g. 'Run tests').",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current status. Only ONE item may be in_progress at a time.",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Task priority.",
                            },
                        },
                        "required": ["id", "content", "status", "priority"],
                    },
                },
            },
            "required": ["todos"],
        },
    },

    # ================================================================
    # SUBAGENT SPAWNING
    # ================================================================

    "agent": {
        "name": "agent",
        "description": (
            "Spawn a subagent with a fresh context that works a subtask to completion "
            "and returns its final report. PARALLEL FAN-OUT: put several `agent` calls "
            "in ONE assistant message and they run concurrently (up to "
            f"{_SUBAGENT_MAX_PER_TURN} per turn; extra calls are queued, never "
            "dropped). Whenever a task splits into independent parts — different "
            "files, modules, questions, datasets or hypotheses — launch one subagent "
            "per part in the same message instead of doing the parts one by one.\n\n"
            f"subagent_type='explore' (default): read-only tools ({_EXPLORE_TOOL_LIST}); "
            "explore calls may start while your message is still streaming. "
            "'general': full toolset (no nested agent) for delegated work that edits "
            "files or runs code; general calls start after your message is complete. "
            "Parallel general subagents share the workspace — give each one disjoint "
            "files.\n\n"
            "Each subagent sees ONLY its prompt (not this conversation), so make every "
            "prompt self-contained: goal, relevant paths/context, constraints, and the "
            "exact output you want back. Skip subagents for a lookup a single grep or "
            "file_reader answers. The result holds the subagent's final report plus a "
            "transcript_path for the full log; the user does not see it — relay what "
            "matters yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Self-contained task for the subagent — brief it like a smart colleague who "
                        "just walked in. Include: the goal, which files/areas to look in or change, "
                        "constraints, and what form the final report should take."
                    ),
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["explore", "general"],
                    "description": (
                        "'explore' for read-only research (default, may start early). "
                        "'general' for full-toolset work that edits files or runs commands."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what this agent will do (3-5 words). Used for logging.",
                },
            },
            "required": ["prompt"],
        },
    },

    # ================================================================
    # WEB SEARCH
    # ================================================================

    "websearch": {
        "name": "websearch",
        "description": (
            "Search the web for current information, news, documentation, or facts not in your "
            "training data. Returns results with titles, URLs, and content snippets.\n\n"
            "Use for: library docs, API references, error messages, recent events, version changelogs.\n"
            "Do not use for: information already in your training data or available in uploaded docs "
            "(use rag instead)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific for best results.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    },

    # ================================================================
    # KNOWLEDGE BASE
    # ================================================================

    "rag": {
        "name": "rag",
        "description": (
            "Retrieve relevant information from user-uploaded document collections using semantic "
            "search. Use for searching uploaded PDFs, docs, or datasets.\n\n"
            "You must specify which collection to search. Use only existing collection_name values "
            "listed in context. Results are document chunks ranked by relevance."
            + ("" if _AIRGAPPED else
               "\n\nUse rag for uploaded documents; use websearch for live web results.")
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Name of the document collection to search. Must be an existing collection.",
                },
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document chunks.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of document chunks to return (default: 10).",
                },
            },
            "required": ["collection_name", "query"],
        },
    },

    # ================================================================
    # MEMORY
    # ================================================================

    "memo": {
        "name": "memo",
        "description": (
            "Read or write persistent memory that survives across sessions.\n\n"
            "Use memo for facts that should persist across sessions: important results, decisions, "
            "file paths, project-specific conventions, user preferences.\n\n"
            "For current-session task tracking, use todo_write instead.\n\n"
            "Memory is per-user and automatically shown at the start of every session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["write", "read", "list", "delete"],
                    "description": (
                        "'write': save a value under a key. "
                        "'read': retrieve a value by key. "
                        "'list': show all saved entries. "
                        "'delete': remove an entry by key."
                    ),
                },
                "key": {
                    "type": "string",
                    "description": "Memory key name (e.g. 'best_lr', 'dataset_path', 'project_status').",
                },
                "value": {
                    "type": "string",
                    "description": "Value to store. Required for 'write' operation.",
                },
            },
            "required": ["operation"],
        },
    },

    # ================================================================
    # PROCESS MANAGEMENT
    # ================================================================

    "process_monitor": {
        "name": "process_monitor",
        "description": (
            "Background process lifecycle: start, status, read_output, kill, list. "
            "Use instead of shell_exec for servers/watchers, or to observe long "
            "incremental output. Each process gets a handle like 'proc_1'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["start", "status", "read_output", "kill", "list"],
                    "description": (
                        "'start': launch a background command, returns a handle. "
                        "'status': check if running/exited (by handle). "
                        "'read_output': get stdout/stderr since last read or from an offset (by handle). "
                        "'kill': terminate a process (by handle). "
                        "'list': show all tracked processes for this session."
                    ),
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to run (required for 'start').",
                },
                "handle": {
                    "type": "string",
                    "description": "Process handle (e.g. 'proc_1'). Required for 'status', 'read_output', 'kill'.",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory (only for 'start'). Defaults to the session workspace (server CWD when unset).",
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Line offset for 'read_output'. Use next_offset from a previous read to get new lines. "
                        "If omitted, returns the last 200 lines. "
                        "In stream=both, passing either stream's next_offset continues each stream "
                        "from its own last read (stdout and stderr count lines separately); any other "
                        "value is used as an explicit line position for both streams (past a stream's end, "
                        "that stream resumes from its own last read, or its tail if never read)."
                    ),
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to return for 'read_output' (default: 200).",
                },
                "stream": {
                    "type": "string",
                    "enum": ["stdout", "stderr", "both"],
                    "description": "Which output stream to read (default: 'both'). Only for 'read_output'.",
                },
            },
            "required": ["operation"],
        },
    },

    # ================================================================
    # SHELL LINTING
    # ================================================================

    "shell_lint": {
        "name": "shell_lint",
        "description": (
            "Static analysis on a shell script (.ps1/.sh/.bash) before running it. "
            "Uses PSScriptAnalyzer / shellcheck where available, falls back to "
            "syntax-only checks. Returns file:line:severity:rule:message. Worth "
            "running after non-trivial edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or repo-relative path to the .ps1, .sh, or .bash file to lint.",
                },
            },
            "required": ["path"],
        },
    },

}

# Airgapped deployments have no internet: don't advertise websearch at all
# (config.AVAILABLE_TOOLS drops it too; the tool itself also fails fast).
if _AIRGAPPED:
    TOOL_SCHEMAS.pop("websearch", None)

# Per-tool metadata: safety flags, UI activity descriptions, user-facing names.
# Metadata stays for every tool (incl. websearch) so lookups never KeyError.
# `agent` is concurrency-safe (parallel fan-out) but not read-only; the loop
# early-starts only subagent_type='explore' calls.
# Mirrors openclaude's Tool.ts metadata fields.
TOOL_METADATA: dict = {
    "shell_exec":      {"is_read_only": False, "is_destructive": True,  "is_concurrency_safe": False, "activity": "Running shell command",   "user_name": "Shell"},
    "file_reader":     {"is_read_only": True,  "is_destructive": False, "is_concurrency_safe": True,  "activity": "Reading file",            "user_name": "File Reader"},
    "file_edit":       {"is_read_only": False, "is_destructive": False, "is_concurrency_safe": False, "activity": "Editing file",            "user_name": "File Editor"},
    "apply_patch":     {"is_read_only": False, "is_destructive": False, "is_concurrency_safe": False, "activity": "Applying V4A patch",      "user_name": "Apply Patch"},
    "shell_lint":      {"is_read_only": True,  "is_destructive": False, "is_concurrency_safe": True,  "activity": "Linting shell script",    "user_name": "Shell Lint"},
    "file_writer":     {"is_read_only": False, "is_destructive": True,  "is_concurrency_safe": False, "activity": "Writing file",            "user_name": "File Writer"},
    "file_navigator":  {"is_read_only": True,  "is_destructive": False, "is_concurrency_safe": True,  "activity": "Navigating files",        "user_name": "File Navigator"},
    "grep":            {"is_read_only": True,  "is_destructive": False, "is_concurrency_safe": True,  "activity": "Searching files",         "user_name": "Grep"},
    "code_exec":       {"is_read_only": False, "is_destructive": False, "is_concurrency_safe": False, "activity": "Executing code",          "user_name": "Code Exec"},
    "websearch":       {"is_read_only": True,  "is_destructive": False, "is_concurrency_safe": True,  "activity": "Searching the web",       "user_name": "Web Search"},
    "rag":             {"is_read_only": True,  "is_destructive": False, "is_concurrency_safe": True,  "activity": "Searching documents",     "user_name": "RAG"},
    "memo":            {"is_read_only": False, "is_destructive": False, "is_concurrency_safe": False, "activity": "Accessing memory",        "user_name": "Memo"},
    "todo_write":      {"is_read_only": False, "is_destructive": False, "is_concurrency_safe": False, "activity": "Updating task list",      "user_name": "Todo"},
    "process_monitor": {"is_read_only": False, "is_destructive": True,  "is_concurrency_safe": False, "activity": "Managing processes",      "user_name": "Process Monitor"},
    "agent":           {"is_read_only": False, "is_destructive": False, "is_concurrency_safe": True,  "activity": "Running subagent",        "user_name": "Agent"},
}
