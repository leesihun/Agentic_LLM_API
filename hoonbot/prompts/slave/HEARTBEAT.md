# Slave Heartbeat

You are running a scheduled local-readiness check on a slave node. Your
output goes to the master's heartbeat aggregator (or the local log if the
master is unreachable).

# Priority Order

Walk this list and stop at the first real finding:

1. **Local LLM API** — reachable on the configured local URL, returns a
   healthy `/health` response.
2. **Model credentials** — `data/.llm_key` and `data/.llm_model` exist
   and are non-empty.
3. **Cluster master reachability** — `CLUSTER_MASTER_API_URL` reachable,
   `/api/cluster/heartbeat` accepts this node's payload.
4. **Local runtime risk** — disk space below threshold, llama.cpp slot
   exhaustion, log file unbounded growth, memory.md over its soft cap
   (~8,000 chars), GPU memory unavailable on a GPU-tagged node.
5. **Local task work** — leased task in flight that has stalled past the
   expected duration.

# Reporting Rules

- If everything is healthy, respond **exactly** with `HEARTBEAT_OK`.
- For unreachable local LLM API, report the configured URL.
- For missing credentials, report which credential file is missing — the
  filename, not the contents.
- For master unreachability, report the configured master URL and the
  failure mode (timeout, connection refused, 401, etc.).
- For risk signals, report the concrete value (disk free GB, slot count,
  log size).
- Never include the actual credential value, cluster token, or any
  `.env` contents.

# Memory and Status

- Live readiness values (disk, GPU, master reachability, last heartbeat)
  go to the `status_file` (data/status.md), overwritten each tick — never
  into memory.md.
- memory.md holds durable facts only and must stay under ~8,000 chars (hard
  cap 12,000). Prune or merge stale entries with `file_edit` /
  `apply_patch`; never paste logs or command output into it.

# Sample Outputs

Healthy:
```
HEARTBEAT_OK
```

Local LLM API down:
```
Local LLM API unreachable at http://127.0.0.1:10002 (connection refused).
Slave cannot lease tasks until LLM API recovers.
```

Missing credentials:
```
Missing credential file: data/.llm_model. Run scripts/setup_credentials.py
to recreate.
```

Master unreachable:
```
Cluster master http://192.168.0.10:10002 unreachable (timeout 5s).
Last successful heartbeat: 2m18s ago.
```
