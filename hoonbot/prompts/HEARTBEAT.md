# Hoonbot Heartbeat

You are running a scheduled health and follow-up check for Hoonbot. No human
is waiting live, but anything you report will be posted to the Messenger
home room.

# Priority Order

Walk this list in order and stop at the first concrete finding worth
reporting. Bundle multiple findings only if they are all real.

1. **Critical service health** — Messenger reachable, LLM API reachable,
   model credentials present, memory file readable, configured skill paths
   present. If any are unhealthy, report the concrete failure with the
   configured path or URL.
2. **Pending work** — reminders due, scheduled tasks, proactive follow-ups
   recorded in memory. Handle each one and report what changed (sent reply,
   set new reminder, marked done).
3. **Cluster state** (if cluster mode is active) — known nodes healthy?
   Any task lease stuck or expired? Report node names and ages, not
   summaries.
4. **Drift / risk signals** — disk pressure (<10% free on any mount),
   memory.md over its soft cap (~8,000 chars), repeated webhook failures,
   broken bot registration.

# Reporting Rules

- If everything in the priority list is healthy and there is nothing to
  follow up on, respond **exactly** with `HEARTBEAT_OK`. Nothing else.
- For system health and status review tasks, report the observed values under
  `System Review` even when there is no alert; do not reduce those review
  tasks to `HEARTBEAT_OK`.
- Do not send intermediate Messenger messages while running heartbeat tasks.
  Return reminder text, notifications, and findings in your task response so
  the heartbeat orchestrator can post exactly one final bubble.
- For any failure, name the concrete cause: which service, which URL or
  path, which credential file, which node, which task ID.
- For pending work you handled, name what you did and the artifact (message
  ID, file path, reminder ID). Do not list pending work without acting on it
  first.
- Keep the report short enough for a chat room — bullets are fine, full
  paragraphs are usually too long.
- Do not invent node names, capability lists, or task statuses. If
  cluster state is unavailable, say so.
- Do not expose secrets, cluster tokens, or credential file contents.

# Sample Outputs

Healthy:
```
HEARTBEAT_OK
```

Service down:
```
LLM API unreachable at http://192.168.0.10:10002 (last 3 attempts timed out).
Hoonbot will keep retrying.
```

Pending work handled:
```
Reminder "review PR #42" was due 14:00. Sent reminder to Heartbeat room
(message id 18421). Removed from memory.
```

Cluster issue:
```
slave-02 last seen 6m ago (stale > 90s). Task lease task-7e3a held by
slave-02 expired; needs reassignment.
```

---

# Heartbeat Checklist

This file is read every heartbeat interval.

Your job is to find things to do yourself and actually do them proactively:
run the check, observe its output, and report what was done. If a job came
due between the previous heartbeat and this one, do it now.

---

## Memory

- Read memory.md for open reminders, follow-ups, and recurring jobs that are
  due. The heartbeat interval is limited — queue anything that cannot finish
  this tick as a dated reminder.
- Live status (GPU/RAM/disk numbers, training progress, service health,
  last-run times) goes to the `status_file` (data/status.md): OVERWRITE it
  each tick with the fresh values. Never append live status to memory.md.
- memory.md holds durable facts only and must stay under ~8,000 chars (hard
  cap 12,000). If it is over, prune or merge stale entries with `file_edit`
  / `apply_patch`. Never paste logs, command output, or reports into it.

## Reminders

- Check if any reminders are due. If so, queue the notification text for the
  final heartbeat report.
- If a reminder has fired, remove it from the reminder list in memory.md.

## System Health

- Run `nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv` for live GPU stats (util, memory, temp).
- Run `free -m` for live RAM.
- Run `df -h` for disk usage.
- If GPU memory is critically low (<5% free on any GPU), alert immediately.
- If any process has crashed or stalled, alert immediately.
- Report the observed values in the System Review section of the report
  (complete numbers, compact table or bullets) and in data/status.md.

## Priority Order

1. Crash/alert detection -> report immediately
2. System health (GPU, CPU, disk)
3. Training status (live log tail)
4. Reminders
5. Status update: overwrite data/status.md with this tick's live values
6. Memory hygiene: memory.md under ~8,000 chars, stale entries pruned
