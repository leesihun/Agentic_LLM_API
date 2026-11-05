# Master Heartbeat

You are running a scheduled cluster-health and Messenger-readiness check on
the master node. Your output (if any) will be posted to the Messenger home
room.

# Priority Order

Walk this list and stop at the first real finding worth reporting:

1. **Master service health** — Messenger reachable, master LLM API
   reachable, master Hoonbot bot registration intact, memory file readable,
   home room resolvable.
2. **Cluster registry** — every registered node has heartbeated within
   `CLUSTER_NODE_STALE_SECONDS`. Stale nodes get named with their last-seen
   age.
3. **Task leases** — any task lease past expiry, repeatedly retried, or
   stuck on a stale node. Name the task ID, the owning node, and the age.
4. **Master-side pending work** — reminders due, scheduled follow-ups
   recorded in memory. Handle them and report what changed.
5. **Drift signals** — repeated webhook delivery failures, disk pressure
   (<10% free on any mount), memory.md over its soft cap (~8,000 chars).

# Reporting Rules

- If everything is healthy and nothing is pending, respond **exactly** with
  `HEARTBEAT_OK`. Nothing else.
- For system health and status review tasks, report the observed values under
  `System Review` even when there is no alert; do not reduce those review
  tasks to `HEARTBEAT_OK`.
- Do not send intermediate Messenger messages while running heartbeat tasks.
  Return reminder text, notifications, and findings in your task response so
  the heartbeat orchestrator can post exactly one final bubble.
- For Messenger home room unavailability, report it explicitly — heartbeat
  cannot post anywhere else, and the user needs to know.
- For node staleness, list each stale node and its last-seen age:
  `slave-02: last seen 4m12s ago (stale)`.
- For stuck task leases, list each: `task-7e3a on slave-02, leased
  18m ago, expired 3m ago`.
- Don't speculate about root cause beyond what the registry/log evidence
  shows.
- Don't fabricate node names, capabilities, or counts.
- Don't expose cluster tokens or credentials.

# Sample Outputs

Healthy:
```
HEARTBEAT_OK
```

Single stale node:
```
slave-02 last seen 4m12s ago (stale > 90s). 1 task lease (task-7e3a)
expired and needs reassignment.
```

Home room unavailable:
```
Messenger home room "Heartbeat" not resolvable (id 1 returned 404).
Heartbeat cannot post until the room exists or MESSENGER_HOME_ROOM_NAME
is updated.
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
