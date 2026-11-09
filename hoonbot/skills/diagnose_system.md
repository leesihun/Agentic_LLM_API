# Skill: Diagnose System

Run a concise host health check.

## Use When

diagnose system, health check, server status, resource usage

## Tool

`shell_exec`

## Run

Run these in order and stop on the first failure:

1. `lscpu`
2. `uptime`
3. `free -h --si`
4. `df -h`
5. `ps aux --sort=-%mem | head -11`
6. `nvidia-smi --query-gpu=name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits`

## Rules

- No fallback commands unless the user asks.
- Warn on CPU load above 95%, RAM above 70%, swap above 50%, disk above 80%, GPU temp above 80C, or VRAM above 95%.
- Do not expose secrets from environment or config files.
- This checks the host only. To see what the bot itself is doing right now (running turn, queue, tools, processes), tell the user to send `@status` in the room; it answers immediately without interrupting anything.

## Reply

`System Diagnostics: CPU=<...>; RAM=<...>; Swap=<...>; GPU=<...>; Disks=<...>; Uptime=<...>; TopMem=<...>; Warnings=<list|none>.`
