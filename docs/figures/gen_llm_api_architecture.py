"""Generate the LLM-API architecture figure as a standalone SVG."""
import sys
from html import escape

W, H = 1500, 1060
SANS = "Arial, Helvetica, sans-serif"
MONO = "Consolas, 'DejaVu Sans Mono', monospace"

INK = "#1b1b1b"
SUB = "#4d4d4d"
LINE = "#3a3a3a"
C_THOUGHT = "#2f5d8a"
C_ACTION = "#2f7a45"
C_OBS = "#7a4f8f"
C_LLM = "#b5651d"

out = []


def add(s):
    out.append(s)


def rect(x, y, w, h, fill="#fff", stroke=LINE, sw=1.2, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=14, weight="normal", fill=INK, anchor="start", family=SANS,
         style="normal", lh=None):
    lines = s.split("\n") if isinstance(s, str) else s
    lh = lh or size * 1.25
    for i, ln in enumerate(lines):
        add(f'<text x="{x}" y="{y + i * lh:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
            f'text-anchor="{anchor}">{escape(ln)}</text>')


def path(d, stroke=LINE, sw=1.5, dash=None, end=True, start=False, color_marker="k"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    me = f' marker-end="url(#ah-{color_marker})"' if end else ""
    ms = f' marker-start="url(#as-{color_marker})"' if start else ""
    add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{da}{me}{ms}/>')


def diamond(cx, cy, hw, hh, label, size=13.5):
    pts = f"{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}"
    add(f'<polygon points="{pts}" fill="#fff" stroke="{LINE}" stroke-width="1.3"/>')
    lines = label.split("\n")
    y0 = cy - (len(lines) - 1) * size * 0.6 + size * 0.35
    text(cx, y0, lines, size=size, anchor="middle", weight="bold", lh=size * 1.2)


def pill(x, y, w, h, label, size=13.5, fill="#fff", stroke=LINE, weight="bold", family=SANS,
         color=INK):
    rect(x, y, w, h, fill=fill, stroke=stroke, rx=h / 2, sw=1.3)
    lines = label.split("\n")
    y0 = y + h / 2 - (len(lines) - 1) * size * 0.6 + size * 0.35
    text(x + w / 2, y0, lines, size=size, anchor="middle", weight=weight, family=family,
         fill=color, lh=size * 1.2)


def chip(x, y, w, label, safe, h=25, size=13.5):
    rect(x, y, w, h, fill="#fff", stroke="#8a8a8a", rx=4, sw=1)
    mx, my = x + 11, y + h / 2
    if safe:
        add(f'<circle cx="{mx}" cy="{my}" r="4.2" fill="{C_ACTION}"/>')
    else:
        add(f'<rect x="{mx - 4}" y="{my - 4}" width="8" height="8" fill="{LINE}"/>')
    text(x + 22, y + h / 2 + size * 0.35, label, size=size, family=MONO)


def tag(x, y, label, color, size=12):
    w = len(label) * size * 0.68 + 14
    rect(x, y, w, size + 8, fill=color, stroke=color, rx=3, sw=1)
    text(x + w / 2, y + size + 2.5, label, size=size, weight="bold", fill="#fff",
         anchor="middle")
    return w


def step_box(x, y, w, h, title, body, color=None, tagname=None, fill="#fff"):
    rect(x, y, w, h, fill=fill, stroke=LINE, sw=1.3)
    ty = y + 22
    if tagname:
        add(f'<rect x="{x}" y="{y}" width="5" height="{h}" fill="{color}" rx="2"/>')
        tag(x + 12, y + 8, tagname, color)
        ty = y + 46
    text(x + 12, ty, title, size=14.5, weight="bold")
    text(x + 12, ty + 20, body, size=12.5, fill=SUB, lh=16)


def section(x, y, w, h, letter, title, fill):
    rect(x, y, w, h, fill=fill, stroke="#555", sw=1.4, rx=8)
    add(f'<circle cx="{x + 20}" cy="{y + 20}" r="11" fill="{INK}"/>')
    text(x + 20, y + 25, letter, size=13, weight="bold", fill="#fff", anchor="middle")
    text(x + 38, y + 26, title, size=17, weight="bold")


def cylinder(cx, top, w, h, fill="#fff"):
    rx, ry = w / 2, 7
    add(f'<path d="M{cx - rx},{top} v{h} a{rx},{ry} 0 0 0 {w},0 v{-h}" fill="{fill}" '
        f'stroke="{LINE}" stroke-width="1.2"/>')
    add(f'<ellipse cx="{cx}" cy="{top}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{LINE}" '
        f'stroke-width="1.2"/>')


# ----------------------------------------------------------------------------- defs
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">')
add('<defs>')
for key, col in (("k", LINE), ("o", C_LLM), ("g", C_ACTION)):
    add(f'<marker id="ah-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    add(f'<marker id="as-{key}" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M10,0 L0,5 L10,10 z" fill="{col}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ----------------------------------------------------------------------------- (A) Interfaces
section(15, 15, 200, 745, "A", "Interfaces", "#f6f6f6")
ifaces = [
    ("Chat Messenger", "web & desktop client\nreal-time, multi-user"),
    ("Autonomous Bot", "webhook-driven\nlong-term memory\nscheduled heartbeat"),
    ("OpenAI-compatible\nClients", "REST / SSE\nSDKs, scripts"),
    ("Cluster Worker\nNodes", "distributed task\ndelegation"),
]
iy = 58
centers = []
for title, body in ifaces:
    nt = title.count("\n") + 1
    nb = body.count("\n") + 1
    h = 22 + nt * 18 + nb * 16 + 6
    rect(28, iy, 174, h, fill="#fff", stroke=LINE, sw=1.3)
    text(40, iy + 22, title, size=14.5, weight="bold", lh=18)
    text(40, iy + 22 + nt * 18 + 2, body, size=12.5, fill=SUB, lh=16)
    centers.append(iy + h / 2)
    iy += h + 16

# bus to gateway
BUS_X = 224
GW_Y = 88
for cy in centers:
    path(f"M202,{cy} H{BUS_X}", end=False, start=True)
path(f"M{BUS_X},{max(centers)} V{GW_Y} H247", end=True)

# notation legend
ly = 590
text(28, ly, "Notation", size=13.5, weight="bold")
rect(28, ly + 12, 34, 20, rx=3)
text(70, ly + 27, "process step", size=12.5, fill=SUB)
add(f'<polygon points="45,{ly + 40} 62,{ly + 51} 45,{ly + 62} 28,{ly + 51}" fill="#fff" '
    f'stroke="{LINE}" stroke-width="1.2"/>')
text(70, ly + 56, "decision", size=12.5, fill=SUB)
rect(28, ly + 70, 34, 18, rx=9)
text(70, ly + 84, "terminal", size=12.5, fill=SUB)
path(f"M28,{ly + 104} H62", sw=1.5)
text(70, ly + 108, "control flow", size=12.5, fill=SUB)
path(f"M28,{ly + 126} H62", stroke=C_LLM, sw=1.8, dash="5,3", color_marker="o")
text(70, ly + 130, "LLM inference", size=12.5, fill=SUB)
path(f"M28,{ly + 148} H62", stroke=C_ACTION, sw=1.8, color_marker="g", start=True)
text(70, ly + 152, "tool invocation", size=12.5, fill=SUB)

# ----------------------------------------------------------------------------- (B) Server
section(232, 15, 768, 745, "B", "LLM API Server", "#f7f8fa")

# gateway
rect(247, 50, 738, 78, fill="#fff", stroke=LINE, sw=1.3)
text(259, 72, "API Gateway (FastAPI)", size=14.5, weight="bold")
text(430, 72, "— OpenAI-compatible, streaming", size=12.5, fill=SUB)
gx = 259
for lab in ("/v1/chat/completions", "JWT auth", "session store", "async document ingest",
            "cluster control plane"):
    fam = MONO if lab.startswith("/") else SANS
    w = len(lab) * (7.0 if fam == MONO else 6.5) + 22
    rect(gx, 86, w, 28, fill="#f2f4f7", stroke="#9aa3ad", rx=4, sw=1)
    text(gx + w / 2, 105, lab, size=12.5, anchor="middle", family=fam)
    gx += w + 9

# agent loop frame
rect(247, 142, 738, 608, fill="#eef3f9", stroke=C_THOUGHT, sw=1.4, rx=8, dash="7,4")
text(282, 166, "ReAct Agent Loop", size=16, weight="bold", fill=C_THOUGHT)
text(431, 166, "single reasoning–acting loop, iteration t = 1 … N", size=12.5, fill=SUB)

# row 1
step_box(280, 250, 160, 120, "Context Manager",
         "micro-compaction of\nold iterations;\nauto-compact on\ncontext overflow")
step_box(470, 250, 170, 120, "Result Formatting",
         "normalize output;\nspill oversize result\nto disk (pointer kept)",
         color=C_OBS, tagname="OBSERVATION")
# action box
AX, AY, AW, AH = 672, 186, 298, 252
rect(AX, AY, AW, AH, fill="#fff", stroke=LINE, sw=1.3)
add(f'<rect x="{AX}" y="{AY}" width="5" height="{AH}" fill="{C_ACTION}" rx="2"/>')
tag(AX + 12, AY + 8, "ACTION", C_ACTION)
text(AX + 12, AY + 46, "Tool Dispatch", size=14.5, weight="bold")
text(AX + 12, AY + 65, "capability-aware segmented scheduling", size=12.5, fill=SUB)
# mini schedule diagram
sy = AY + 92
g = AX + 16
for i, lab in enumerate(("file_reader", "grep", "rag")):
    chip(g, sy + i * 31, 100, lab, True, h=24, size=12)
add(f'<path d="M{g + 104},{sy} h5 v{3 * 31 - 7} h-5" fill="none" stroke="{SUB}" '
    f'stroke-width="1.2"/>')
path(f"M{g + 112},{sy + 43} H{g + 124}", sw=1.3)
chip(g + 126, sy + 31, 88, "file_edit", False, h=24, size=12)
path(f"M{g + 216},{sy + 43} H{g + 226}", sw=1.3)
for i, lab in enumerate(("grep", "rag")):
    chip(g + 228, sy + 16 + i * 31, 50, lab, True, h=24, size=12)
text(g + 50, sy + 108, "parallel", size=12, anchor="middle", fill=C_ACTION, weight="bold")
text(g + 170, sy + 108, "serial barrier", size=12, anchor="middle", fill=INK, weight="bold")
text(g + 253, sy + 108, "parallel", size=12, anchor="middle", fill=C_ACTION, weight="bold")
text(AX + 12, AY + AH - 30, "read-only calls start mid-stream as soon\nas their arguments are parsed",
     size=12, fill=SUB, lh=15)

# row 2
step_box(280, 532, 160, 106, "Context Assembly",
         "cached system prompt\n+ workspace, env,\nRAG collections")
step_box(470, 532, 170, 106, "LLM Reasoning",
         "streamed decoding;\ntool-call parsing", color=C_THOUGHT, tagname="THOUGHT")
D1X, D2X, DY = 712, 828, 585
diamond(D1X, DY, 44, 44, "Tool\ncall(s)?")
diamond(D2X, DY, 44, 44, "Done-\ngate?", size=12.5)
pill(900, 561, 70, 48, "Final\nAnswer", size=13)

# control-flow arrows
path(f"M262,128 V{DY} H280")                         # gateway -> context assembly
text(268, 520, "request", size=12, fill=SUB, anchor="start")
path(f"M440,{DY} H470")                              # assembly -> thought
path(f"M640,{DY} H{D1X - 44}")                       # thought -> D1
path(f"M{D1X},{DY - 44} V{AY + AH}")                 # D1 yes -> action
text(D1X + 6, DY - 58, "Yes", size=12.5, weight="bold", fill=C_ACTION)
path(f"M{D1X + 44},{DY} H{D2X - 44}")                # D1 no -> D2
text(D1X + 49, DY - 7, "No", size=12.5, weight="bold")
path(f"M{D2X + 44},{DY} H900")                       # D2 yes -> answer
text(D2X + 47, DY - 7, "Yes", size=12.5, weight="bold")
path(f"M{AX},{310} H640")                            # action -> observation
path(f"M470,{310} H440")                             # observation -> context manager
path(f"M360,370 V532")                               # context manager -> assembly
path(f"M{D2X},{DY + 44} V690 H600 V638")             # D2 no -> thought
text(D2X - 6, 706, "No: open to-dos remain → re-enter loop (goal mode)", size=12,
     fill=SUB, anchor="end")
text(D2X + 4, 663, "No", size=12.5, weight="bold")

# loop emblem
cx, cy, r = 556, 452, 34
add(f'<path d="M{cx + r},{cy} A{r},{r} 0 1 1 {cx},{cy - r}" fill="none" stroke="{C_THOUGHT}" '
    f'stroke-width="2" marker-end="url(#ah-k)"/>')
text(cx, cy - 2, "ReAct", size=12.5, weight="bold", anchor="middle", fill=C_THOUGHT)
text(cx, cy + 13, "cycle", size=12, anchor="middle", fill=C_THOUGHT)

# safeguards
rect(280, 656, 205, 76, fill="#fff", stroke="#8a8a8a", sw=1, dash="4,3")
text(292, 676, "Loop safeguards", size=13, weight="bold")
text(292, 694, "iteration budget · stop signal\nrepeat-call / failure detection",
     size=12, fill=SUB, lh=15)

# ----------------------------------------------------------------------------- (C) Tools
TX, TW = 1017, 468
section(TX, 15, TW, 745, "C", "Tool Library", "#f4f8f3")
text(TX + 160, 41, "15 tools via JSON-schema function calling", size=12.5, fill=SUB)

cats = [
    ("Knowledge Retrieval", [("rag", True), ("websearch", True)],
     "hybrid search: FAISS (dense) + BM25 → RRF\nfusion → cross-encoder re-ranking",
     "document QA · literature review · grounded answers"),
    ("Code & Computation", [("code_exec", False), ("shell_exec", False),
                            ("shell_lint", True), ("process_monitor", False)],
     None,
     "data analysis · simulation · plotting · automation"),
    ("File System & Software Engineering",
     [("file_reader", True), ("file_navigator", True), ("grep", True),
      ("file_edit", False), ("apply_patch", False), ("file_writer", False)],
     None,
     "codebase understanding · autonomous editing · reports"),
    ("Planning & Memory", [("todo_write", False), ("memo", False)],
     "task checklist · persistent key–value memory",
     "multi-step goal execution · cross-session recall"),
    ("Delegation", [("agent", True)],
     "spawns a sub-agent (read-only explore or general)\nrunning a nested ReAct loop in a fresh context",
     "task decomposition · parallel investigation"),
]
cy0 = 62
CW = 134
for title, tools, detail, usecase in cats:
    rows = (len(tools) + 2) // 3
    dl = detail.count("\n") + 1 if detail else 0
    h = 30 + rows * 31 + dl * 16 + 26
    rect(TX + 14, cy0, TW - 28, h, fill="#fff", stroke=LINE, sw=1.2)
    add(f'<rect x="{TX + 14}" y="{cy0}" width="5" height="{h}" fill="{C_ACTION}" rx="2"/>')
    text(TX + 28, cy0 + 21, title, size=14.5, weight="bold")
    for i, (nm, safe) in enumerate(tools):
        r_, c_ = divmod(i, 3)
        chip(TX + 28 + c_ * (CW + 8), cy0 + 31 + r_ * 31, CW, nm, safe, size=13)
    yy = cy0 + 31 + rows * 31 + 12
    if detail:
        text(TX + 28, yy, detail, size=12.5, fill=SUB, lh=16)
        yy += dl * 16
    text(TX + 28, yy + 2, "▸ " + usecase, size=12.5, fill=C_ACTION, style="italic")
    cy0 += h + 10

# legend
add(f'<circle cx="{TX + 32}" cy="{cy0 + 12}" r="4.2" fill="{C_ACTION}"/>')
text(TX + 42, cy0 + 17, "concurrency-safe (parallel)", size=12.5, fill=SUB)
add(f'<rect x="{TX + 232}" y="{cy0 + 8}" width="8" height="8" fill="{LINE}"/>')
text(TX + 246, cy0 + 17, "mutating (serial barrier)", size=12.5, fill=SUB)

# action <-> tools
path(f"M{AX + AW},{300} H{TX}", stroke=C_ACTION, sw=2, color_marker="g", start=True)

# ----------------------------------------------------------------------------- (D) Inference
section(15, 778, 985, 267, "D", "Local Inference Layer (on-premise)", "#fcf5ec")
layers = [
    ("Inference Engine — vLLM",
     "OpenAI-compatible serving · continuous batching · prefix (KV) caching\n"
     "tool-call parser · reasoning parser"),
    ("Open-weight LLM (swappable)",
     "e.g., GLM · Qwen · Llama · Mistral · DeepSeek"),
    ("On-premise GPU Server(s)",
     "single node, or master + N worker nodes (cluster)"),
]
ly = 815
for i, (t, b) in enumerate(layers):
    h = 66 if i == 0 else 52
    rect(30, ly, 690, h, fill="#fff", stroke=C_LLM, sw=1.4)
    text(44, ly + 21, t, size=14.5, weight="bold")
    text(300, ly + 21, b, size=12.5, fill=SUB, lh=16)
    ly += h + 8
rect(740, 815, 245, 186, fill="#fff", stroke=C_LLM, sw=1.4)
text(754, 838, "Deployment properties", size=14.5, weight="bold")
props = ["air-gapped operation", "data never leaves premises", "no per-token API cost",
         "model-agnostic backend", "horizontal scale-out"]
for i, p in enumerate(props):
    text(756, 864 + i * 26, "✓", size=14, weight="bold", fill=C_LLM)
    text(776, 864 + i * 26, p, size=13, fill=INK)
text(30, 1030, "The agent requires only an OpenAI-compatible local endpoint: replacing the model "
     "changes no agent or tool code.", size=12, fill=SUB, style="italic")

# thought <-> LLM
path(f"M510,638 V815", stroke=C_LLM, sw=2, dash="6,4", color_marker="o", start=True)
text(518, 728, "prompt + tool schemas  ↓", size=12, fill=C_LLM, weight="bold")
text(518, 743, "token stream  ↑", size=12, fill=C_LLM, weight="bold")

# ----------------------------------------------------------------------------- (E) State
section(TX, 778, TW, 267, "E", "Persistent State", "#f6f6f6")
stores = [("Sessions", "conversation\nhistory"), ("Vector", "FAISS + BM25\nindex"),
          ("Memory", "memo\nstore"), ("Spill", "offloaded\ntool output"),
          ("Jobs & Logs", "async ingest,\naudit trail")]
sw_ = (TW - 28) / len(stores)
for i, (t, b) in enumerate(stores):
    cx = TX + 14 + sw_ * (i + 0.5)
    cylinder(cx, 838, 62, 58, fill="#fff")
    text(cx, 930, t, size=13.5, weight="bold", anchor="middle")
    text(cx, 948, b, size=12, fill=SUB, anchor="middle", lh=15)
text(TX + 18, 1000, "tools read / write state; sessions resume across restarts", size=12,
     fill=SUB, style="italic")

add('</svg>')

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n".join(out))
