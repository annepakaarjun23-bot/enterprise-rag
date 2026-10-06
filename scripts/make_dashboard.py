import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

SRC = Path("evals/results/runs.jsonl")
OUT = Path("docs/images/eval_dashboard.png")
THRESH = 0.7
TOTAL_Q = 50

BLUE, TEAL, ORANGE, PURPLE = "#3b6fd4", "#2a9d8f", "#e07a2f", "#8e6bbf"
GREY, INK, MUTED, CARD = "#c9ced6", "#1f2933", "#6b7280", "#f6f8fb"

runs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]

ret = [r for r in runs
       if r["run"].startswith("retriever_") and r.get("questions") == 50
       and "Contextual Precision_avg" in r and "Contextual Recall_avg" in r]
gen = [r for r in runs if r["run"].startswith("generator_") and r.get("questions") == 50][-1]
e2e = [r for r in runs if r["run"].startswith("e2e_")][-1]

groups = defaultdict(list)
for r in ret:
    groups[(r["mode"], r["k"])].append(r)

E2E_METRICS = ["Contextual Precision", "Contextual Recall", "Faithfulness", "Answer Relevancy"]
overall = sum(e2e[f"{m}_avg"] for m in E2E_METRICS) / len(E2E_METRICS)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

fig = plt.figure(figsize=(15, 11), facecolor="white")
gs = GridSpec(3, 6, figure=fig, height_ratios=[0.55, 1, 1], hspace=0.5, wspace=0.9)
fig.suptitle("Evaluation dashboard", fontsize=20, fontweight="bold", color=INK, x=0.07, ha="left")


def card(ax, label, value, sub, color):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=CARD, ec=GREY, lw=1))
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.025, 0.9, boxstyle="square,pad=0", fc=color, ec=color))
    ax.text(0.12, 0.72, label, fontsize=10, color=MUTED)
    ax.text(0.12, 0.36, f"{value:.2f}", fontsize=26, fontweight="bold", color=INK)
    ax.text(0.12, 0.12, sub, fontsize=8, color=MUTED)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.08)
    ax.axhline(THRESH, color=ORANGE, ls="--", lw=1)
    ax.grid(axis="y", color="#eef0f3")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def label_bars(ax, xs, vals, dy=0.03):
    for x, v in zip(xs, vals):
        ax.text(x, v + dy, f"{v:.2f}", ha="center", fontsize=9.5, color=INK)


# Row 0: headline cards (end-to-end values)
sub = f"End-to-end, {e2e['questions']} questions"
cards = [("Overall average", overall, "Mean of the four metrics", INK),
         ("Contextual Precision", e2e["Contextual Precision_avg"], sub, BLUE),
         ("Contextual Recall", e2e["Contextual Recall_avg"], sub, TEAL),
         ("Faithfulness", e2e["Faithfulness_avg"], sub, ORANGE),
         ("Answer Relevancy", e2e["Answer Relevancy_avg"], sub, PURPLE)]
spans = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
# widen first card by giving it two columns, others one each
for (label, val, s, col), (a, b) in zip(cards, spans):
    card(fig.add_subplot(gs[0, a:b]), label, val, s, col)

# Row 1, panel 1: retrieval only
ax = fig.add_subplot(gs[1, 0:2])
style(ax, "Retrieval only (50 questions)")
w = 0.36
for i, (metric, color) in enumerate([("Contextual Precision_avg", BLUE), ("Contextual Recall_avg", TEAL)]):
    means, lo, hi = [], [], []
    for v in groups.values():
        vals = [r[metric] for r in v]
        m = sum(vals) / len(vals)
        means.append(m); lo.append(m - min(vals)); hi.append(max(vals) - m)
    xs = [j + (i - 0.5) * w for j in range(len(groups))]
    ax.bar(xs, means, w, color=color, label=metric.replace("_avg", "").replace("Contextual ", ""),
           yerr=[lo, hi], capsize=3, error_kw={"ecolor": INK, "lw": 1})
    label_bars(ax, xs, [m + h for m, h in zip(means, hi)], dy=0.02)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{m.title()}\nk={k}\n({len(v)} run{'s' if len(v) > 1 else ''})"
                    for (m, k), v in groups.items()], fontsize=9)
ax.legend(frameon=False, loc="lower right", fontsize=8)

# Row 1, panel 2: generator only
ax = fig.add_subplot(gs[1, 2:4])
style(ax, f"Generator only ({gen['questions']} questions)")
gm = ["Faithfulness", "Answer Relevancy"]
gv = [gen[f"{m}_avg"] for m in gm]
ax.bar(range(2), gv, 0.5, color=[ORANGE, PURPLE])
label_bars(ax, range(2), gv)
ax.set_xticks(range(2)); ax.set_xticklabels(gm)

# Row 1, panel 3: end-to-end
ax = fig.add_subplot(gs[1, 4:6])
style(ax, f"Full pipeline, e2e ({e2e['questions']} questions)")
ev = [e2e[f"{m}_avg"] for m in E2E_METRICS]
ax.bar(range(4), ev, 0.55, color=[BLUE, TEAL, ORANGE, PURPLE])
label_bars(ax, range(4), ev)
ax.set_xticks(range(4)); ax.set_xticklabels([m.replace(" ", "\n") for m in E2E_METRICS], fontsize=8.5)

# Row 2, left: generator only vs full pipeline
ax = fig.add_subplot(gs[2, 0:3])
style(ax, "Generation quality: generator only vs full pipeline")
for i, (src, label, color) in enumerate([(gen, f"Generator only ({gen['questions']} q)", GREY),
                                         (e2e, f"Full pipeline ({e2e['questions']} q)", BLUE)]):
    xs = [j + (i - 0.5) * 0.36 for j in range(2)]
    v = [src[f"{m}_avg"] for m in gm]
    ax.bar(xs, v, 0.36, color=color, label=label)
    label_bars(ax, xs, v)
ax.set_xticks(range(2)); ax.set_xticklabels(gm)
ax.legend(frameon=False, loc="lower right", fontsize=9)

# Row 2, right: history of comparable retrieval runs
ax = fig.add_subplot(gs[2, 3:6])
style(ax, "History of 50-question retrieval runs")
xs = list(range(1, len(ret) + 1))
ax.plot(xs, [r["Contextual Precision_avg"] for r in ret], "-o", color=BLUE, label="Precision")
ax.plot(xs, [r["Contextual Recall_avg"] for r in ret], "-o", color=TEAL, label="Recall")
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['mode']}\nk={r['k']}" for r in ret], fontsize=8)
ax.legend(frameon=False, loc="lower right", fontsize=9)

fig.text(0.07, 0.04,
         f"Dashed line: 0.70 threshold. Judge: gpt-4o-mini. Whiskers: min-max across runs. "
         f"End-to-end run covers {e2e['questions']} of {TOTAL_Q} questions. "
         f"Generator-only and e2e use different question sets and are not directly comparable.",
         fontsize=9, color=MUTED)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")