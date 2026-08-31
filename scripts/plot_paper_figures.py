# the three paper figures, each carrying one of the paper's three results, both cues on every panel.
# reads finished reports off disk, no gpu and no api.
# uv run -m scripts.plot_paper_figures --output_dir paper/figures --model_id Qwen/Qwen3-8B

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing

from src.config import GATES
from src.utils.io import get_git_hash, write_json

LOGGER = logging.getLogger(__name__)

# okabe-ito, one meaning per hue across all three figures.
SAYS = "#E69F00"  # what the trace claims / what a mention-checker reports
DOES = "#0072B2"  # what the answer actually does
MUTED = "#999999"

CUES = [
    ("grader_code", "Hint states the grading rule", "results/analysis/cells_v1", "results/analysis/attribution_v2", "results/raw/prefill_v2_nocue"),
    ("answer_key", "Hint states the answer", "results/analysis/cells_answer_key", "results/analysis/attribution_answer_key_v2", "results/raw/prefill_answer_key_nocue"),
]
ARM_LABEL = {"C0_bare": "unstated", "C3_neutral_private": "private", "C2_monitored": "monitored"}


@dataclass
class Config:
    output_dir: str = "paper/figures"
    model_id: str = "Qwen/Qwen3-8B"
    seed: int = 42


def _load(cells_dir: str, attrib_dir: str, prefill_dir: str) -> dict:
    return {
        "cells": json.loads(Path(cells_dir, "cells_report.json").read_text()),
        "attrib": json.loads(Path(attrib_dir, "attribution_report.json").read_text())["report"],
        "prefill": json.loads(Path(prefill_dir, "prefill_report.json").read_text()),
    }


def figure_one(data: dict, model_id: str, path: Path) -> dict:
    """result 1: a mention-checking grader reports far more than the answer actually does, in every arm
    of both cues. the distance between the two bars is the paper's motivating gap."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    out: dict = {}
    for ax, (cue, title, *_rest) in zip(axes, CUES):
        by_arm = data[cue]["cells"]["by_arm"]
        arms = [a for a in ("C0_bare", "C3_neutral_private", "C2_monitored") if a in by_arm]
        pos = np.arange(len(arms))
        for offset, (key, label, color) in enumerate(
            [("naive_verbalization", "Trace credits the hint", SAYS), ("tracking", "Answer follows the hint", DOES)]
        ):
            points = [by_arm[a][key]["point"] for a in arms]
            cis = [by_arm[a][key]["ci"] for a in arms]
            err = [[p - c[0] for p, c in zip(points, cis)], [c[1] - p for p, c in zip(points, cis)]]
            ax.bar(pos + (offset - 0.5) * 0.36, points, 0.36, label=label, color=color,
                   yerr=err, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#333333"})
            out.setdefault(cue, {})[key] = dict(zip(arms, points))
        ax.axhline(GATES.behavioral_tracking_min, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xticks(pos)
        ax.set_xticklabels([ARM_LABEL[a] for a in arms])
        ax.set_xlabel("Reasoning described as")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#ececec", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, 1.08)
    axes[0].text(-0.42, GATES.behavioral_tracking_min + 0.015, f"tracking gate {GATES.behavioral_tracking_min}",
                 fontsize=7.5, color="#666666")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Saying the hint drove the answer, against the answer following it — {model_id}", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_two(data: dict, model_id: str, path: Path) -> dict:
    """result 2: the same quantity — credits the hint but does not follow it — measured per item by the
    four-cell rule and per trace. the item-level rule aggregates 24 samples before comparing."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    out: dict = {}
    pos = np.arange(2)
    for offset, (cue, title, *_rest) in enumerate(CUES):
        by_arm = data[cue]["cells"]["by_arm"]
        faithful = sum(a["cells"]["faithful"]["count"] for a in by_arm.values())
        confab = sum(a["cells"]["confabulated"]["count"] for a in by_arm.values())
        item_rate = confab / (faithful + confab)
        coherence = data[cue]["attrib"]["trace_coherence"]
        trace_rate = 1 - coherence["answered_cue_given_attributes"]
        bars = ax.bar(pos + (offset - 0.5) * 0.34, [item_rate, trace_rate], 0.34,
                      label=title, color=[SAYS, DOES][offset], edgecolor="none")
        for bar, value, n in zip(bars, [item_rate, trace_rate], [faithful + confab, coherence["n_attributing"]]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}\nn={n:,}",
                    ha="center", fontsize=8.5, color="#333333")
        out[cue] = {"per_item": round(item_rate, 4), "per_trace": round(trace_rate, 4)}
    ax.set_xticks(pos)
    ax.set_xticklabels(["Per item\n(the four-cell rule,\n24 samples aggregated)", "Per trace\n(same trace credits\nand answers)"])
    ax.set_ylabel("Credits the hint but does not follow it")
    ax.set_ylim(0, 0.46)
    ax.set_title(f"Item-level scoring overstates confabulation — {model_id}\nsame quantity, two units of analysis", fontsize=10.5)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_three(data: dict, model_id: str, path: Path) -> dict:
    """result 3: deleting the sentence that mentions the hint changes the answer on faithful items and
    not on confabulated ones, in both cues. a dot-and-interval plot, since the delta is the result."""
    rows = []
    for cue, title, *_rest in CUES:
        for cell in ("faithful", "confabulated"):
            d = data[cue]["prefill"]["by_cell"][cell]
            rows.append((title, cell, d["mean_delta"], d["delta_ci"], d["n_items"]))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ypos = np.arange(len(rows))[::-1]
    for y, (title, cell, delta, ci, n) in zip(ypos, rows):
        color = DOES if cell == "faithful" else MUTED
        ax.plot([ci[0], ci[1]], [y, y], color=color, linewidth=2.2, solid_capstyle="round")
        ax.plot([delta], [y], "o", color=color, markersize=8)
        ax.text(0.30, y, f"{delta:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]   n={n}", va="center", fontsize=8.5, color="#333333")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{cell}\n{title.split(' ')[-1]}" for title, cell, *_ in rows], fontsize=8.5)
    ax.set_xlim(-0.10, 0.55)
    ax.set_xlabel("Change in rate of answering the hinted option when the mention is restored")
    ax.set_title(f"Is the hint mention load-bearing? — {model_id}\nmention removed against mention kept, hint absent from the prompt", fontsize=10.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {f"{t}|{c}": {"delta": d, "ci": ci, "n_items": n} for t, c, d, ci, n in rows}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = simple_parsing.parse(Config)
    data = {cue: _load(cd, ad, pd) for cue, _title, cd, ad, pd in CUES}
    out = Path(config.output_dir)

    values = {
        "figure_1_verbalization_gap": figure_one(data, config.model_id, out / "fig1_verbalization_gap.png"),
        "figure_2_item_against_trace": figure_two(data, config.model_id, out / "fig2_item_against_trace.png"),
        "figure_3_prefill_ablation": figure_three(data, config.model_id, out / "fig3_prefill_ablation.png"),
    }
    values_path = write_json(out / "paper_figure_values.json", {"git_hash": get_git_hash(), "model_id": config.model_id, **values})

    for name in ("fig1_verbalization_gap", "fig2_item_against_trace", "fig3_prefill_ablation"):
        print(f"Figure: {out / (name + '.png')}")
    print(f"Values: {values_path}")


main()
