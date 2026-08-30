# the two headline figures for the paper: the gap between what a mention-checking grader reports and what
# actually tracks, and the collapse of that gap when confabulation is measured per trace rather than per item.
# reads finished reports off disk, no gpu and no api.
# uv run -m scripts.plot_headline --cells_report_path results/analysis/cells_v1/cells_report.json --attribution_report_path results/analysis/attribution_v1/attribution_report.json --output_dir results/analysis/headline_v1 --model_id Qwen/Qwen3-8B

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

# okabe-ito. one meaning per hue across both figures: orange is what the naive metric reports, blue is
# what the cue swap shows is actually happening.
APPARENT = "#E69F00"
ACTUAL = "#0072B2"
NEUTRAL = "#999999"
ARM_LABELS = {"C0_bare": "C0 bare", "C2_monitored": "C2 monitored", "C3_neutral_private": "C3 private"}


@dataclass
class Config:
    cells_report_path: str = ""
    attribution_report_path: str = ""
    output_dir: str = "results/analysis/headline_v1"
    model_id: str = "Qwen/Qwen3-8B"
    seed: int = 42


def _bar_errors(points: list[float], cis: list[list[float]]) -> list[list[float]]:
    return [[p - ci[0] for p, ci in zip(points, cis)], [ci[1] - p for p, ci in zip(points, cis)]]


def gap_figure(by_arm: dict, model_id: str, path: Path) -> tuple[Path, dict]:
    """what a mention-checking grader reports against what the answer actually does, per framing arm.
    the distance between the two bars is the paper's headline and every arm shows the same one."""
    arms = list(by_arm)
    series = [
        ("naive_verbalization", "Trace credits the hint\n(what a mention-checking grader reports)", APPARENT),
        ("tracking", "Answer follows the hint as it moves\n(what the cue swap measures)", ACTUAL),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    positions = np.arange(len(arms))
    plotted: dict = {}
    for offset, (key, label, color) in enumerate(series):
        points = [by_arm[a][key]["point"] for a in arms]
        cis = [by_arm[a][key]["ci"] for a in arms]
        ax.bar(
            positions + (offset - 0.5) * 0.36,
            points,
            0.36,
            label=label,
            color=color,
            yerr=_bar_errors(points, cis),
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "#333333"},
        )
        plotted[key] = {a: {"point": p, "ci": ci} for a, p, ci in zip(arms, points, cis)}

    ax.axhline(GATES.behavioral_tracking_min, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(
        -0.45,
        GATES.behavioral_tracking_min + 0.015,
        f"tracking gate {GATES.behavioral_tracking_min}",
        fontsize=8,
        color="#666666",
        ha="left",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Saying the hint drove the answer, against the answer following the hint\n"
        f"{model_id}, {by_arm[arms[0]]['n_items']} items per arm, item-level 95% intervals",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, plotted


def resolution_figure(report: dict, cells_by_arm: dict, model_id: str, path: Path) -> tuple[Path, dict]:
    """the same quantity measured per item and per trace, and what the item-level number is made of.
    the cell rule aggregates 24 traces and three placements before comparing, so it can score an item
    confabulated when no single trace credited a hint it did not follow."""
    coherence = report["trace_coherence"]
    decomposition = report["confabulation_decomposition"]

    n_faithful = sum(c["cells"]["faithful"]["count"] for c in cells_by_arm.values())
    n_confabulated = sum(c["cells"]["confabulated"]["count"] for c in cells_by_arm.values())
    item_rate = n_confabulated / (n_faithful + n_confabulated)
    trace_rate = 1 - coherence["answered_cue_given_attributes"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})

    bars = left.bar(
        [0, 1],
        [item_rate, trace_rate],
        0.55,
        color=[APPARENT, ACTUAL],
    )
    for bar, value, n in zip(bars, [item_rate, trace_rate], [n_faithful + n_confabulated, coherence["n_attributing"]]):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}\nn = {n:,}",
            ha="center",
            fontsize=9,
            color="#333333",
        )
    left.set_xticks([0, 1])
    left.set_xticklabels(
        ["Per item\n(the four-cell rule)", "Per trace\n(same trace credits\nand answers)"], fontsize=9
    )
    left.set_ylabel("Credits the hint but does not follow it")
    left.set_ylim(0, 0.46)
    left.set_title("Apparent confabulation, measured two ways", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)
    left.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    left.set_axisbelow(True)

    counts = decomposition["counts"]
    threshold_only = counts["modal_is_the_cue_below_threshold"]
    different = counts["modal_is_a_different_option"]
    right.barh([0], [different], 0.5, color=ACTUAL, label="Modal answer is a different option")
    right.barh(
        [0],
        [threshold_only],
        0.5,
        left=[different],
        color=NEUTRAL,
        label=f"Modal answer is the cue, below the {decomposition['modal_min_count']}-of-8 bar",
    )
    for x, value, color in [(different / 2, different, "white"), (different + threshold_only / 2, threshold_only, "white")]:
        right.text(x, 0, f"{value}", ha="center", va="center", fontsize=11, color=color, fontweight="bold")
    right.set_yticks([])
    right.set_ylim(-1.1, 1.1)
    right.set_xlabel("Placements where a confabulated item fails to track")
    right.set_xlim(0, decomposition["n_failing_placements"] * 1.02)
    right.set_title(
        f"What the item-level number is made of\n{decomposition['share_below_threshold_only']:.0%} is the "
        f"modal threshold alone",
        fontsize=10,
    )
    right.spines[["top", "right", "left"]].set_visible(False)
    right.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)

    fig.suptitle(
        f"Item-level cell assignment overstates confabulation — {model_id}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, {
        "item_level_confabulation": round(item_rate, 4),
        "trace_level_confabulation": round(trace_rate, 4),
        "n_attributing_items": n_faithful + n_confabulated,
        "n_attributing_traces": coherence["n_attributing"],
        "failing_placements": decomposition["counts"],
        "share_below_threshold_only": decomposition["share_below_threshold_only"],
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = simple_parsing.parse(Config)
    if not config.cells_report_path:
        raise ValueError("--cells_report_path is required")
    if not config.attribution_report_path:
        raise ValueError("--attribution_report_path is required")

    cells = json.loads(Path(config.cells_report_path).read_text())
    attribution = json.loads(Path(config.attribution_report_path).read_text())["report"]

    out = Path(config.output_dir)
    figures = out / "figures"
    gap_path, gap_values = gap_figure(cells["by_arm"], config.model_id, figures / "verbalization_gap.png")
    res_path, res_values = resolution_figure(
        attribution, cells["by_arm"], config.model_id, figures / "item_against_trace_level.png"
    )

    # every number on a figure has to exist in a structured file, so the figures are auditable.
    values_path = write_json(
        out / "headline_figure_values.json",
        {
            "git_hash": get_git_hash(),
            "cells_report_path": config.cells_report_path,
            "attribution_report_path": config.attribution_report_path,
            "model_id": config.model_id,
            "verbalization_gap": gap_values,
            "item_against_trace_level": res_values,
        },
    )

    print(f"Figure: {gap_path}")
    print(f"Figure: {res_path}")
    print(f"Values: {values_path}")
    for arm, d in gap_values["tracking"].items():
        naive = gap_values["naive_verbalization"][arm]["point"]
        print(f"  {arm:<20} credits {naive}  tracks {d['point']}  gap {round(naive - d['point'], 4)}")
    print(
        f"  confabulation per item {res_values['item_level_confabulation']} against per trace "
        f"{res_values['trace_level_confabulation']}"
    )


main()
