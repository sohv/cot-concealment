# analysis stage: pulls the four cell assignments together across arms, computes the correction factor
# between what a mention-checking grader reports and what the cue swap validates, and plots the headline
# comparison of behavioural tracking against verbalization per arm. reads rows off disk, no gpu, no api.
# uv run -m scripts.analyze_cells --judge_dir results/raw/judge_v1 --tracking_results_path results/raw/sweep_v1_resume/tracking_results.jsonl --output_dir results/analysis/cells_v1 --seed 42

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing

from src.config import GATES, SCORING
from src.metrics.bootstrap import (
    attribution_precision,
    bootstrap_cells,
    bootstrap_proportion,
    corrected_faithfulness,
    correction_factor,
    naive_verbalization,
)
from src.utils.io import get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)

BAR_SERIES = (
    ("naive_verbalization", "Naive verbalization\n(mention-checking grader)", "#dd8452"),
    ("tracking", "Behavioural tracking\n(answer follows the hint)", "#4c72b0"),
    ("attribution_precision", "Attribution precision\n(credited hints that track)", "#55a868"),
    ("corrected_faithfulness", "Corrected faithfulness\n(real influence disclosed)", "#8172b3"),
)
CELL_COLORS = {
    "faithful": "#55a868",
    "confabulated": "#c44e52",
    "silent": "#8172b3",
    "independent": "#b8b8b8",
}


@dataclass
class Config:
    judge_dir: str = ""
    tracking_results_path: str = ""
    output_dir: str = "results/analysis/cells_v1"
    model_id: str = "Qwen/Qwen3-8B"
    seed: int = 42


def paired_arm_deltas(cells_by_arm: dict[str, dict[str, str]], arms: list[str]) -> list[dict]:
    """the arms share their item set, so a difference in the correction factor is read over the items
    both arms scored, never off the overlap of two marginal intervals."""
    out = []
    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            shared = sorted(set(cells_by_arm[a]) & set(cells_by_arm[b]))
            rng = np.random.default_rng(SCORING.bootstrap_seed)
            draws = []
            for _ in range(SCORING.bootstrap_resamples):
                pick = rng.choice(len(shared), size=len(shared), replace=True)
                items = [shared[i] for i in pick]
                summaries = [Counter(cells_by_arm[arm][i] for i in items) for arm in (a, b)]
                values = []
                for c in summaries:
                    denominator = c["faithful"] + c["silent"]
                    values.append(c["faithful"] / denominator if denominator else None)
                if None not in values:
                    draws.append(values[0] - values[1])
            out.append(
                {
                    "a": a,
                    "b": b,
                    "n_shared_items": len(shared),
                    "difference_corrected_faithfulness": round(float(np.mean(draws)), 4) if draws else None,
                    "paired_ci": [
                        round(float(np.quantile(draws, 0.025)), 4),
                        round(float(np.quantile(draws, 0.975)), 4),
                    ]
                    if draws
                    else None,
                }
            )
    return out


def headline_figure(report: dict, arms: list[str], model_id: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(2.6 * len(arms) + 3.4, 4.8))
    width = 0.2
    positions = np.arange(len(arms))
    for offset, (key, label, color) in enumerate(BAR_SERIES):
        points, lows, highs = [], [], []
        for arm in arms:
            d = report["by_arm"][arm][key]
            point = d["point"] or 0.0
            ci = d["ci"] or [point, point]
            points.append(point)
            lows.append(point - ci[0])
            highs.append(ci[1] - point)
        ax.bar(
            positions + (offset - (len(BAR_SERIES) - 1) / 2) * width,
            points,
            width,
            label=label,
            color=color,
            yerr=[lows, highs],
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "#444444"},
        )

    ax.axhline(
        GATES.behavioral_tracking_min,
        color="#c44e52",
        linestyle="--",
        linewidth=1,
        label=f"Tracking gate {GATES.behavioral_tracking_min}",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([a.replace("_", " ") for a in arms])
    ax.set_ylabel("Rate")
    ax.set_title(
        f"Tracking against verbalization by framing arm\n{model_id}, {report['n_items_per_arm']} items per arm, "
        f"item-level 95% intervals",
        fontsize=10,
    )
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=7, loc="upper center", ncol=4, frameon=False, columnspacing=1.0, handlelength=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def cell_composition_figure(report: dict, arms: list[str], model_id: str, path: Path) -> Path:
    """the four cells stacked per arm. with the silent cell empty this is the clearest single picture:
    the confabulated block is the whole gap between what the grader sees and what the swap validates."""
    fig, ax = plt.subplots(figsize=(1.6 * len(arms) + 4.4, 4.4))
    bottoms = np.zeros(len(arms))
    for cell, color in CELL_COLORS.items():
        values = np.array([report["by_arm"][arm]["cells"][cell]["point"] for arm in arms])
        bars = ax.bar([a.replace("_", " ") for a in arms], values, 0.55, bottom=bottoms, label=cell, color=color)
        for bar, value, arm in zip(bars, values, arms):
            if value > 0.03:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + value / 2,
                    f"{report['by_arm'][arm]['cells'][cell]['count']}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )
        bottoms += values
    ax.set_ylabel("Share of items")
    ax.set_ylim(0, 1)
    ax.set_title(f"Cell composition by framing arm\n{model_id}, {report['n_items_per_arm']} items per arm", fontsize=10)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = simple_parsing.parse(Config)
    if not config.judge_dir:
        raise ValueError("--judge_dir is required")
    if not config.tracking_results_path:
        raise ValueError("--tracking_results_path is required")

    judge_dir = Path(config.judge_dir)
    cell_rows = read_jsonl(judge_dir / "cell_results.jsonl")
    judge_report = json.loads((judge_dir / "judge_report.json").read_text())
    tracking_rows = read_jsonl(config.tracking_results_path)
    arms = sorted({r["arm"] for r in cell_rows})

    cells_by_arm = {arm: {r["item_id"]: r["cell"] for r in cell_rows if r["arm"] == arm} for arm in arms}
    tracking_by_arm = {arm: {r["item_id"]: r["tracks_all"] for r in tracking_rows if r["arm"] == arm} for arm in arms}

    by_arm = {}
    for arm in arms:
        items = sorted(cells_by_arm[arm])
        summary = bootstrap_cells([cells_by_arm[arm][i] for i in items])
        by_arm[arm] = {
            "n_items": len(items),
            "cells": summary["cells"],
            "tracking": bootstrap_proportion([tracking_by_arm[arm][i] for i in items]),
            "naive_verbalization": summary["naive_verbalization"],
            "corrected_faithfulness": summary["corrected_faithfulness"],
            "attribution_precision": summary["attribution_precision"],
            "correction_factor": summary["correction_factor"],
            "confabulation_rate": summary["confabulation_rate"],
            "zero_cell_upper_bounds": summary["zero_cell_upper_bounds"],
        }

    n_per_arm = sorted({d["n_items"] for d in by_arm.values()})
    report = {
        "git_hash": get_git_hash(),
        "judge_dir": str(judge_dir),
        "judge_model": judge_report["judge_model"],
        "tracking_results_path": config.tracking_results_path,
        "arms": arms,
        "n_items_per_arm": n_per_arm[0] if len(n_per_arm) == 1 else n_per_arm,
        "scoring": SCORING.model_dump(),
        "false_attribution": judge_report["false_attribution"],
        "by_arm": by_arm,
        "paired_arm_deltas": paired_arm_deltas(cells_by_arm, arms),
        "sensitivity_grid_by_arm": {
            arm: [
                row
                | {
                    "corrected_faithfulness": corrected_faithfulness(row["counts"]),
                    "attribution_precision": attribution_precision(row["counts"]),
                    "naive_verbalization": naive_verbalization(row["counts"]),
                    "correction_factor": correction_factor(row["counts"]),
                }
                for row in judge_report["by_arm"][arm]["sensitivity_grid"]
            ]
            for arm in arms
        },
    }

    out = Path(config.output_dir)
    report_path = write_json(out / "cells_report.json", report)
    figure_path = headline_figure(report, arms, config.model_id, out / "figures" / "tracking_vs_verbalization.png")
    cells_figure_path = cell_composition_figure(report, arms, config.model_id, out / "figures" / "cell_composition.png")

    print(f"Report: {report_path}")
    print(f"Figure: {figure_path}")
    print(f"Figure: {cells_figure_path}")
    print(
        f"False attribution on V0: {report['false_attribution']['rate']}, "
        f"gate <{GATES.false_attribution_max}: {report['false_attribution']['gate_passes']}"
    )
    for arm, d in by_arm.items():
        counts = ", ".join(f"{c} {d['cells'][c]['count']}" for c in SCORING.cells)
        print(f"\n{arm}: n={d['n_items']}, {counts}")
        for label, key in [
            ("tracking", "tracking"),
            ("naive verbalization", "naive_verbalization"),
            ("attribution precision", "attribution_precision"),
            ("corrected faithfulness", "corrected_faithfulness"),
            ("correction factor", "correction_factor"),
        ]:
            print(f"  {label:<24} {d[key]['point']} {d[key]['ci']}")
        print(
            f"  {'confabulation rate':<24} {d['confabulation_rate']['point']} {d['confabulation_rate']['ci']}, "
            f"report threshold {GATES.confabulation_report_threshold}"
        )
        if d["zero_cell_upper_bounds"]:
            bounds = ", ".join(f"{c} 0 observed, 95% upper bound {b}" for c, b in d["zero_cell_upper_bounds"].items())
            print(f"  {'empty cells':<24} {bounds}")
    print("\nPaired over shared items, difference in corrected faithfulness")
    for p in report["paired_arm_deltas"]:
        print(f"  {p['a']} vs {p['b']}: {p['difference_corrected_faithfulness']} {p['paired_ci']}")


main()
