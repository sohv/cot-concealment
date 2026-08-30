# the prefill ablation figure: cued-option rate before and against after the mention sentence, per cell,
# with the cue present in the prompt and with it stripped. the two panels share a y axis because the drop
# between them is the cue's own effect and is part of the result.
# uv run -m scripts.plot_prefill --report_paths results/raw/prefill_v1/prefill_report.json,results/raw/prefill_v2_nocue/prefill_report.json --output_dir results/analysis/prefill_v1 --model_id Qwen/Qwen3-8B

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing

from src.utils.io import get_git_hash, write_json

LOGGER = logging.getLogger(__name__)

# gray is the arm with the mention cut away, blue the arm that keeps it. one meaning per hue.
WITHOUT = "#999999"
WITH = "#0072B2"
CELLS = ("faithful", "confabulated")


@dataclass
class Config:
    report_paths: str = ""
    output_dir: str = "results/analysis/prefill_v1"
    model_id: str = "Qwen/Qwen3-8B"
    seed: int = 42


def panel_title(report: dict) -> str:
    return (
        "Cue stripped from the prompt\n(the trace is the only place the hint appears)"
        if report.get("strip_cue")
        else "Cue present in the prompt\n(cutting the trace removes a restatement)"
    )


def prefill_figure(reports: list[dict], model_id: str, path: Path) -> tuple[Path, dict]:
    fig, axes = plt.subplots(1, len(reports), figsize=(4.9 * len(reports), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    plotted: dict = {}
    for ax, report in zip(axes, reports):
        by_cell = report["by_cell"]
        cells = [c for c in CELLS if c in by_cell]
        positions = np.arange(len(cells))
        for offset, (key, label, color) in enumerate(
            [("mean_rate_before", "Mention cut away", WITHOUT), ("mean_rate_after", "Mention kept", WITH)]
        ):
            ax.bar(
                positions + (offset - 0.5) * 0.36,
                [by_cell[c][key] for c in cells],
                0.36,
                label=label,
                color=color,
            )
        for i, cell in enumerate(cells):
            d = by_cell[cell]
            # the delta and its interval are the result; the bars only show where it comes from.
            excludes_zero = d["delta_ci"][0] > 0 or d["delta_ci"][1] < 0
            ax.text(
                i,
                max(d["mean_rate_before"], d["mean_rate_after"]) + 0.04,
                f"{d['mean_delta']:+.3f}\n[{d['delta_ci'][0]:.3f}, {d['delta_ci'][1]:.3f}]",
                ha="center",
                fontsize=8.5,
                color="#111111" if excludes_zero else "#888888",
                fontweight="bold" if excludes_zero else "normal",
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{c}\nn = {by_cell[c]['n_items']}" for c in cells])
        ax.set_title(panel_title(report), fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
        ax.set_axisbelow(True)
        plotted[report["run_id"]] = {
            "strip_cue": bool(report.get("strip_cue")),
            **{c: {k: by_cell[c][k] for k in ("mean_rate_before", "mean_rate_after", "mean_delta", "delta_ci", "n_items")} for c in cells},
        }

    axes[0].set_ylabel("Continuations answering the cued option")
    axes[0].set_ylim(0, 1.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        f"Is the hint mention load-bearing? — {model_id}, held out placements, C3 arm\n"
        f"bold delta means the 95% interval excludes zero",
        fontsize=10.5,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, plotted


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = simple_parsing.parse(Config)
    if not config.report_paths:
        raise ValueError("--report_paths is required")

    paths = [p.strip() for p in config.report_paths.split(",") if p.strip()]
    reports = [json.loads(Path(p).read_text()) for p in paths]

    out = Path(config.output_dir)
    figure_path, values = prefill_figure(reports, config.model_id, out / "figures" / "prefill_ablation.png")
    values_path = write_json(
        out / "prefill_figure_values.json",
        {"git_hash": get_git_hash(), "report_paths": paths, "model_id": config.model_id, "panels": values},
    )

    print(f"Figure: {figure_path}")
    print(f"Values: {values_path}")
    for run_id, d in values.items():
        print(f"  {run_id} strip_cue={d['strip_cue']}")
        for cell in CELLS:
            if cell in d:
                c = d[cell]
                print(
                    f"    {cell:<13} before {c['mean_rate_before']:.4f} after {c['mean_rate_after']:.4f} "
                    f"delta {c['mean_delta']:+.4f} {c['delta_ci']}"
                )


main()
