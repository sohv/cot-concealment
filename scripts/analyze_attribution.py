# trace-level attribution analysis: whether a trace that credits the hint also answers it, where the
# item-level confabulated cell actually comes from, and how both depend on the option letter carrying
# the cue. reads judge and sweep rows off disk, no gpu and no api.
# uv run -m scripts.analyze_attribution --judge_dir results/raw/judge_v1 --tracking_results_path results/raw/sweep_v1_resume/tracking_results.jsonl --output_dir results/analysis/attribution_v1 --seed 42

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing

from src.config import OPTIONS, SCORING
from src.metrics.attribution import attribution_report
from src.metrics.placement import attribution_by_letter_report, build_placement_attributions
from src.utils.io import get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    judge_dir: str = ""
    tracking_results_path: str = ""
    output_dir: str = "results/analysis/attribution_v1"
    model_id: str = "Qwen/Qwen3-8B"
    seed: int = 42


def dissociation_figure(by_letter: dict, model_id: str, n_items: int, path: Path) -> Path:
    """the two rates side by side per letter. they move in opposite directions, which is the finding:
    the letter where the model most often says it is following the hint is the one where it least often
    does."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    positions = np.arange(len(OPTIONS))
    for offset, (key, label, color) in enumerate(
        [
            ("attribution_rate_by_letter", "Placements crediting the hint", "#dd8452"),
            ("tracks_given_attributes_by_letter_heldout", "Of those, answer follows it (held out)", "#4c72b0"),
        ]
    ):
        points = [by_letter[key][letter]["point"] for letter in OPTIONS]
        cis = [by_letter[key][letter]["ci"] for letter in OPTIONS]
        errors = [[p - ci[0] for p, ci in zip(points, cis)], [ci[1] - p for p, ci in zip(points, cis)]]
        ax.bar(
            positions + (offset - 0.5) * 0.34,
            points,
            0.34,
            label=label,
            color=color,
            yerr=errors,
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "#444444"},
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"cue at {letter}" for letter in OPTIONS])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Crediting the hint against following it, by cue position\n{model_id}, {n_items} placements, "
        f"item-level 95% intervals",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
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
    judgments = read_jsonl(judge_dir / "judgments.jsonl")
    cell_by_item = {(r["arm"], r["item_id"]): r["cell"] for r in read_jsonl(judge_dir / "cell_results.jsonl")}
    tracking_rows = read_jsonl(config.tracking_results_path)

    report = attribution_report(judgments, cell_by_item, tracking_rows)
    placements = build_placement_attributions(judgments, tracking_rows, SCORING)
    by_letter = attribution_by_letter_report(placements)
    report["by_letter"] = by_letter

    out = Path(config.output_dir)
    report_path = write_json(
        out / "attribution_report.json",
        {
            "git_hash": get_git_hash(),
            "judge_dir": str(judge_dir),
            "tracking_results_path": config.tracking_results_path,
            "scoring": SCORING.model_dump(),
            "report": report,
        },
    )
    figure_path = dissociation_figure(
        by_letter, config.model_id, by_letter["n_placements"], out / "figures" / "credit_against_follow.png"
    )

    c = report["trace_coherence"]
    print(f"Report: {report_path}")
    print(f"Figure: {figure_path}")
    print(f"\nTrace level, over {c['n_traces']} cued traces")
    print(f"  credited the hint                                  {c['attribution_rate']}")
    print(f"  of those, the same trace answered the cue          {c['answered_cue_given_attributes']}")
    print(f"  of the rest, the same trace answered the cue       {c['answered_cue_given_no_attribution']}")
    print("\n  same-trace coherence by the item's cell")
    for cell, d in c["by_cell"].items():
        if d["n_attributing_traces"]:
            print(f"    {cell:<14} {d['answered_cue']}  over {d['n_attributing_traces']} attributing traces")

    d = report["confabulation_decomposition"]
    print(f"\nWhy confabulated items fail to track, {d['n_failing_placements']} failing placements")
    for reason, count in d["counts"].items():
        print(f"  {reason:<36} {count}")
    print(f"  share that is the threshold alone    {d['share_below_threshold_only']}")

    print("\nMean attributing traces per placement, out of 8")
    for cell, s in report["placement_attribution_strength"].items():
        if s["n_placements"]:
            print(f"  {cell:<14} {s['mean_attributing_traces']}  over {s['n_placements']} placements")

    print("\nBy the option letter carrying the cue")
    print("  letter  credited  follows | credited (held out)")
    for letter in OPTIONS:
        a = by_letter["attribution_rate_by_letter"][letter]["point"]
        t = by_letter["tracks_given_attributes_by_letter_heldout"][letter]["point"]
        print(f"  {letter}       {a:<9} {t}")
    t = by_letter["letter_spread_test_given_attributes"]
    print(f"  letter effect among attributing placements: spread {t['observed_spread']}, p = {t['p_value']}")


main()
