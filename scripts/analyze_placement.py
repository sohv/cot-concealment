# re-aggregates the finished sweep by placement: variant index, the option letter carrying the cue, and
# how many of an item's three placements track. reads existing rows only, no engine and no gpu.
# uv run -m scripts.analyze_placement --tracking_results_path results/raw/sweep_v1_resume/tracking_results.jsonl --variants_path data/processed/variants_seed42.jsonl --output_dir results/analysis/placement_v1 --seed 42

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.config import CUED_VARIANTS, OPTIONS, SCORING
from src.metrics.placement import EDGE_OPTIONS, build_observations, placement_report
from src.utils.io import get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    tracking_results_path: str = ""
    variants_path: str = ""
    output_dir: str = "results/analysis/placement_v1"
    seed: int = 42


def print_rates(heading: str, rates: dict) -> None:
    print(heading)
    for level, d in rates.items():
        ci = d["ci"]
        interval = "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
        print(f"  {level}  {d['point']:.3f}{interval}  n={d['n_observations']}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = simple_parsing.parse(Config)
    if not config.tracking_results_path:
        raise ValueError("--tracking_results_path is required")
    if not config.variants_path:
        raise ValueError("--variants_path is required")

    tracking_rows = read_jsonl(config.tracking_results_path)
    variant_rows = read_jsonl(config.variants_path)
    arms = sorted({r["arm"] for r in tracking_rows})
    obs = build_observations(tracking_rows, variant_rows)
    LOGGER.info(f"{len(obs)} placement observations over {len({o.item_id for o in obs})} items and {len(arms)} arms")

    report = placement_report(obs, arms)
    report_path = write_json(
        Path(config.output_dir) / "placement_report.json",
        {
            "git_hash": get_git_hash(),
            "tracking_results_path": config.tracking_results_path,
            "variants_path": config.variants_path,
            "seed": config.seed,
            "report": report,
        },
    )

    print(f"Report: {report_path}")
    print(f"{report['n_observations']} placements over {report['n_items']} items and {len(arms)} arms")
    print_rates("\nTracking by placement, pooled over arms", report["by_variant"])
    print_rates("\nTracking by option letter carrying the cue", report["by_cued_option"])
    print_rates(
        f"\nTracking by option letter, held out placements only {list(SCORING.heldout_variants)}",
        report["by_cued_option_heldout"],
    )
    print_rates("\nTracking by the item's correct option letter", report["by_correct_option"])

    for label, key in [("all placements", "letter_spread_test"), ("held out only", "letter_spread_test_heldout")]:
        t = report[key]
        print(
            f"\nLetter effect, {label}: observed spread {t['observed_spread']:.3f}, "
            f"permuted mean {t['permuted_spread_mean']:.3f}, 95th pct {t['permuted_spread_q95']:.3f}, p = {t['p_value']:.4f}"
        )

    d = report["placement_count_distribution"]
    print(f"\nPlacements tracking per item and arm, {len(CUED_VARIANTS)} placements, n = {d['n_groups']} pairs")
    print("  k  observed  binomial")
    for k, (o, e) in enumerate(zip(d["observed"], d["expected_binomial"])):
        print(f"  {k}  {o:>8}  {e:>8.1f}")
    print(
        f"  chi2 {d['chi2']:.1f}, p = {d['p_value']:.2e}; all {len(CUED_VARIANTS)} track {d['observed_all']:.3f} "
        f"against {d['independence_prediction_all']:.3f} if placements were independent"
    )

    c = report["compound_tracking_by_item_group"]
    print(f"\nTracks at every placement, by the item's correct letter (post hoc, weak letters {list(EDGE_OPTIONS)})")
    for letter, d in c["by_correct_option"].items():
        forced = len([e for e in EDGE_OPTIONS if e != letter])
        print(f"  correct {letter}  {d['tracks_all']}  n={d['n_groups']}  cue forced onto {forced} weak letters")
    print("\nTracks at every placement, by how many placements land on a weak letter")
    for n, d in c["by_n_edge_placements"].items():
        if d["n_groups"]:
            print(f"  {n} weak placements  {d['tracks_all']}  n={d['n_groups']}")


main()
