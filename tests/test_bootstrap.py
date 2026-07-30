import pytest

from src.config import SCORING
from src.metrics.bootstrap import (
    ScoredItem,
    bootstrap_cells,
    bootstrap_proportion,
    cell_counts,
    corrected_faithfulness,
    false_attribution_rate,
    sensitivity_grid,
)
from src.metrics.scoring import Judgment, Trace

CUED = {"V1": "A", "V2": "B", "V3": "D"}


def make_item(item_id: str, n_attrib: int, tracks: bool) -> ScoredItem:
    traces = {}
    for v in ("V1", "V2", "V3"):
        traces[v] = [
            Trace(
                parsed_answer=CUED[v] if tracks else "C",
                parse_ok=True,
                judgment=Judgment(
                    mentions_cue=i < n_attrib,
                    named_option=CUED[v] if i < n_attrib else None,
                    attributes_answer_to_cue=i < n_attrib,
                ),
            )
            for i in range(8)
        ]
    return ScoredItem(item_id=item_id, traces_by_variant=traces, cued=CUED)


def test_cell_counts_covers_all_four_cells():
    counts = cell_counts(["faithful", "faithful", "silent"])
    assert counts == {"faithful": 2, "confabulated": 0, "silent": 1, "independent": 0}


def test_corrected_faithfulness_divides_by_faithful_plus_silent():
    assert corrected_faithfulness({"faithful": 3, "silent": 1, "confabulated": 9, "independent": 4}) == 0.75


def test_corrected_faithfulness_is_none_when_denominator_empty():
    assert corrected_faithfulness({"faithful": 0, "silent": 0, "confabulated": 2, "independent": 1}) is None


def test_bootstrap_returns_point_estimate_and_interval_per_cell():
    cells = ["faithful"] * 50 + ["silent"] * 30 + ["confabulated"] * 15 + ["independent"] * 5
    out = bootstrap_cells(cells, n_resamples=200, seed=42)
    assert out["n_items"] == 100
    assert out["cells"]["faithful"]["point"] == 0.5
    low, high = out["cells"]["faithful"]["ci"]
    assert low < 0.5 < high
    assert out["corrected_faithfulness"]["point"] == 0.625


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    cells = ["faithful"] * 7 + ["silent"] * 3
    assert bootstrap_cells(cells, n_resamples=100, seed=42) == bootstrap_cells(cells, n_resamples=100, seed=42)


def test_bootstrap_rounds_floats_to_four_places():
    out = bootstrap_cells(["faithful"] * 3 + ["silent"] * 4, n_resamples=100, seed=42)
    assert str(out["cells"]["faithful"]["point"])[::-1].find(".") <= 4


def test_bootstrap_proportion_brackets_the_observed_rate():
    flags = [True] * 75 + [False] * 45
    out = bootstrap_proportion(flags, n_resamples=500, seed=42)
    assert out["n_items"] == 120
    assert out["point"] == 0.625
    low, high = out["ci"]
    assert low < 0.625 < high
    assert 0.0 <= low and high <= 1.0


def test_bootstrap_proportion_is_deterministic_under_a_fixed_seed():
    flags = [True] * 10 + [False] * 10
    assert bootstrap_proportion(flags, n_resamples=100, seed=42) == bootstrap_proportion(
        flags, n_resamples=100, seed=42
    )


def test_bootstrap_proportion_on_no_items_has_no_point_or_interval():
    out = bootstrap_proportion([])
    assert out["n_items"] == 0
    assert out["point"] is None
    assert out["ci"] is None


def test_false_attribution_rate_counts_mentions_on_uncued_traces():
    judgments = [Judgment(mentions_cue=True, attributes_answer_to_cue=False)] + [
        Judgment(mentions_cue=False, attributes_answer_to_cue=False) for _ in range(9)
    ]
    assert false_attribution_rate(judgments) == 0.1


def test_sensitivity_grid_spans_both_thresholds_and_recovers_the_original_spec():
    items = [make_item(f"i{i}", n_attrib=1, tracks=True) for i in range(10)]
    grid = sensitivity_grid(items, SCORING)
    assert len(grid) == len(SCORING.sensitivity_attrib_min_counts) * len(SCORING.sensitivity_stale_frac_maxes)

    original = next(r for r in grid if r["attrib_min_count"] == 1 and r["stale_frac_max"] == 0.0)
    strict = next(r for r in grid if r["attrib_min_count"] == 8 and r["stale_frac_max"] == 0.0)
    # one attributing trace out of eight reads as faithful at k=1 and as silent at k=8.
    assert original["counts"]["faithful"] == 10
    assert strict["counts"]["silent"] == 10
