import numpy as np
import pytest

from src.config import CUED_VARIANTS, SCORING
from src.metrics.placement import (
    Observation,
    build_observations,
    letter_spread_test,
    placement_count_distribution,
    placement_report,
    rates_by,
)

DISTRACTOR_MAP = {"V1": "A", "V2": "B", "V3": "D"}


def variant_rows(item_ids: list[str], distractor_map: dict | None = None, correct: str = "C") -> list[dict]:
    return [
        {"item_id": i, "variant": v, "distractor_map": distractor_map or DISTRACTOR_MAP, "correct": correct}
        for i in item_ids
        for v in ("V0",) + CUED_VARIANTS
    ]


def tracking_row(item_id: str, arm: str, tracks: dict[str, bool]) -> dict:
    return {"arm": arm, "item_id": item_id, **{f"tracks_{v}": tracks[v] for v in CUED_VARIANTS}}


def obs_from(specs: list[tuple[str, str, dict]], distractor_map: dict | None = None) -> list[Observation]:
    item_ids = sorted({item_id for item_id, _, _ in specs})
    rows = [tracking_row(item_id, arm, tracks) for item_id, arm, tracks in specs]
    return build_observations(rows, variant_rows(item_ids, distractor_map))


def test_build_observations_expands_one_row_per_placement():
    obs = obs_from([("pool_0", "C0_bare", {"V1": True, "V2": False, "V3": True})])
    assert len(obs) == 3
    assert [o.variant for o in obs] == list(CUED_VARIANTS)
    assert [o.tracks for o in obs] == [True, False, True]


def test_build_observations_attaches_the_cue_letter_from_the_distractor_map():
    obs = obs_from([("pool_0", "C0_bare", dict.fromkeys(CUED_VARIANTS, True))])
    assert {o.variant: o.cued_option for o in obs} == DISTRACTOR_MAP


def test_build_observations_never_attaches_the_correct_letter_as_the_cue():
    obs = obs_from([("pool_0", "C0_bare", dict.fromkeys(CUED_VARIANTS, True))])
    assert all(o.cued_option != o.correct for o in obs)


def test_rates_by_variant_recovers_the_hand_computed_rate():
    specs = [
        ("pool_0", "C0_bare", {"V1": True, "V2": True, "V3": False}),
        ("pool_1", "C0_bare", {"V1": True, "V2": False, "V3": False}),
    ]
    rates = rates_by(obs_from(specs), CUED_VARIANTS, lambda o: o.variant)
    assert rates["V1"]["point"] == 1.0
    assert rates["V2"]["point"] == 0.5
    assert rates["V3"]["point"] == 0.0
    assert rates["V1"]["n_observations"] == 2


def test_rates_by_brackets_the_point_estimate_in_its_interval():
    specs = [(f"pool_{i}", "C0_bare", {"V1": True, "V2": i % 2 == 0, "V3": False}) for i in range(30)]
    rates = rates_by(obs_from(specs), CUED_VARIANTS, lambda o: o.variant)
    low, high = rates["V2"]["ci"]
    assert low <= rates["V2"]["point"] <= high


def test_rates_by_is_deterministic_under_a_fixed_seed():
    specs = [(f"pool_{i}", "C0_bare", {"V1": True, "V2": i % 3 == 0, "V3": i % 2 == 0}) for i in range(20)]
    obs = obs_from(specs)
    assert rates_by(obs, CUED_VARIANTS, lambda o: o.variant) == rates_by(obs, CUED_VARIANTS, lambda o: o.variant)


def test_rates_by_resamples_items_not_placements():
    """both sets carry the same 0.5 rate and the same outcome per row; only the number of items differs.
    a placement level bootstrap would see 12 draws either way and return the same width."""
    few_items = [(f"pool_{i}", "C0_bare", dict.fromkeys(CUED_VARIANTS, i % 2 == 0)) for i in range(4)]
    many_items = [(f"pool_{i}", "C0_bare", dict.fromkeys(CUED_VARIANTS, i % 2 == 0)) for i in range(40)]
    wide = rates_by(obs_from(few_items), CUED_VARIANTS, lambda o: o.variant)["V1"]["ci"]
    narrow = rates_by(obs_from(many_items), CUED_VARIANTS, lambda o: o.variant)["V1"]["ci"]
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_rates_by_reports_an_unobserved_level_as_empty():
    obs = obs_from([("pool_0", "C0_bare", dict.fromkeys(CUED_VARIANTS, True))])
    rates = rates_by(obs, ("A", "B", "C", "D"), lambda o: o.cued_option)
    assert rates["C"]["n_observations"] == 0
    assert rates["C"]["point"] is None


def test_letter_spread_test_finds_no_effect_when_outcomes_ignore_the_letter():
    rng = np.random.default_rng(0)
    specs = [(f"pool_{i}", "C0_bare", {v: bool(rng.integers(0, 2)) for v in CUED_VARIANTS}) for i in range(120)]
    assert letter_spread_test(obs_from(specs), CUED_VARIANTS, n_permutations=200)["p_value"] > 0.05


def test_letter_spread_test_finds_an_effect_when_one_letter_never_tracks():
    specs = [(f"pool_{i}", "C0_bare", {"V1": True, "V2": True, "V3": False}) for i in range(120)]
    result = letter_spread_test(obs_from(specs), CUED_VARIANTS, n_permutations=200)
    assert result["observed_spread"] == 1.0
    assert result["p_value"] < 0.05


def test_letter_spread_test_runs_on_the_heldout_pair():
    specs = [(f"pool_{i}", "C0_bare", {"V1": True, "V2": i % 2 == 0, "V3": False}) for i in range(40)]
    result = letter_spread_test(obs_from(specs), SCORING.heldout_variants, n_permutations=100)
    assert result["n_groups"] == 40
    assert result["variants"] == list(SCORING.heldout_variants)


def test_placement_count_distribution_matches_the_binomial_when_placements_are_independent():
    rng = np.random.default_rng(1)
    specs = [(f"pool_{i}", "C0_bare", {v: bool(rng.integers(0, 2)) for v in CUED_VARIANTS}) for i in range(400)]
    assert placement_count_distribution(obs_from(specs))["p_value"] > 0.05


def test_placement_count_distribution_detects_item_level_clumping():
    """every item either tracks everywhere or nowhere. the mean rate is 0.5 either way, so only the
    shape of the distribution separates this from a coin flip per placement."""
    specs = [(f"pool_{i}", "C0_bare", dict.fromkeys(CUED_VARIANTS, i % 2 == 0)) for i in range(120)]
    result = placement_count_distribution(obs_from(specs))
    assert result["observed"] == [60, 0, 0, 60]
    assert result["p_value"] < 0.001
    assert result["observed_all"] == 0.5
    assert result["independence_prediction_all"] == 0.125


def test_placement_report_covers_every_breakdown():
    specs = [
        (f"pool_{i}", arm, {"V1": True, "V2": i % 2 == 0, "V3": i % 3 == 0})
        for i in range(20)
        for arm in ("C0_bare", "C3_neutral_private")
    ]
    report = placement_report(obs_from(specs), ["C0_bare", "C3_neutral_private"])
    assert report["n_observations"] == 120
    assert report["n_items"] == 20
    assert set(report["by_variant_within_arm"]) == {"C0_bare", "C3_neutral_private"}
    assert report["by_cued_option_heldout"]["A"]["n_observations"] == 0


def test_letter_spread_test_treats_the_item_as_the_unit_not_the_arm_replicate():
    """an item's distractor map is keyed on the item alone, so its arm replicates carry identical letters
    and correlated outcomes. duplicating an arm adds no independent information and must not shrink the
    p value; permuting per row rather than per item is what made this anti-conservative."""
    from src.metrics.placement import Observation, letter_spread_test

    def build(arms: list[str]) -> list[Observation]:
        obs = []
        for i in range(24):
            letters = ["B", "C", "A"] if i % 2 else ["C", "B", "D"]
            for arm in arms:
                for variant, letter in zip(CUED_VARIANTS, letters):
                    obs.append(
                        Observation(
                            arm=arm,
                            item_id=f"i{i}",
                            variant=variant,
                            cued_option=letter,
                            correct="A" if letter != "A" else "B",
                            tracks=letter in ("B", "C"),
                        )
                    )
        return obs

    one = letter_spread_test(build(["C0_bare"]), n_permutations=400)
    three = letter_spread_test(build(["C0_bare", "C3_neutral_private", "C2_monitored"]), n_permutations=400)

    assert one["n_items"] == three["n_items"] == 24
    assert three["n_groups"] == 3 * one["n_groups"]
    # the observed statistic is unchanged by duplication, and so is the null it is judged against.
    assert one["observed_spread"] == three["observed_spread"]
    assert abs(one["permuted_spread_q95"] - three["permuted_spread_q95"]) < 0.05
    assert three["p_value"] >= one["p_value"] * 0.5
