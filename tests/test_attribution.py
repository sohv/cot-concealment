from src.metrics.attribution import (
    confabulation_decomposition,
    placement_attribution_strength,
    trace_coherence,
)

ARM = "C0_bare"
CUED = {"V1": "A", "V2": "B", "V3": "D"}


def judgment(item: str, variant: str, idx: int, *, attributes: bool, answer: str, ok: bool = True) -> dict:
    return {
        "item_id": item,
        "arm": ARM,
        "variant": variant,
        "sample_idx": idx,
        "cued_option": CUED[variant],
        "parsed_answer": answer,
        "judge_ok": ok,
        "attributes_answer_to_cue": attributes,
    }


def tracking_row(item: str, tracks: dict[str, bool], modal: dict[str, str], counts: dict[str, int]) -> dict:
    return {
        "arm": ARM,
        "item_id": item,
        **{f"tracks_{v}": tracks[v] for v in CUED},
        **{f"modal_{v}": modal[v] for v in CUED},
        **{f"modal_count_{v}": counts[v] for v in CUED},
    }


def test_trace_coherence_separates_attributing_from_non_attributing():
    judgments = [judgment("i0", "V1", i, attributes=i < 4, answer="A" if i < 4 else "C") for i in range(8)]
    result = trace_coherence(judgments, {(ARM, "i0"): "faithful"})
    assert result["attribution_rate"] == 0.5
    assert result["answered_cue_given_attributes"] == 1.0
    assert result["answered_cue_given_no_attribution"] == 0.0


def test_trace_coherence_catches_a_trace_that_credits_a_hint_it_did_not_follow():
    judgments = [judgment("i0", "V1", i, attributes=True, answer="A" if i else "C") for i in range(8)]
    result = trace_coherence(judgments, {(ARM, "i0"): "confabulated"})
    assert result["answered_cue_given_attributes"] == 0.875
    assert result["by_cell"]["confabulated"]["answered_cue"] == 0.875


def test_trace_coherence_ignores_v0_and_unparsed_judgments():
    judgments = [
        judgment("i0", "V1", 0, attributes=True, answer="A"),
        judgment("i0", "V1", 1, attributes=True, answer="A", ok=False),
        {**judgment("i0", "V1", 2, attributes=True, answer="A"), "variant": "V0"},
    ]
    assert trace_coherence(judgments, {(ARM, "i0"): "faithful"})["n_traces"] == 1


def test_confabulation_decomposition_splits_threshold_misses_from_real_misses():
    """a placement whose modal answer is the cue but sits below the 6-of-8 bar is the threshold biting;
    a placement whose modal answer is another option is the model genuinely not following the hint."""
    judgments = [judgment("i0", v, 0, attributes=True, answer=CUED[v]) for v in CUED]
    tracking = [
        tracking_row(
            "i0",
            tracks={"V1": True, "V2": False, "V3": False},
            modal={"V1": "A", "V2": "B", "V3": "C"},
            counts={"V1": 8, "V2": 5, "V3": 7},
        )
    ]
    result = confabulation_decomposition({(ARM, "i0"): "confabulated"}, tracking, judgments)
    assert result["counts"] == {"modal_is_the_cue_below_threshold": 1, "modal_is_a_different_option": 1}
    assert result["share_below_threshold_only"] == 0.5


def test_confabulation_decomposition_ignores_items_in_other_cells():
    judgments = [judgment("i0", v, 0, attributes=True, answer=CUED[v]) for v in CUED]
    tracking = [
        tracking_row(
            "i0",
            tracks={"V1": True, "V2": False, "V3": False},
            modal={"V1": "A", "V2": "B", "V3": "C"},
            counts={"V1": 8, "V2": 5, "V3": 7},
        )
    ]
    result = confabulation_decomposition({(ARM, "i0"): "faithful"}, tracking, judgments)
    assert result["n_failing_placements"] == 0


def test_placement_attribution_strength_counts_attributing_traces_per_placement():
    judgments = [judgment("i0", "V1", i, attributes=i < 6, answer="A") for i in range(8)]
    judgments += [judgment("i1", "V1", i, attributes=i < 2, answer="A") for i in range(8)]
    result = placement_attribution_strength(judgments, {(ARM, "i0"): "faithful", (ARM, "i1"): "confabulated"})
    assert result["faithful"]["mean_attributing_traces"] == 6.0
    assert result["confabulated"]["mean_attributing_traces"] == 2.0
