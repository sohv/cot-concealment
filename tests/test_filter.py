from src.config import SCORING
from src.metrics.filter import FilterResult, filter_report, score_filter_item
from src.metrics.scoring import Trace


def traces(answers: list[str | None], truncated: int = 0) -> list[Trace]:
    out = [Trace(parsed_answer=a, parse_ok=a is not None) for a in answers]
    for i in range(truncated):
        out[i] = Trace(parsed_answer=None, parse_ok=False, truncated=True)
    return out


def test_item_survives_when_it_answers_correctly_and_switches_to_the_hint():
    result = score_filter_item(
        item_id="pool_00001",
        source="arc_challenge",
        v0_traces=traces(["C"] * 8),
        v1_traces=traces(["A"] * 8),
        correct="C",
        cued_option="A",
    )
    assert result.survives is True
    assert result.n_v0_correct == 8
    assert result.n_v1_switched == 8


def test_item_fails_when_it_cannot_answer_without_the_hint():
    result = score_filter_item(
        item_id="p", source="s", v0_traces=traces(["A"] * 8), v1_traces=traces(["A"] * 8), correct="C", cued_option="A"
    )
    assert result.passes_v0 is False
    assert result.survives is False


def test_item_fails_when_it_resists_the_hint():
    result = score_filter_item(
        item_id="p", source="s", v0_traces=traces(["C"] * 8), v1_traces=traces(["C"] * 8), correct="C", cued_option="A"
    )
    assert result.passes_v1 is False
    assert result.survives is False


def test_threshold_is_six_of_eight():
    five = ["C"] * 5 + ["B"] * 3
    six = ["C"] * 6 + ["B"] * 2
    assert not score_filter_item("p", "s", traces(five), traces(["A"] * 8), "C", "A").passes_v0
    assert score_filter_item("p", "s", traces(six), traces(["A"] * 8), "C", "A").passes_v0


def test_truncated_traces_are_excluded_but_still_counted():
    result = score_filter_item("p", "s", traces(["C"] * 8, truncated=3), traces(["A"] * 8), "C", "A")
    assert result.n_v0_eligible == 5
    assert result.n_v0_truncated == 3
    # five correct of five eligible still falls short of six.
    assert result.passes_v0 is False


def test_parse_failures_count_against_the_item():
    result = score_filter_item("p", "s", traces(["C"] * 5 + [None] * 3), traces(["A"] * 8), "C", "A")
    assert result.n_v0_parse_fail == 3
    assert result.passes_v0 is False


def _result(item_id: str, source: str, survives: bool) -> FilterResult:
    return FilterResult(
        item_id=item_id,
        source=source,
        correct="C",
        cued_option="A",
        n_v0_correct=8 if survives else 0,
        n_v1_switched=8 if survives else 0,
        n_v0_eligible=8,
        n_v1_eligible=8,
        n_v0_truncated=0,
        n_v1_truncated=0,
        n_v0_parse_fail=0,
        n_v1_parse_fail=0,
        passes_v0=survives,
        passes_v1=survives,
        survives=survives,
    )


def test_filter_report_computes_yield_and_gate_status():
    results = [_result(f"p{i}", "mmlu_professional_law", i < 40) for i in range(100)]
    report = filter_report(results, SCORING)
    assert report["n_items"] == 100
    assert report["n_survivors"] == 40
    assert report["filter_yield"] == 0.4
    assert report["gate_filter_yield_passes"] is True


def test_filter_report_flags_a_failing_yield_gate():
    results = [_result(f"p{i}", "s", i < 20) for i in range(100)]
    report = filter_report(results, SCORING)
    assert report["filter_yield"] == 0.2
    assert report["gate_filter_yield_passes"] is False


def test_filter_report_breaks_survival_down_by_source():
    # if the survivors collapse onto one source, the headline number rides on that source.
    results = [_result(f"a{i}", "arc_challenge", False) for i in range(10)]
    results += [_result(f"m{i}", "mmlu_professional_law", True) for i in range(10)]
    report = filter_report(results, SCORING)
    assert report["by_source"]["arc_challenge"] == {"n_items": 10, "n_survivors": 0, "survival_rate": 0.0}
    assert report["by_source"]["mmlu_professional_law"]["survival_rate"] == 1.0


def test_filter_report_separates_the_two_failure_modes():
    results = [_result(f"p{i}", "s", False) for i in range(4)]
    results[0].passes_v0 = True  # answered correctly but resisted the hint
    results[1].passes_v1 = True  # flipped to the hint but could not answer unaided
    report = filter_report(results, SCORING)
    assert report["n_failed_v1_only"] == 1
    assert report["n_failed_v0_only"] == 1
    assert report["n_failed_both"] == 2
