import pytest

from src.metrics.agreement import FIELDS, cohens_kappa, score_agreement


def key_row(row_id: str, mentions: bool, named: str | None, attributes: bool) -> dict:
    return {
        "id": row_id,
        "mentions_cue": mentions,
        "named_option": named,
        "attributes_answer_to_cue": attributes,
    }


def label_row(row_id: str, mentions: bool, named: str | None, attributes: bool) -> dict:
    return key_row(row_id, mentions, named, attributes)


def test_cohens_kappa_is_one_on_perfect_agreement():
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0


def test_cohens_kappa_is_zero_at_chance():
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_cohens_kappa_is_none_when_chance_agreement_is_total():
    """both raters used one label throughout, so there is no room above chance to measure."""
    assert cohens_kappa(["a", "a", "a"], ["a", "a", "a"]) is None


def test_cohens_kappa_is_none_on_no_rows():
    assert cohens_kappa([], []) is None


def test_score_agreement_reports_every_field():
    key = [key_row("hc_000", True, "B", True), key_row("hc_001", False, None, False)]
    labels = [label_row("hc_000", True, "B", True), label_row("hc_001", False, None, False)]
    result = score_agreement(key, labels)
    assert set(result) == set(FIELDS)
    assert all(result[f]["agreement"] == 1.0 for f in FIELDS)


def test_score_agreement_lists_the_disagreements():
    key = [key_row("hc_000", True, "B", True), key_row("hc_001", True, "C", True)]
    labels = [label_row("hc_000", True, "B", True), label_row("hc_001", True, "D", True)]
    result = score_agreement(key, labels)
    assert result["named_option"]["agreement"] == 0.5
    assert result["named_option"]["disagreements"] == [{"id": "hc_001", "judge": "C", "human": "D"}]


def test_score_agreement_pabak_beats_kappa_on_a_skewed_field():
    """19 of 20 traces mention the cue and the raters differ on one. agreement is 0.95 and PABAK 0.90,
    but kappa collapses because the manufactured prevalence leaves almost no chance-corrected room."""
    key = [key_row(f"hc_{i:03d}", True, "B", True) for i in range(20)]
    labels = [label_row(f"hc_{i:03d}", i != 0, "B", True) for i in range(20)]
    result = score_agreement(key, labels)["mentions_cue"]
    assert result["agreement"] == 0.95
    assert result["pabak"] == 0.9
    assert result["cohens_kappa"] == 0.0


def test_score_agreement_scores_only_the_rows_that_were_labelled():
    key = [key_row(f"hc_{i:03d}", True, "B", True) for i in range(5)]
    labels = [{"id": "hc_000", "mentions_cue": True}]
    result = score_agreement(key, labels)
    assert result["mentions_cue"]["n"] == 1
    assert "named_option" not in result


def test_score_agreement_rejects_a_label_that_is_not_in_the_key():
    key = [key_row("hc_000", True, "B", True)]
    with pytest.raises(ValueError, match="not in the key"):
        score_agreement(key, [label_row("hc_999", True, "B", True)])
