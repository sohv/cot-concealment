import pytest

from src.data.items import is_excluded_source
from src.data.pool import assemble_pool, normalize_arc_row, normalize_mmlu_row

MMLU_ROW = {
    "question": "Which doctrine bars relitigation of an issue already decided?",
    "subject": "professional_law",
    "choices": ["Res judicata", "Laches", "Estoppel", "Mootness"],
    "answer": 0,
}

ARC_ROW = {
    "id": "Mercury_7175875",
    "question": "Which property of a mineral can be determined just by looking at it?",
    "choices": {"text": ["luster", "mass", "weight", "hardness"], "label": ["A", "B", "C", "D"]},
    "answerKey": "A",
}

ARC_NUMERIC_LABELS = {
    "id": "Mercury_400",
    "question": "What is a producer in a food chain?",
    "choices": {"text": ["a plant", "a fox", "a hawk", "a wolf"], "label": ["1", "2", "3", "4"]},
    "answerKey": "1",
}


def test_normalize_mmlu_row_maps_answer_index_to_a_letter():
    item = normalize_mmlu_row(MMLU_ROW, item_id="pool_00001")
    assert item.correct == "A"
    assert item.options == {"A": "Res judicata", "B": "Laches", "C": "Estoppel", "D": "Mootness"}
    assert item.source == "mmlu_professional_law"


def test_normalize_mmlu_row_handles_a_later_answer_index():
    item = normalize_mmlu_row(MMLU_ROW | {"answer": 3}, item_id="pool_00001")
    assert item.correct == "D"


def test_normalize_arc_row_maps_letter_labels():
    item = normalize_arc_row(ARC_ROW, item_id="pool_00002")
    assert item.correct == "A"
    assert item.options["D"] == "hardness"
    assert item.source == "arc_challenge"


def test_normalize_arc_row_maps_numeric_labels_by_position():
    # arc mixes "1".."4" and "A".."D" label schemes across rows.
    item = normalize_arc_row(ARC_NUMERIC_LABELS, item_id="pool_00003")
    assert item.correct == "A"
    assert item.options["B"] == "a fox"


def test_normalize_arc_row_rejects_rows_without_exactly_four_options():
    three = {
        "id": "x",
        "question": "q",
        "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]},
        "answerKey": "A",
    }
    assert normalize_arc_row(three, item_id="pool_00004") is None


def test_normalize_arc_row_rejects_an_answer_key_outside_the_labels():
    bad = ARC_ROW | {"answerKey": "E"}
    assert normalize_arc_row(bad, item_id="pool_00005") is None


def test_normalize_mmlu_row_rejects_an_excluded_subject():
    assert normalize_mmlu_row(MMLU_ROW | {"subject": "elementary_mathematics"}, item_id="p") is None


@pytest.mark.parametrize(
    "source",
    [
        "mmlu_elementary_mathematics",
        "mmlu_college_physics",
        "mmlu_econometrics",
        "mmlu_high_school_chemistry",
        "mmlu_astronomy",
    ],
)
def test_computational_sources_are_excluded(source):
    assert is_excluded_source(source)


@pytest.mark.parametrize("source", ["mmlu_professional_law", "mmlu_world_religions", "arc_challenge"])
def test_non_computational_sources_are_kept(source):
    assert not is_excluded_source(source)


def test_assemble_pool_assigns_stable_sequential_ids():
    rows = [(MMLU_ROW | {"question": f"q{i}"}, "mmlu") for i in range(5)]
    pool = assemble_pool(rows, n_items=5, seed=42)
    assert [it.item_id for it in pool] == [f"pool_{i:05d}" for i in range(5)]


def test_assemble_pool_deduplicates_on_question_text():
    rows = [(MMLU_ROW, "mmlu"), (MMLU_ROW, "mmlu"), (MMLU_ROW | {"question": "other"}, "mmlu")]
    pool = assemble_pool(rows, n_items=10, seed=42)
    assert len(pool) == 2


def test_assemble_pool_is_deterministic_and_respects_the_cap():
    rows = [(MMLU_ROW | {"question": f"q{i}"}, "mmlu") for i in range(50)]
    first = assemble_pool(rows, n_items=10, seed=42)
    second = assemble_pool(rows, n_items=10, seed=42)
    assert len(first) == 10
    assert [it.question for it in first] == [it.question for it in second]


def test_assemble_pool_shuffles_before_capping():
    rows = [(MMLU_ROW | {"question": f"q{i}"}, "mmlu") for i in range(50)]
    assert [it.question for it in assemble_pool(rows, n_items=10, seed=1)] != [
        it.question for it in assemble_pool(rows, n_items=10, seed=2)
    ]
