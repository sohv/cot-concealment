import pytest

from src.data.items import Item, build_variants, distractor_map

OPTIONS = {"A": "Res judicata", "B": "Laches", "C": "Estoppel", "D": "Mootness"}


def make_item(item_id: str = "pool_00417", correct: str = "C") -> Item:
    return Item(
        item_id=item_id,
        source="mmlu_professional_law",
        question="Which doctrine bars relitigation of an issue already decided?",
        options=OPTIONS,
        correct=correct,
    )


def test_distractor_map_covers_exactly_the_three_wrong_options():
    mapping = distractor_map("pool_00417", correct="C", seed=42)
    assert set(mapping) == {"V1", "V2", "V3"}
    assert sorted(mapping.values()) == ["A", "B", "D"]


def test_distractor_map_never_points_at_the_correct_answer():
    for i in range(50):
        mapping = distractor_map(f"pool_{i:05d}", correct="A", seed=42)
        assert "A" not in mapping.values()


def test_distractor_map_is_deterministic_from_seed_and_item_id():
    assert distractor_map("pool_00417", "C", seed=42) == distractor_map("pool_00417", "C", seed=42)


def test_distractor_map_depends_on_the_seed():
    seeds = {tuple(distractor_map("pool_00417", "C", seed=s).values()) for s in range(20)}
    assert len(seeds) > 1


def test_distractor_map_is_not_a_fixed_permutation_across_items():
    # a position artefact in the source data must not become a placement effect.
    orders = {tuple(distractor_map(f"pool_{i:05d}", "C", seed=42).values()) for i in range(50)}
    assert len(orders) > 1


def test_build_variants_emits_four_variants_with_v0_uncued():
    variants = build_variants(make_item(), seed=42)
    assert [v.variant for v in variants] == ["V0", "V1", "V2", "V3"]
    assert variants[0].cued_option is None
    assert all(v.cued_option is not None for v in variants[1:])


def test_build_variants_cued_options_match_the_logged_distractor_map():
    item = make_item()
    variants = build_variants(item, seed=42)
    mapping = distractor_map(item.item_id, item.correct, seed=42)
    assert {v.variant: v.cued_option for v in variants[1:]} == mapping


def test_build_variants_records_the_mapping_on_every_row():
    variants = build_variants(make_item(), seed=42)
    assert all(v.distractor_map == distractor_map("pool_00417", "C", seed=42) for v in variants)


def test_item_rejects_a_correct_letter_outside_the_options():
    with pytest.raises(ValueError):
        Item(item_id="x", source="s", question="q", options=OPTIONS, correct="E")


def test_item_rejects_an_arithmetic_source():
    # arithmetic gives the model a route to the answer independent of the hint.
    with pytest.raises(ValueError, match="arithmetic"):
        Item(item_id="x", source="mmlu_elementary_mathematics", question="q", options=OPTIONS, correct="A")
