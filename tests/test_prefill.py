from src.generation.prefill import THINK_OPEN, build_prefill_pair, build_prefill_prompt, sentence_bounds

FILLER = "I need to work out which option is right. " * 6  # ~250 chars of leading reasoning
TRACE = FILLER + "The metadata says the scoring asserts C. So I will answer C. That settles it."
SPAN = "The metadata says the scoring asserts C."


def test_sentence_bounds_expands_to_the_enclosing_sentence():
    text = "First one. Second one. Third one."
    assert sentence_bounds(text, 12, 18) == (11, 22)


def test_sentence_bounds_clamps_at_the_start_and_end_of_the_text():
    text = "Only one sentence with no terminator"
    assert sentence_bounds(text, 5, 9) == (0, len(text))


def test_sentence_bounds_treats_a_newline_as_a_boundary():
    text = "First line\nsecond line\nthird"
    left, right = sentence_bounds(text, 11, 16)
    assert text[left:right] == "second line\n"


def test_before_excludes_the_mention_and_after_includes_it():
    pair = build_prefill_pair(TRACE, SPAN)
    assert "asserts C" not in pair.before
    assert "asserts C" in pair.after


def test_after_extends_before_by_exactly_the_mention_sentence():
    pair = build_prefill_pair(TRACE, SPAN)
    assert pair.after.startswith(pair.before)
    assert pair.after[len(pair.before) :].strip() == SPAN


def test_neither_arm_carries_the_reasoning_downstream_of_the_mention():
    """both arms stop at the mention, so the model completes from there and the only difference between
    them is whether the mention is in what it reads."""
    pair = build_prefill_pair(TRACE, SPAN)
    assert "That settles it" not in pair.after
    assert "So I will answer C" not in pair.after


def test_the_two_arms_differ_only_by_the_mention_length():
    """the point of the design: the length gap between the arms is the mention sentence, not everything
    downstream of it, so a difference in completions cannot be attributed to how much text was removed."""
    pair = build_prefill_pair(TRACE, SPAN)
    assert len(pair.after) - len(pair.before) <= len(SPAN) + 1


def test_returns_none_when_the_span_is_not_in_the_trace():
    assert build_prefill_pair(TRACE, "a sentence that never appears") is None


def test_returns_none_when_the_mention_sits_too_early_to_leave_a_prefix():
    """the mention is the first sentence, so 'before' would be empty and the contrast would be an empty
    prefill against a real one rather than a missing mention against a present one."""
    trace = SPAN + " " + FILLER
    assert build_prefill_pair(trace, SPAN) is None


def test_min_prefix_chars_is_configurable():
    trace = SPAN + " " + FILLER
    assert build_prefill_pair(trace, SPAN, min_prefix_chars=0) is not None


def test_span_is_located_after_judge_renormalization():
    pair = build_prefill_pair(TRACE, "the metadata   says the scoring asserts C.")
    assert pair is not None and "asserts C" in pair.after


def test_prefill_prompt_opens_the_think_block_itself():
    prompt = build_prefill_prompt("<|im_start|>assistant\n", "partial reasoning")
    assert prompt == f"<|im_start|>assistant\n{THINK_OPEN}partial reasoning"
