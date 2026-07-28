import pytest

from src.generation.parsing import find_span, normalize_span, parse_answer, split_trace

THINK_END = 151668


class StubTokenizer:
    """decodes each int as a single char via a fixed vocab, so parsing is testable without a gpu."""

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.vocab[t] for t in token_ids)


@pytest.fixture
def tok():
    vocab = {i: chr(ord("a") + i) for i in range(26)}
    vocab[THINK_END] = "</think>"
    vocab[100] = " "
    return StubTokenizer(vocab)


def test_split_trace_splits_on_terminator(tok):
    think, content, truncated = split_trace([0, 1, 2, THINK_END, 3, 4], tok)
    assert think == "abc"
    assert content == "de"
    assert truncated is False


def test_split_trace_excludes_terminator_from_both_sides(tok):
    think, content, _ = split_trace([0, 1, THINK_END, 2], tok)
    assert "</think>" not in think
    assert "</think>" not in content


def test_split_trace_uses_first_terminator_not_last(tok):
    # a trace that echoes the terminator inside the content block must not pull content into think.
    think, content, _ = split_trace([0, THINK_END, 1, THINK_END, 2], tok)
    assert think == "a"
    assert content.startswith("b")


def test_split_trace_strips_the_model_emitted_open_tag(tok):
    # confirmed on real generations: 160 of 160 think blocks began with a literal <think>.
    vocab = {0: "<think>", 1: "abc", THINK_END: "</think>", 2: "de"}
    think, content, _ = split_trace([0, 1, THINK_END, 2], StubTokenizer(vocab))
    assert think == "abc"
    assert not think.startswith("<think>")


def test_split_trace_strips_the_open_tag_on_truncated_traces(tok):
    vocab = {0: "<think>", 1: "abc"}
    think, content, truncated = split_trace([0, 1], StubTokenizer(vocab))
    assert truncated is True
    assert content == "abc"


def test_split_trace_marks_truncated_when_terminator_absent(tok):
    think, content, truncated = split_trace([0, 1, 2], tok)
    assert think is None
    assert content == "abc"
    assert truncated is True


@pytest.mark.parametrize(
    "content,expected",
    [
        ('{"answer": "C"}', "C"),
        ('"answer":"B"', "B"),
        ('"answer" = "A"', "A"),
        ("the answer is D", None),
        ("no letter here", None),
    ],
)
def test_parse_answer(content, expected):
    answer, ok = parse_answer(content)
    assert answer == expected
    assert ok is (expected is not None)


def test_normalize_span_collapses_whitespace_and_quotes():
    assert normalize_span("The\n  grader’s  note") == "the grader's note"


def test_find_span_exact():
    think = "I recall the grader marked B as correct."
    assert think[find_span(think, "the grader marked B") :].startswith("the grader marked B")


def test_find_span_survives_judge_renormalization():
    # judges return spans with collapsed whitespace and straightened quotes; byte equality fails.
    think = "the reviewer’s note\n   says B"
    idx = find_span(think, "the reviewer's note says B")
    assert idx == 0


def test_find_span_returns_none_when_absent():
    assert find_span("nothing relevant here at all", "the grader marked B") is None
