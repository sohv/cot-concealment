# builds the two prefills for the ablation: the trace truncated just before the sentence carrying the
# hint mention, and just after it. the two differ by that sentence and nothing else, so the length
# confound is the mention's own length rather than everything downstream of it.

from pydantic import BaseModel

from src.generation.parsing import find_span

SENTENCE_ENDS = ".!?\n"
THINK_OPEN = "<think>\n"


class PrefillPair(BaseModel):
    before: str
    after: str
    span_start: int
    sentence_start: int
    sentence_end: int


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """expands [start, end) outward to the enclosing sentence. cutting mid sentence would leave the
    'before' arm holding half a mention, which is the one thing that arm must not contain."""
    left = 0
    for i in range(min(start, len(text)) - 1, -1, -1):
        if text[i] in SENTENCE_ENDS:
            left = i + 1
            break
    # step past the whitespace after the terminator so the boundary is the sentence's first character.
    while left < len(text) and text[left].isspace():
        left += 1
    right = len(text)
    for i in range(max(end - 1, 0), len(text)):
        if text[i] in SENTENCE_ENDS:
            right = i + 1
            break
    return left, right


def build_prefill_pair(think_text: str, span: str, min_prefix_chars: int = 200) -> PrefillPair | None:
    """None when the span cannot be located, or when the mention sits so early that the 'before' arm
    would carry almost no reasoning — that item compares a near-empty prefill against a real one and is
    a length contrast, not a mention contrast."""
    if not span or not think_text:
        return None
    start = find_span(think_text, span)
    if start is None:
        return None
    end = min(len(think_text), start + len(span))
    left, right = sentence_bounds(think_text, start, end)
    if left < min_prefix_chars:
        return None
    return PrefillPair(
        before=think_text[:left].rstrip(),
        after=think_text[:right].rstrip(),
        span_start=start,
        sentence_start=left,
        sentence_end=right,
    )


def build_prefill_prompt(chat_prefix: str, prefill_text: str) -> str:
    """qwen3 emits its own opening think tag, so the chat template does not inject one and the prefill
    has to open the block itself or the continuation is not inside a think block at all."""
    return f"{chat_prefix}{THINK_OPEN}{prefill_text}"
