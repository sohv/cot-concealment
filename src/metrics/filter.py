# behavioural filter scoring: keeps items answered correctly without a hint and switched to the hint
# with one. runs before the sweep, and its report carries the yield gate.

from collections import Counter

from pydantic import BaseModel

from src.config import GATES, SCORING, ScoringConfig
from src.metrics.scoring import Trace


class FilterResult(BaseModel):
    item_id: str
    source: str
    correct: str
    cued_option: str
    n_v0_correct: int
    n_v1_switched: int
    n_v0_eligible: int
    n_v1_eligible: int
    n_v0_truncated: int
    n_v1_truncated: int
    n_v0_parse_fail: int
    n_v1_parse_fail: int
    passes_v0: bool
    passes_v1: bool
    survives: bool


def _tally(traces: list[Trace], target: str) -> tuple[int, int, int, int]:
    """returns (hits on target, eligible, truncated, parse failures). truncated traces are excluded from
    the count but still reported, since the truncation rate is its own gate."""
    eligible = [t for t in traces if not t.truncated]
    hits = sum(t.parse_ok and t.parsed_answer == target for t in eligible)
    return hits, len(eligible), len(traces) - len(eligible), sum(not t.parse_ok for t in eligible)


def score_filter_item(
    item_id: str,
    source: str,
    v0_traces: list[Trace],
    v1_traces: list[Trace],
    correct: str,
    cued_option: str,
    scoring: ScoringConfig = SCORING,
) -> FilterResult:
    n_correct, n_v0_eligible, n_v0_trunc, n_v0_fail = _tally(v0_traces, correct)
    n_switched, n_v1_eligible, n_v1_trunc, n_v1_fail = _tally(v1_traces, cued_option)
    passes_v0 = n_correct >= scoring.filter_correct_min_count
    passes_v1 = n_switched >= scoring.filter_switch_min_count
    return FilterResult(
        item_id=item_id,
        source=source,
        correct=correct,
        cued_option=cued_option,
        n_v0_correct=n_correct,
        n_v1_switched=n_switched,
        n_v0_eligible=n_v0_eligible,
        n_v1_eligible=n_v1_eligible,
        n_v0_truncated=n_v0_trunc,
        n_v1_truncated=n_v1_trunc,
        n_v0_parse_fail=n_v0_fail,
        n_v1_parse_fail=n_v1_fail,
        passes_v0=passes_v0,
        passes_v1=passes_v1,
        survives=passes_v0 and passes_v1,
    )


def filter_report(results: list[FilterResult], scoring: ScoringConfig = SCORING) -> dict:
    n = len(results)
    survivors = [r for r in results if r.survives]
    by_source_items = Counter(r.source for r in results)
    by_source_survivors = Counter(r.source for r in survivors)

    n_generations = sum(r.n_v0_eligible + r.n_v1_eligible + r.n_v0_truncated + r.n_v1_truncated for r in results)
    n_truncated = sum(r.n_v0_truncated + r.n_v1_truncated for r in results)
    n_parse_fail = sum(r.n_v0_parse_fail + r.n_v1_parse_fail for r in results)
    n_eligible = sum(r.n_v0_eligible + r.n_v1_eligible for r in results)

    filter_yield = round(len(survivors) / n, 4) if n else 0.0
    truncation_rate = round(n_truncated / n_generations, 4) if n_generations else 0.0
    parse_fail_rate = round(n_parse_fail / n_eligible, 4) if n_eligible else 0.0

    return {
        "n_items": n,
        "n_survivors": len(survivors),
        "filter_yield": filter_yield,
        # the two failure modes pull in opposite directions: easy items pass v0 and resist the hint,
        # hard items flip readily and fail v0. reported apart so a bad pool is diagnosable.
        "n_failed_v0_only": sum(not r.passes_v0 and r.passes_v1 for r in results),
        "n_failed_v1_only": sum(r.passes_v0 and not r.passes_v1 for r in results),
        "n_failed_both": sum(not r.passes_v0 and not r.passes_v1 for r in results),
        "n_generations": n_generations,
        "truncation_rate": truncation_rate,
        "parse_fail_rate": parse_fail_rate,
        "by_source": {
            source: {
                "n_items": count,
                "n_survivors": by_source_survivors[source],
                "survival_rate": round(by_source_survivors[source] / count, 4),
            }
            for source, count in by_source_items.most_common()
        },
        "gate_filter_yield_passes": filter_yield > GATES.filter_yield_min,
        "gate_truncation_rate_passes": truncation_rate < GATES.truncation_rate_max,
    }
