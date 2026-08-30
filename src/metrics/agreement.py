# judge against human agreement for the hand check. kappa on a cell-stratified sample has a manufactured
# prevalence, so PABAK sits beside it rather than replacing it.

from collections import Counter

FIELDS = ("mentions_cue", "named_option", "attributes_answer_to_cue")


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    """unweighted, over whatever label set the two raters used between them. None when chance agreement
    is already 1, which happens when both raters used a single label throughout."""
    n = len(a)
    if not n:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(x) / n) * (b.count(x) / n) for x in labels)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def score_agreement(judge_rows: list[dict], labels: list[dict], fields: tuple[str, ...] = FIELDS) -> dict:
    """per field agreement, kappa and PABAK, with every disagreement listed so it can be read rather
    than summarized away. a labelled id with no entry in the key is a mismatched sheet, so it raises."""
    by_id = {j["id"]: j for j in judge_rows}
    missing = [row["id"] for row in labels if row["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} labelled ids are not in the key, e.g. {missing[:3]}")

    out = {}
    for field in fields:
        labelled = [row for row in labels if field in row]
        if not labelled:
            continue
        judge_labels = [str(by_id[row["id"]][field]) for row in labelled]
        human_labels = [str(row[field]) for row in labelled]
        agreement = sum(x == y for x, y in zip(judge_labels, human_labels)) / len(labelled)
        out[field] = {
            "n": len(labelled),
            "agreement": round(agreement, 4),
            "cohens_kappa": cohens_kappa(judge_labels, human_labels),
            "pabak": round(2 * agreement - 1, 4),
            "judge_distribution": dict(Counter(judge_labels)),
            "human_distribution": dict(Counter(human_labels)),
            "disagreements": [
                {"id": row["id"], "judge": j, "human": h}
                for row, j, h in zip(labelled, judge_labels, human_labels)
                if j != h
            ],
        }
    return out
