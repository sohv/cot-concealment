# loads mmlu non computational subsets and arc challenge from huggingface and normalizes them into the
# four option Item schema. rows that cannot be normalized cleanly are dropped and counted, never patched.

import hashlib
import logging
import random
from collections import Counter

from pydantic import BaseModel

from src.config import OPTIONS
from src.data.items import Item, is_excluded_source

LOGGER = logging.getLogger(__name__)

MMLU_DATASET = "cais/mmlu"
ARC_DATASET = "allenai/ai2_arc"
ARC_CONFIG = "ARC-Challenge"


def normalize_mmlu_row(row: dict, item_id: str) -> Item | None:
    source = f"mmlu_{row['subject']}"
    if is_excluded_source(source):
        return None
    choices = row["choices"]
    if len(choices) != len(OPTIONS):
        return None
    if not 0 <= row["answer"] < len(OPTIONS):
        return None
    return Item(
        item_id=item_id,
        source=source,
        question=row["question"].strip(),
        options=dict(zip(OPTIONS, [c.strip() for c in choices])),
        correct=OPTIONS[row["answer"]],
    )


def normalize_arc_row(row: dict, item_id: str) -> Item | None:
    """arc mixes letter and numeric label schemes and has rows with three or five options, so labels are
    remapped by position and anything that is not four options is dropped."""
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    if len(texts) != len(OPTIONS) or len(labels) != len(OPTIONS):
        return None
    if row["answerKey"] not in labels:
        return None
    return Item(
        item_id=item_id,
        source="arc_challenge",
        source_id=row.get("id"),
        question=row["question"].strip(),
        options=dict(zip(OPTIONS, [t.strip() for t in texts])),
        correct=OPTIONS[labels.index(row["answerKey"])],
    )


NORMALIZERS = {"mmlu": normalize_mmlu_row, "arc": normalize_arc_row}


def drop_reason(row: dict, kind: str) -> str | None:
    """why a row cannot enter the pool, or None if it can. separates the deliberate subject exclusion
    from malformed rows, since lumping them together hides a source that silently contributes nothing."""
    if kind == "mmlu":
        if is_excluded_source(f"mmlu_{row['subject']}"):
            return "excluded_subject"
        if len(row["choices"]) != len(OPTIONS):
            return "wrong_option_count"
        if not 0 <= row["answer"] < len(OPTIONS):
            return "answer_out_of_range"
        return None
    if len(row["choices"]["text"]) != len(OPTIONS) or len(row["choices"]["label"]) != len(OPTIONS):
        return "wrong_option_count"
    if row["answerKey"] not in row["choices"]["label"]:
        return "answer_key_not_in_labels"
    return None


class PoolBuild(BaseModel):
    items: list[Item]
    stats: dict[str, int]

    def __len__(self) -> int:
        return len(self.items)


def assemble_pool(rows: list[tuple[dict, str]], n_items: int, seed: int) -> PoolBuild:
    """deduplicates on question text, shuffles under the seed, then caps. ids are assigned after the
    shuffle so pool_00000 is stable for a given seed and source set."""
    seen: set[str] = set()
    unique = []
    for row, kind in rows:
        key = hashlib.md5(row["question"].strip().lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append((row, kind))

    random.Random(seed).shuffle(unique)

    items: list[Item] = []
    drops: Counter = Counter()
    n_considered = 0
    for row, kind in unique:
        if len(items) >= n_items:
            break
        n_considered += 1
        reason = drop_reason(row, kind)
        if reason:
            drops[reason] += 1
            continue
        items.append(NORMALIZERS[kind](row, item_id=f"pool_{len(items):05d}"))

    stats = {
        "n_source_rows": len(rows),
        "n_duplicates": len(rows) - len(unique),
        "n_considered": n_considered,
        "n_kept": len(items),
        "n_dropped": sum(drops.values()),
        **{f"dropped_{reason}": count for reason, count in sorted(drops.items())},
    }
    LOGGER.info(f"assembled pool: {stats}")
    return PoolBuild(items=items, stats=stats)


def load_source_rows(mmlu_subjects: str = "all", arc_split: str = "test") -> list[tuple[dict, str]]:
    from datasets import load_dataset

    mmlu = load_dataset(MMLU_DATASET, mmlu_subjects, split="test")
    arc = load_dataset(ARC_DATASET, ARC_CONFIG, split=arc_split)
    LOGGER.info(f"loaded {len(mmlu)} mmlu rows and {len(arc)} arc rows")
    return [(dict(r), "mmlu") for r in mmlu] + [(dict(r), "arc") for r in arc]
