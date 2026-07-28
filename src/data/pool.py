# loads mmlu non computational subsets and arc challenge from huggingface and normalizes them into the
# four option Item schema. rows that cannot be normalized cleanly are dropped and counted, never patched.

import hashlib
import logging
import random

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
        question=row["question"].strip(),
        options=dict(zip(OPTIONS, [t.strip() for t in texts])),
        correct=OPTIONS[labels.index(row["answerKey"])],
    )


NORMALIZERS = {"mmlu": normalize_mmlu_row, "arc": normalize_arc_row}


def assemble_pool(rows: list[tuple[dict, str]], n_items: int, seed: int) -> list[Item]:
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

    pool: list[Item] = []
    n_dropped = 0
    for row, kind in unique:
        if len(pool) >= n_items:
            break
        item = NORMALIZERS[kind](row, item_id=f"pool_{len(pool):05d}")
        if item is None:
            n_dropped += 1
            continue
        pool.append(item)

    LOGGER.info(
        f"assembled {len(pool)} items, dropped {n_dropped} unnormalizable, {len(rows) - len(unique)} duplicates"
    )
    return pool


def load_source_rows(mmlu_subjects: str = "all", arc_split: str = "train") -> list[tuple[dict, str]]:
    from datasets import load_dataset

    mmlu = load_dataset(MMLU_DATASET, mmlu_subjects, split="test")
    arc = load_dataset(ARC_DATASET, ARC_CONFIG, split=arc_split)
    LOGGER.info(f"loaded {len(mmlu)} mmlu rows and {len(arc)} arc rows")
    return [(dict(r), "mmlu") for r in mmlu] + [(dict(r), "arc") for r in arc]
