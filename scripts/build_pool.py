# builds the raw item pool from mmlu non computational subsets and arc challenge, and expands it into
# the four variants per item with a seeded distractor map.
# uv run -m scripts.build_pool --output_dir data/processed --n_items 300 --seed 42

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.data.items import build_variants
from src.data.pool import assemble_pool, load_source_rows
from src.utils.io import append_jsonl, get_git_hash, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    output_dir: str = "data/processed"
    n_items: int = 300
    mmlu_subjects: str = "all"
    arc_split: str = "test"  # matches the mmlu split; train rows are likelier to sit verbatim in pretraining
    seed: int = 42


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    load_dotenv()
    config = simple_parsing.parse(Config)

    out = Path(config.output_dir)
    existing = [
        p for p in (out / f"pool_seed{config.seed}.jsonl", out / f"variants_seed{config.seed}.jsonl") if p.exists()
    ]
    if existing:
        raise FileExistsError(f"{existing} already exist; the writes append, so move them aside before rebuilding")

    rows = load_source_rows(config.mmlu_subjects, config.arc_split)
    build = assemble_pool(rows, n_items=config.n_items, seed=config.seed)
    if len(build.items) < config.n_items:
        LOGGER.warning(f"pool is short: {len(build.items)} of {config.n_items} requested")

    variants = [v for item in build.items for v in build_variants(item, seed=config.seed)]
    by_source = Counter(item.source for item in build.items)

    pool_path = append_jsonl(out / f"pool_seed{config.seed}.jsonl", [item.model_dump() for item in build.items])
    variants_path = append_jsonl(out / f"variants_seed{config.seed}.jsonl", [v.model_dump() for v in variants])
    meta_path = write_json(
        out / f"pool_seed{config.seed}_meta.json",
        {
            "git_hash": get_git_hash(),
            "seed": config.seed,
            "n_items": len(build.items),
            "n_variants": len(variants),
            "mmlu_subjects": config.mmlu_subjects,
            "arc_split": config.arc_split,
            "build_stats": build.stats,
            "items_by_source": dict(by_source.most_common()),
        },
    )

    print(f"Pool: {pool_path} ({len(build.items)} items)")
    print(f"Variants: {variants_path} ({len(variants)} rows)")
    print(f"Metadata: {meta_path}")
    print(f"Subjects: {len(by_source)} distinct, top 5 {by_source.most_common(5)}")


main()
