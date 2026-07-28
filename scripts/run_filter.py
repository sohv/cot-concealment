# behavioural filter: samples V0 and V1 eight times per item and keeps items answered correctly without
# a hint and switched to the hint with one. also yields the first truncation and parse failure rates.
# uv run -m scripts.run_filter --variants_path data/processed/variants_seed42.jsonl --output_dir results/raw/filter_v1 --model_id Qwen/Qwen3-8B --seed 42

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.config import GATES, RUN, SCORING
from src.generation.engine import generate, load_engine, to_generations
from src.generation.prompts import ARMS, build_chat, build_user_message, prompt_fingerprint
from src.metrics.filter import filter_report, score_filter_item
from src.metrics.scoring import Trace
from src.utils.io import append_jsonl, get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)

FILTER_VARIANTS = ("V0", "V1")


@dataclass
class Config:
    variants_path: str = ""
    output_dir: str = "results/raw/filter_v1"
    model_id: str = "Qwen/Qwen3-8B"
    revision: str = ""
    engine_version: str = ""
    arm: str = "C3_neutral_private"
    num_tasks: int | None = None
    gpu_memory_utilization: float = 0.90
    seed: int = 42


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    load_dotenv()
    config = simple_parsing.parse(Config)
    if not config.variants_path:
        raise ValueError("--variants_path is required")

    out = Path(config.output_dir)
    generations_path = out / "generations.jsonl"
    if generations_path.exists():
        raise FileExistsError(f"{generations_path} exists; the write appends, so use a new output_dir")

    # only override the frozen pins when explicitly given, or an empty cli default silently unpins them.
    update = {"model": config.model_id, "seed": config.seed}
    if config.revision:
        update["revision"] = config.revision
    if config.engine_version:
        update["engine_version"] = config.engine_version
    run = RUN.model_copy(update=update)
    run_id = out.name

    rows = [r for r in read_jsonl(config.variants_path) if r["variant"] in FILTER_VARIANTS]
    item_ids = sorted({r["item_id"] for r in rows})
    if config.num_tasks:
        item_ids = item_ids[: config.num_tasks]
    rows = [r for r in rows if r["item_id"] in set(item_ids)]
    LOGGER.info(f"filtering {len(item_ids)} items over {len(rows)} prompts")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(run.model, revision=run.revision or None)
    engine = load_engine(run, config.gpu_memory_utilization)

    prompts = [
        build_chat(tok, ARMS[config.arm], build_user_message(r["question"], r["options"], r["cued_option"]))
        for r in rows
    ]
    outputs = generate(engine, prompts, run)

    fingerprint = prompt_fingerprint(config.arm)
    traces_by_item: dict[str, dict[str, list[Trace]]] = defaultdict(dict)
    for row, output in zip(rows, outputs):
        generations = to_generations(output, tok, run)
        append_jsonl(
            generations_path,
            [
                row
                | g.model_dump()
                | {"stage": "filter", "arm": config.arm, "run_id": run_id, "prompt_fingerprint": fingerprint}
                for g in generations
            ],
        )
        traces_by_item[row["item_id"]][row["variant"]] = [
            Trace(parsed_answer=g.parsed_answer, parse_ok=g.parse_ok, truncated=g.truncated) for g in generations
        ]

    by_item = {r["item_id"]: r for r in rows if r["variant"] == "V1"}
    results = [
        score_filter_item(
            item_id=item_id,
            source=by_item[item_id]["source"],
            v0_traces=traces_by_item[item_id]["V0"],
            v1_traces=traces_by_item[item_id]["V1"],
            correct=by_item[item_id]["correct"],
            cued_option=by_item[item_id]["cued_option"],
            scoring=SCORING,
        )
        for item_id in item_ids
    ]
    report = filter_report(results, SCORING)

    survivors = [r.item_id for r in results if r.survives]
    append_jsonl(out / "filter_results.jsonl", [r.model_dump() for r in results])
    write_json(out / "survivors.json", {"run_id": run_id, "git_hash": get_git_hash(), "item_ids": survivors})
    report_path = write_json(
        out / "filter_report.json",
        {
            "run_id": run_id,
            "git_hash": get_git_hash(),
            "arm": config.arm,
            "prompt_fingerprint": fingerprint,
            "run": run.model_dump(),
            "report": report,
        },
    )

    print(f"Generations: {generations_path}")
    print(f"Report: {report_path}")
    print(
        f"Filter yield: {report['filter_yield']} ({report['n_survivors']} of {report['n_items']}), gate >{GATES.filter_yield_min}: {report['gate_filter_yield_passes']}"
    )
    print(
        f"Truncation rate: {report['truncation_rate']}, gate <{GATES.truncation_rate_max}: {report['gate_truncation_rate_passes']}"
    )
    print(f"Parse failure rate: {report['parse_fail_rate']}")
    print(
        f"Failed V0 only {report['n_failed_v0_only']}, failed V1 only {report['n_failed_v1_only']}, failed both {report['n_failed_both']}"
    )


main()
