# elicitation positive control: on faithful items, where the hint demonstrably drove the answer and the
# trace already credited it, ask the model afterwards whether anything besides the question influenced it.
# a failure to recover the hint here is an instrument failure, not evidence that the reason was absent.
# uv run -m scripts.run_interrogation --generations_paths results/raw/sweep_v1/generations.jsonl,results/raw/sweep_v1_resume/generations.jsonl --judge_dir results/raw/judge_v1 --variants_path data/processed/variants_seed42.jsonl --output_dir results/raw/interrogation_v1 --model_id Qwen/Qwen3-8B --cue_name grader_code --seed 42

import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from dotenv import load_dotenv

from src.config import RUN, SCORING
from src.generation.engine import generate_chunked, load_engine, to_generations
from src.generation.interrogation import INTERROGATIONS, Interrogation, build_interrogation_messages
from src.generation.interrogation import build_judge_messages as build_interrogation_judge_messages
from src.generation.judge import normalize_named_option
from src.generation.prompts import ARMS, build_chat_from_messages, build_user_message
from src.metrics.bootstrap import bootstrap_proportion
from src.utils.cache import cached_completion
from src.utils.io import append_jsonl, get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    generations_paths: str = ""
    judge_dir: str = ""
    variants_path: str = ""
    output_dir: str = "results/raw/interrogation_v1"
    model_id: str = "Qwen/Qwen3-8B"
    arm: str = "C3_neutral_private"
    cue_name: str = ""
    cell: str = "faithful"  # the positive control population: influence is established and was verbalized
    kinds: str = "open,direct"
    n_samples: int = 4
    max_tokens: int = 2048
    chunk_size: int = 50
    gpu_memory_utilization: float = 0.90
    judge_model: str = "openai/gpt-4.1-mini"
    max_concurrent: int = 40
    seed: int = 42


def select_items(generations: dict, cell_by_item: dict, config: Config, variants: tuple[str, ...]) -> list[dict]:
    """one generation per item: a held out placement whose answer actually followed the hint, so the
    interrogation asks about a case where the influence is not in doubt."""
    chosen: dict[str, dict] = {}
    for (arm, item_id, variant, _idx), row in sorted(generations.items()):
        if arm != config.arm or variant not in variants or item_id in chosen:
            continue
        if cell_by_item.get((arm, item_id)) != config.cell:
            continue
        if row["truncated"] or not row["parse_ok"] or row["parsed_answer"] != row["cued_option"]:
            continue
        chosen[item_id] = row
    return list(chosen.values())


async def judge_replies(rows: list[dict], config: Config) -> list[dict]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    sem = asyncio.Semaphore(config.max_concurrent)

    async def one(row: dict) -> dict:
        async with sem:
            text = await cached_completion(
                client,
                model=config.judge_model,
                messages=build_interrogation_judge_messages(row["content_text"]),
                cache_dir="cache/interrogation_judge",
                temperature=0,
            )
        import json

        stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw = json.loads(stripped)
        judgment = Interrogation(
            acknowledges=bool(raw.get("acknowledges", False)),
            named_option=normalize_named_option(raw.get("named_option")),
            mentions_element_without_influence=bool(raw.get("mentions_element_without_influence", False)),
        )
        return row | judgment.model_dump() | {"names_correct_option": judgment.named_option == row["cued_option"]}

    return await asyncio.gather(*[one(r) for r in rows])


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    load_dotenv()
    config = simple_parsing.parse(Config)
    for required in ("generations_paths", "judge_dir", "variants_path"):
        if not getattr(config, required):
            raise ValueError(f"--{required} is required")

    out = Path(config.output_dir)
    generations_path = out / "generations.jsonl"
    if generations_path.exists():
        raise FileExistsError(f"{generations_path} exists; the write appends, so use a new output_dir")

    run = RUN.model_copy(update={"model": config.model_id, "seed": config.seed, "max_tokens": config.max_tokens,
                                 "n_samples": config.n_samples})
    cue_name = config.cue_name or run.cue_name
    kinds = [k.strip() for k in config.kinds.split(",") if k.strip()]

    generations = {
        (r["arm"], r["item_id"], r["variant"], r["sample_idx"]): r
        for path in [p.strip() for p in config.generations_paths.split(",") if p.strip()]
        for r in read_jsonl(path)
    }
    cell_by_item = {(r["arm"], r["item_id"]): r["cell"] for r in read_jsonl(Path(config.judge_dir) / "cell_results.jsonl")}
    variant_rows = {r["item_id"]: r for r in read_jsonl(config.variants_path) if r["variant"] == "V1"}
    selected = select_items(generations, cell_by_item, config, SCORING.heldout_variants)
    if not selected:
        raise ValueError(f"no {config.cell} items on arm {config.arm} with a held out placement that followed the hint")
    LOGGER.info(f"selected {len(selected)} {config.cell} items on {config.arm}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(run.model, revision=run.revision or None)
    engine = load_engine(run, config.gpu_memory_utilization)

    jobs = []
    for row in selected:
        item = variant_rows[row["item_id"]]
        user_message = build_user_message(item["question"], item["options"], row["cued_option"], cue_name)
        for kind in kinds:
            messages = build_interrogation_messages(ARMS[config.arm], user_message, row["content_text"], kind)
            jobs.append((row, kind, build_chat_from_messages(tok, messages)))
    LOGGER.info(f"{len(jobs)} prompts x {run.n_samples} samples = {len(jobs) * run.n_samples} generations")

    records = []
    for offset, outputs in generate_chunked(engine, [p for _, _, p in jobs], run, config.chunk_size):
        batch = []
        for (row, kind, _prompt), output in zip(jobs[offset : offset + len(outputs)], outputs):
            for g in to_generations(output, tok, run):
                batch.append(
                    {
                        "item_id": row["item_id"],
                        "arm": config.arm,
                        "variant": row["variant"],
                        "cell": config.cell,
                        "kind": kind,
                        "cued_option": row["cued_option"],
                        "correct": row["correct"],
                        "prior_answer": row["parsed_answer"],
                        "cue_name": cue_name,
                        "stage": "interrogation",
                        "run_id": out.name,
                        **g.model_dump(),
                    }
                )
        append_jsonl(generations_path, batch)
        records += batch

    judged = asyncio.run(judge_replies([r for r in records if not r["truncated"]], config))
    append_jsonl(out / "interrogation_judgments.jsonl", judged)

    by_kind = {}
    for kind in kinds:
        rows = [r for r in judged if r["kind"] == kind]
        per_item = defaultdict(list)
        for r in rows:
            per_item[r["item_id"]].append(r)
        # the item is the unit: n_samples replies about one item are not independent observations.
        ack = [float(np.mean([x["acknowledges"] for x in v])) for v in per_item.values()]
        correct = [float(np.mean([x["names_correct_option"] for x in v])) for v in per_item.values()]
        any_ack = [any(x["acknowledges"] for x in v) for v in per_item.values()]
        by_kind[kind] = {
            "n_items": len(per_item),
            "n_replies": len(rows),
            "mean_acknowledges": round(float(np.mean(ack)), 4),
            "acknowledges_at_least_once": bootstrap_proportion(any_ack),
            "mean_names_correct_option": round(float(np.mean(correct)), 4),
            "mentions_without_influence": round(float(np.mean([r["mentions_element_without_influence"] for r in rows])), 4),
        }

    report_path = write_json(
        out / "interrogation_report.json",
        {
            "run_id": out.name,
            "git_hash": get_git_hash(),
            "arm": config.arm,
            "cell": config.cell,
            "cue_name": cue_name,
            "judge_model": config.judge_model,
            "run": run.model_dump(),
            "design": (
                "positive control. every item is one where the hint demonstrably drove the answer and the "
                "trace already credited it, so the reason is established as present and articulable. a low "
                "recovery rate here is a failure of interrogation as an instrument, not evidence that the "
                "reason was never represented."
            ),
            "interrogations": {k: INTERROGATIONS[k] for k in kinds},
            "n_items": len(selected),
            "by_kind": by_kind,
        },
    )

    print(f"Generations: {generations_path}")
    print(f"Report: {report_path}")
    for kind, d in by_kind.items():
        print(
            f"{kind:8} n={d['n_items']}  acknowledges {d['mean_acknowledges']}  "
            f"at least once {d['acknowledges_at_least_once']['point']} {d['acknowledges_at_least_once']['ci']}  "
            f"names the right option {d['mean_names_correct_option']}  sees it but denies influence {d['mentions_without_influence']}"
        )


main()
