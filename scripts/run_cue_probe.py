# compares candidate cue strengths: samples V0 to find items the model already knows, then samples each
# cue on those items and reports how often the answer flips to the cued option.
# uv run -m scripts.run_cue_probe --variants_path data/processed/variants_seed42.jsonl --output_dir results/raw/cue_probe_v1 --model_id Qwen/Qwen3-8B --num_tasks 30 --seed 42

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.config import RUN, SCORING
from src.generation.engine import generate, load_engine, to_generations
from src.generation.prompts import ARMS, CUE_BLOCKS, build_chat, build_user_message, prompt_fingerprint
from src.utils.io import append_jsonl, get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    variants_path: str = ""
    output_dir: str = "results/raw/cue_probe_v1"
    model_id: str = "Qwen/Qwen3-8B"
    arm: str = "C3_neutral_private"
    num_tasks: int = 30
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
        raise FileExistsError(f"{generations_path} exists; use a new output_dir")

    run = RUN.model_copy(update={"model": config.model_id, "seed": config.seed})
    items = [r for r in read_jsonl(config.variants_path) if r["variant"] == "V1"][: config.num_tasks]
    LOGGER.info(f"probing {len(CUE_BLOCKS)} cues over {len(items)} items")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(run.model, revision=run.revision or None)
    engine = load_engine(run, config.gpu_memory_utilization)

    # every condition is built up front so one batch keeps the gpu saturated and prefix caching works.
    conditions = [("V0", None)] + [("V1", cue) for cue in CUE_BLOCKS]
    jobs = [(item, variant, cue) for variant, cue in conditions for item in items]
    prompts = [
        build_chat(
            tok,
            ARMS[config.arm],
            build_user_message(
                it["question"], it["options"], None if cue is None else it["cued_option"], cue or "reviewer_note"
            ),
        )
        for it, _, cue in jobs
    ]
    outputs = generate(engine, prompts, run)

    hits: dict[str, dict[str, int]] = defaultdict(dict)
    for (item, variant, cue), output in zip(jobs, outputs):
        generations = to_generations(output, tok, run)
        label = "V0" if cue is None else cue
        target = item["correct"] if cue is None else item["cued_option"]
        hits[label][item["item_id"]] = sum(g.parse_ok and g.parsed_answer == target for g in generations)
        append_jsonl(
            generations_path,
            [
                item
                | g.model_dump()
                | {
                    "stage": "cue_probe",
                    "cue_name": label,
                    "arm": config.arm,
                    "target": target,
                    "prompt_fingerprint": prompt_fingerprint(config.arm, cue or "reviewer_note"),
                }
                for g in generations
            ],
        )

    known = [i for i, n in hits["V0"].items() if n >= SCORING.filter_correct_min_count]
    LOGGER.info(f"{len(known)} of {len(items)} items answered correctly without a cue")

    summary = []
    for cue in CUE_BLOCKS:
        counts = [hits[cue][i] for i in known]
        n_pass = sum(c >= SCORING.filter_switch_min_count for c in counts)
        summary.append(
            {
                "cue_name": cue,
                "n_known_items": len(known),
                "mean_switch_rate": round(sum(counts) / (len(counts) * run.n_samples), 4) if counts else 0.0,
                "n_items_passing_filter": n_pass,
                "filter_yield_on_known": round(n_pass / len(known), 4) if known else 0.0,
            }
        )
    summary.sort(key=lambda s: -s["filter_yield_on_known"])

    report_path = write_json(
        out / "cue_probe_report.json",
        {
            "git_hash": get_git_hash(),
            "arm": config.arm,
            "run": run.model_dump(),
            "n_items": len(items),
            "n_known_items": len(known),
            "v0_accuracy": round(sum(hits["V0"].values()) / (len(items) * run.n_samples), 4),
            "summary": summary,
        },
    )

    print(f"Generations: {generations_path}")
    print(f"Report: {report_path}")
    print(f"Items answered correctly without a cue: {len(known)} of {len(items)}")
    print("cue                mean_switch  items_passing  yield")
    for s in summary:
        print(
            f"  {s['cue_name']:<17} {s['mean_switch_rate']:>10}  {s['n_items_passing_filter']:>13}  {s['filter_yield_on_known']:>6}"
        )


main()
