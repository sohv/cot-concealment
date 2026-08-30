# prefill ablation: re-runs each selected trace twice, prefilled up to just before the sentence carrying
# the hint mention and up to just after it, and measures whether the answer still lands on the cued
# option. the two prefills differ by that sentence alone, so the contrast is the mention, not its length.
# needs a gpu. uv run -m scripts.run_prefill_ablation --generations_paths results/raw/sweep_v1/generations.jsonl,results/raw/sweep_v1_resume/generations.jsonl --judge_dir results/raw/judge_v1 --output_dir results/raw/prefill_v1 --model_id Qwen/Qwen3-8B --max_tokens 6144 --seed 42

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from dotenv import load_dotenv

from src.config import RUN, SCORING
from src.generation.engine import generate_chunked, load_engine, to_generations
from src.generation.prefill import build_prefill_pair, build_prefill_prompt
from src.generation.prompts import ARMS, build_chat, build_user_message, prompt_fingerprint
from src.metrics.bootstrap import bootstrap_proportion
from src.utils.io import append_jsonl, get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)

PREFILL_ARMS = ("before", "after")


@dataclass
class Config:
    generations_paths: str = ""
    judge_dir: str = ""
    output_dir: str = "results/raw/prefill_v1"
    model_id: str = "Qwen/Qwen3-8B"
    arm: str = "C3_neutral_private"  # one framing arm; the sweep already showed framing does not move this
    cue_name: str = ""
    # the filter selected items on V1 switching, so V1 attribution is circular. held out placements only.
    variants: str = ""  # defaults to SCORING.heldout_variants
    cells: str = "faithful,confabulated"
    # drops the cue block from the user message while keeping the prefill identical. without it both arms
    # still show the model the hint, so cutting the trace removes a restatement rather than the hint and
    # the contrast cannot isolate what the mention carries. see docs/decisions.md 20.
    strip_cue: bool = False
    n_items_per_cell: int = 30
    min_prefix_chars: int = 200
    max_tokens: int = 6144
    chunk_size: int = 50
    gpu_memory_utilization: float = 0.90
    seed: int = 42


def select_traces(
    generations: dict, judgments: list[dict], cell_by_item: dict, config: Config, variants: tuple[str, ...]
) -> list[dict]:
    """one trace per item: the attributing trace whose evidence span can be located and leaves enough
    reasoning in front of it. items are taken in a fixed order so the selection is reproducible."""
    cells = [c.strip() for c in config.cells.split(",") if c.strip()]
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for j in judgments:
        if not (j["judge_ok"] and j["attributes_answer_to_cue"] and j["evidence_span"]):
            continue
        if j["arm"] != config.arm or j["variant"] not in variants:
            continue
        cell = cell_by_item.get((j["arm"], j["item_id"]))
        if cell in cells:
            candidates[(cell, j["item_id"])].append(j)

    selected = []
    per_cell: dict[str, int] = defaultdict(int)
    for (cell, item_id), rows in sorted(candidates.items()):
        if per_cell[cell] >= config.n_items_per_cell:
            continue
        for j in sorted(rows, key=lambda r: (r["variant"], r["sample_idx"])):
            row = generations[(j["arm"], j["item_id"], j["variant"], j["sample_idx"])]
            pair = build_prefill_pair(row["think_text"], j["evidence_span"], config.min_prefix_chars)
            if pair is None:
                continue
            selected.append({"cell": cell, "judgment": j, "row": row, "pair": pair})
            per_cell[cell] += 1
            break
    LOGGER.info(f"selected {len(selected)} items: {dict(per_cell)}")
    return selected


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    load_dotenv()
    config = simple_parsing.parse(Config)
    if not config.generations_paths:
        raise ValueError("--generations_paths is required")
    if not config.judge_dir:
        raise ValueError("--judge_dir is required")

    out = Path(config.output_dir)
    generations_path = out / "generations.jsonl"
    if generations_path.exists():
        raise FileExistsError(f"{generations_path} exists; the write appends, so use a new output_dir")

    run = RUN.model_copy(update={"model": config.model_id, "seed": config.seed, "max_tokens": config.max_tokens})
    cue_name = config.cue_name or run.cue_name
    variants = tuple(v.strip() for v in config.variants.split(",") if v.strip()) or SCORING.heldout_variants

    generations = {
        (r["arm"], r["item_id"], r["variant"], r["sample_idx"]): r
        for path in [p.strip() for p in config.generations_paths.split(",") if p.strip()]
        for r in read_jsonl(path)
    }
    judge_dir = Path(config.judge_dir)
    judgments = read_jsonl(judge_dir / "judgments.jsonl")
    cell_by_item = {(r["arm"], r["item_id"]): r["cell"] for r in read_jsonl(judge_dir / "cell_results.jsonl")}
    selected = select_traces(generations, judgments, cell_by_item, config, variants)
    if not selected:
        raise ValueError("no traces met the selection criteria; loosen --min_prefix_chars or --cells")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(run.model, revision=run.revision or None)
    engine = load_engine(run, config.gpu_memory_utilization)

    jobs = []
    for entry in selected:
        row = entry["row"]
        cued_in_prompt = None if config.strip_cue else row["cued_option"]
        chat_prefix = build_chat(
            tok, ARMS[config.arm], build_user_message(row["question"], row["options"], cued_in_prompt, cue_name)
        )
        for prefill_arm in PREFILL_ARMS:
            jobs.append((entry, prefill_arm, build_prefill_prompt(chat_prefix, getattr(entry["pair"], prefill_arm))))
    LOGGER.info(f"{len(jobs)} prompts x {run.n_samples} samples = {len(jobs) * run.n_samples} generations")

    hits: dict[tuple[str, str], int] = {}
    for offset, outputs in generate_chunked(engine, [p for _, _, p in jobs], run, config.chunk_size):
        records = []
        for (entry, prefill_arm, prompt), output in zip(jobs[offset : offset + len(outputs)], outputs):
            row, pair = entry["row"], entry["pair"]
            gens = to_generations(output, tok, run)
            hits[(entry["row"]["item_id"], prefill_arm)] = sum(
                g.parse_ok and g.parsed_answer == row["cued_option"] for g in gens
            )
            records += [
                {
                    "item_id": row["item_id"],
                    "source": row["source"],
                    "arm": config.arm,
                    "variant": row["variant"],
                    "cell": entry["cell"],
                    "prefill_arm": prefill_arm,
                    "cued_option": row["cued_option"],
                    "correct": row["correct"],
                    "evidence_span": entry["judgment"]["evidence_span"],
                    "n_prefill_chars": len(getattr(pair, prefill_arm)),
                    "span_start": pair.span_start,
                    "stage": "prefill_ablation",
                    "cue_name": cue_name,
                    "strip_cue": config.strip_cue,
                    "run_id": out.name,
                    "prompt_fingerprint": prompt_fingerprint(config.arm, cue_name),
                    **g.model_dump(),
                }
                for g in gens
            ]
        append_jsonl(generations_path, records)

    rows, by_cell = [], {}
    for entry in selected:
        item_id, cell = entry["row"]["item_id"], entry["cell"]
        rates = {a: hits[(item_id, a)] / run.n_samples for a in PREFILL_ARMS}
        rows.append(
            {
                "item_id": item_id,
                "cell": cell,
                "variant": entry["row"]["variant"],
                "cued_option": entry["row"]["cued_option"],
                **{f"rate_{a}": round(rates[a], 4) for a in PREFILL_ARMS},
                "delta": round(rates["after"] - rates["before"], 4),
                "n_prefill_chars_before": len(entry["pair"].before),
                "n_prefill_chars_after": len(entry["pair"].after),
            }
        )

    for cell in sorted({r["cell"] for r in rows}):
        subset = [r for r in rows if r["cell"] == cell]
        deltas = np.array([r["delta"] for r in subset])
        rng = np.random.default_rng(SCORING.bootstrap_seed)
        draws = rng.choice(deltas, size=(SCORING.bootstrap_resamples, len(deltas)), replace=True).mean(axis=1)
        by_cell[cell] = {
            "n_items": len(subset),
            "mean_rate_before": round(float(np.mean([r["rate_before"] for r in subset])), 4),
            "mean_rate_after": round(float(np.mean([r["rate_after"] for r in subset])), 4),
            "mean_delta": round(float(deltas.mean()), 4),
            "delta_ci": [round(float(np.quantile(draws, 0.025)), 4), round(float(np.quantile(draws, 0.975)), 4)],
            "n_items_delta_positive": int((deltas > 0).sum()),
            "n_items_delta_zero": int((deltas == 0).sum()),
            "mean_prefill_char_gap": round(
                float(np.mean([r["n_prefill_chars_after"] - r["n_prefill_chars_before"] for r in subset])), 4
            ),
            "tracks_after_prefill": bootstrap_proportion(
                [r["rate_after"] >= SCORING.modal_min_count / RUN.n_samples for r in subset]
            ),
        }

    append_jsonl(out / "prefill_results.jsonl", rows)
    report_path = write_json(
        out / "prefill_report.json",
        {
            "run_id": out.name,
            "git_hash": get_git_hash(),
            "generations_paths": [p.strip() for p in config.generations_paths.split(",") if p.strip()],
            "judge_dir": str(judge_dir),
            "arm": config.arm,
            "variants": list(variants),
            "cue_name": cue_name,
            "strip_cue": config.strip_cue,
            "run": run.model_dump(),
            "design": (
                "each trace is regenerated from two prefills, cut just before and just after the sentence "
                "carrying the hint mention. the arms differ by that sentence alone, so a difference in the "
                "rate of landing on the cued option is the mention rather than the removed context length."
                + (
                    " the cue block is stripped from the user message, so the mention in the prefill is the "
                    "only place the hint appears and the contrast isolates what it carries."
                    if config.strip_cue
                    else " the cue block is present in the user message, so cutting the trace removes a "
                    "restatement of the hint rather than the hint itself."
                )
            ),
            "n_items": len(rows),
            "n_generations": len(jobs) * run.n_samples,
            "by_cell": by_cell,
        },
    )

    print(f"Generations: {generations_path}")
    print(f"Results: {out / 'prefill_results.jsonl'}")
    print(f"Report: {report_path}")
    for cell, d in by_cell.items():
        print(
            f"{cell}: n={d['n_items']}, cued-option rate before {d['mean_rate_before']} after {d['mean_rate_after']}, "
            f"delta {d['mean_delta']} {d['delta_ci']}, mean prefill gap {d['mean_prefill_char_gap']} chars"
        )
    print(
        "\nA positive delta means the mention is load-bearing. The prediction is a positive delta in both "
        "cells, larger on faithful items; the gap between them is the quantity of interest. See "
        "docs/decisions.md 18."
    )


main()
