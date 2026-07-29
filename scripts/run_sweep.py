# sweep: samples V0 to V3 across every framing arm for the filtered survivors, then reports behavioural
# tracking. tracking needs no judge and no api key, so it settles the tracking gate before any judging spend.
# uv run -m scripts.run_sweep --variants_path data/processed/variants_seed42.jsonl --survivors_path results/raw/filter_v1/survivors.json --output_dir results/raw/sweep_v1 --model_id Qwen/Qwen3-8B --max_tokens 6144 --seed 42

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.config import CUED_VARIANTS, GATES, RUN, SCORING, VARIANTS
from src.generation.engine import generate_chunked, load_engine, to_generations
from src.generation.prompts import ARMS, build_chat, build_user_message, prompt_fingerprint
from src.metrics.scoring import Trace, score_variant
from src.utils.io import append_jsonl, get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    variants_path: str = ""
    survivors_path: str = ""
    output_dir: str = "results/raw/sweep_v1"
    model_id: str = "Qwen/Qwen3-8B"
    arms: str = ""  # comma separated; defaults to RUN.arms
    cue_name: str = ""
    max_tokens: int = 6144  # raised from the filter's 4096; truncation there was 1.90% against a 2% gate
    chunk_size: int = 100
    num_tasks: int | None = None
    gpu_memory_utilization: float = 0.90
    seed: int = 42


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    load_dotenv()
    config = simple_parsing.parse(Config)
    if not config.variants_path:
        raise ValueError("--variants_path is required")
    if not config.survivors_path:
        raise ValueError("--survivors_path is required")

    out = Path(config.output_dir)
    generations_path = out / "generations.jsonl"
    if generations_path.exists():
        raise FileExistsError(f"{generations_path} exists; use a new output_dir")

    run = RUN.model_copy(update={"model": config.model_id, "seed": config.seed, "max_tokens": config.max_tokens})
    arms = [a.strip() for a in config.arms.split(",") if a.strip()] or list(run.arms)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; available {list(ARMS)}")
    cue_name = config.cue_name or run.cue_name

    survivors = json.loads(Path(config.survivors_path).read_text())["item_ids"]
    if config.num_tasks:
        survivors = survivors[: config.num_tasks]
    keep = set(survivors)
    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in read_jsonl(config.variants_path):
        if r["item_id"] in keep:
            by_item[r["item_id"]][r["variant"]] = r

    # ordered arm then item then variant: the system message is the outermost shared prefix and the
    # question stem the next, so prefix caching gets the longest possible reuse.
    jobs = [(arm, by_item[i][v]) for arm in arms for i in survivors for v in VARIANTS]
    LOGGER.info(
        f"sweeping {len(survivors)} items x {len(VARIANTS)} variants x {len(arms)} arms = {len(jobs)} prompts, {len(jobs) * run.n_samples} generations"
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(run.model, revision=run.revision or None)
    engine = load_engine(run, config.gpu_memory_utilization)

    prompts = [
        build_chat(tok, ARMS[arm], build_user_message(r["question"], r["options"], r["cued_option"], cue_name))
        for arm, r in jobs
    ]

    traces: dict[tuple[str, str], dict[str, list[Trace]]] = defaultdict(dict)
    for offset, outputs in generate_chunked(engine, prompts, run, config.chunk_size):
        records = []
        for (arm, row), output in zip(jobs[offset : offset + len(outputs)], outputs):
            generations = to_generations(output, tok, run)
            records += [
                row
                | g.model_dump()
                | {
                    "stage": "sweep",
                    "arm": arm,
                    "cue_name": cue_name,
                    "run_id": out.name,
                    "prompt_fingerprint": prompt_fingerprint(arm, cue_name),
                }
                for g in generations
            ]
            traces[(arm, row["item_id"])][row["variant"]] = [
                Trace(parsed_answer=g.parsed_answer, parse_ok=g.parse_ok, truncated=g.truncated) for g in generations
            ]
        append_jsonl(generations_path, records)

    # behavioural tracking only. the four cells need the judge and are not computed here.
    rows, per_arm = [], {}
    for arm in arms:
        placement = {v: 0 for v in CUED_VARIANTS}
        n_all = n_heldout = 0
        for item_id in survivors:
            cued = by_item[item_id]["V1"]["distractor_map"]
            results = {v: score_variant(traces[(arm, item_id)][v], cued[v], SCORING) for v in CUED_VARIANTS}
            for v in CUED_VARIANTS:
                placement[v] += results[v].tracks
            tracks_all = all(results[v].tracks for v in CUED_VARIANTS)
            tracks_heldout = all(results[v].tracks for v in SCORING.heldout_variants)
            n_all += tracks_all
            n_heldout += tracks_heldout
            rows.append(
                {"arm": arm, "item_id": item_id, "tracks_all": tracks_all, "tracks_heldout": tracks_heldout}
                | {f"tracks_{v}": results[v].tracks for v in CUED_VARIANTS}
                | {f"modal_{v}": results[v].modal_answer for v in CUED_VARIANTS}
                | {f"modal_count_{v}": results[v].modal_count for v in CUED_VARIANTS}
            )
        n = len(survivors)
        per_arm[arm] = {
            "n_items": n,
            "tracking_rate_all_placements": round(n_all / n, 4),
            "tracking_rate_heldout": round(n_heldout / n, 4),
            "per_placement": {v: round(placement[v] / n, 4) for v in CUED_VARIANTS},
            "gate_tracking_passes": n_all / n >= GATES.behavioral_tracking_min,
        }

    all_rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    n_trunc = sum(r["truncated"] for r in all_rows)
    n_parse_fail = sum(not r["parse_ok"] for r in all_rows)

    append_jsonl(out / "tracking_results.jsonl", rows)
    report_path = write_json(
        out / "tracking_report.json",
        {
            "run_id": out.name,
            "git_hash": get_git_hash(),
            "arms": arms,
            "cue_name": cue_name,
            "run": run.model_dump(),
            "n_generations": len(all_rows),
            "truncation_rate": round(n_trunc / len(all_rows), 4),
            "parse_fail_rate": round(n_parse_fail / len(all_rows), 4),
            "gate_truncation_passes": n_trunc / len(all_rows) < GATES.truncation_rate_max,
            "by_arm": per_arm,
        },
    )

    print(f"Generations: {generations_path}")
    print(f"Report: {report_path}")
    print(f"Truncation rate: {round(n_trunc / len(all_rows), 4)}, gate <{GATES.truncation_rate_max}")
    print(f"Parse failure rate: {round(n_parse_fail / len(all_rows), 4)}")
    for arm, d in per_arm.items():
        print(
            f"{arm}: tracking all placements {d['tracking_rate_all_placements']} (gate >={GATES.behavioral_tracking_min}: {d['gate_tracking_passes']}), heldout {d['tracking_rate_heldout']}, per placement {d['per_placement']}"
        )


main()
