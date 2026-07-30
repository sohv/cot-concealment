# sweep: samples V0 to V3 across every framing arm for the filtered survivors, then reports behavioural
# tracking. tracking needs no judge and no api key, so it settles the tracking gate before any judging spend.
# uv run -m scripts.run_sweep --variants_path data/processed/variants_seed42.jsonl --survivors_path results/raw/filter_v1/survivors.json --output_dir results/raw/sweep_v1_resume --resume_from results/raw/sweep_v1/generations.jsonl --model_id Qwen/Qwen3-8B --max_tokens 6144 --chunk_size 50 --gpu_memory_utilization 0.95 --seed 42

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.config import CUED_VARIANTS, GATES, RUN, SCORING, VARIANTS
from src.generation.engine import generate_chunked, load_engine, to_generations
from src.generation.prompts import ARMS, build_chat, build_user_message, prompt_fingerprint
from src.metrics.bootstrap import bootstrap_proportion
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
    # comma separated generations.jsonl from earlier runs of this same sweep. prompts already complete
    # there are skipped and their rows fold into the report, so a killed run resumes instead of restarting.
    resume_from: str = ""
    # report over resume_from alone, written as interim. no engine, no gpu, safe while a run is going.
    report_only: bool = False
    seed: int = 42


def load_prior(
    paths: list[str], cue_name: str, max_tokens: int, n_samples: int
) -> dict[tuple[str, str, str], list[dict]]:
    """prior generations keyed by (arm, item_id, variant). only prompts holding all n_samples rows count
    as done, so a prompt cut off mid write is regenerated rather than half trusted."""
    rows = []
    for path in paths:
        config_path = Path(path).parent / "config.json"
        if config_path.exists():
            prior_max_tokens = json.loads(config_path.read_text())["run"]["max_tokens"]
            if prior_max_tokens != max_tokens:
                raise ValueError(f"{path} was generated at max_tokens {prior_max_tokens}, this run uses {max_tokens}")
        else:
            LOGGER.warning(f"no config.json beside {path}; the sampling params behind those rows are unverified")
        rows += read_jsonl(path)

    by_prompt: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["cue_name"] != cue_name:
            raise ValueError(f"prior row carries cue {r['cue_name']!r}, this run uses {cue_name!r}")
        expected = prompt_fingerprint(r["arm"], cue_name)
        if r["prompt_fingerprint"] != expected:
            raise ValueError(
                f"prompt strings changed since {r['arm']} was generated: row {r['prompt_fingerprint']} vs current {expected}"
            )
        by_prompt[(r["arm"], r["item_id"], r["variant"])].append(r)

    duplicated = {k: len(v) for k, v in by_prompt.items() if len(v) > n_samples}
    if duplicated:
        raise ValueError(f"{len(duplicated)} prior prompts hold more than {n_samples} rows: {duplicated}")
    partial = {k: len(v) for k, v in by_prompt.items() if len(v) < n_samples}
    if partial:
        LOGGER.warning(f"{len(partial)} prior prompts hold fewer than {n_samples} rows and will be regenerated")
    return {k: v for k, v in by_prompt.items() if len(v) == n_samples}


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
    if generations_path.exists() and not config.report_only:
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

    resume_paths = [p.strip() for p in config.resume_from.split(",") if p.strip()]
    prior = load_prior(resume_paths, cue_name, config.max_tokens, run.n_samples) if resume_paths else {}
    prior = {k: v for k, v in prior.items() if k[0] in arms and k[1] in keep}
    if config.report_only and not prior:
        raise ValueError("--report_only reports over prior rows and needs --resume_from")

    # ordered arm then item then variant: the system message is the outermost shared prefix and the
    # question stem the next, so prefix caching gets the longest possible reuse.
    jobs = [(arm, by_item[i][v]) for arm in arms for i in survivors for v in VARIANTS]
    n_planned = len(jobs)
    if config.report_only:
        jobs = []
    elif prior:
        jobs = [(arm, r) for arm, r in jobs if (arm, r["item_id"], r["variant"]) not in prior]
    LOGGER.info(
        f"{n_planned} prompts planned, {len(prior)} already complete in prior runs, {len(jobs)} to generate = {len(jobs) * run.n_samples} generations"
    )

    traces: dict[tuple[str, str], dict[str, list[Trace]]] = defaultdict(dict)
    for (arm, item_id, variant), prior_rows_for_prompt in prior.items():
        traces[(arm, item_id)][variant] = [
            Trace(parsed_answer=r["parsed_answer"], parse_ok=r["parse_ok"], truncated=r["truncated"])
            for r in prior_rows_for_prompt
        ]

    if jobs:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(run.model, revision=run.revision or None)
        engine = load_engine(run, config.gpu_memory_utilization)

        prompts = [
            build_chat(tok, ARMS[arm], build_user_message(r["question"], r["options"], r["cued_option"], cue_name))
            for arm, r in jobs
        ]

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
                    Trace(parsed_answer=g.parsed_answer, parse_ok=g.parse_ok, truncated=g.truncated)
                    for g in generations
                ]
            append_jsonl(generations_path, records)

    # behavioural tracking only. the four cells need the judge and are not computed here.
    # an arm part way through is reported over the items it has finished, never over a partial item.
    rows, per_arm = [], {}
    for arm in arms:
        complete = [i for i in survivors if all(v in traces.get((arm, i), {}) for v in CUED_VARIANTS)]
        tracks_all, tracks_heldout = [], []
        placement: dict[str, list[bool]] = {v: [] for v in CUED_VARIANTS}
        for item_id in complete:
            cued = by_item[item_id]["V1"]["distractor_map"]
            results = {v: score_variant(traces[(arm, item_id)][v], cued[v], SCORING) for v in CUED_VARIANTS}
            for v in CUED_VARIANTS:
                placement[v].append(results[v].tracks)
            tracks_all.append(all(results[v].tracks for v in CUED_VARIANTS))
            tracks_heldout.append(all(results[v].tracks for v in SCORING.heldout_variants))
            rows.append(
                {"arm": arm, "item_id": item_id, "tracks_all": tracks_all[-1], "tracks_heldout": tracks_heldout[-1]}
                | {f"tracks_{v}": results[v].tracks for v in CUED_VARIANTS}
                | {f"modal_{v}": results[v].modal_answer for v in CUED_VARIANTS}
                | {f"modal_count_{v}": results[v].modal_count for v in CUED_VARIANTS}
            )
        tracking = bootstrap_proportion(tracks_all)
        per_arm[arm] = {
            "n_items_requested": len(survivors),
            "n_items_complete": len(complete),
            "tracking_all_placements": tracking,
            "tracking_heldout": bootstrap_proportion(tracks_heldout),
            "per_placement": {v: bootstrap_proportion(placement[v]) for v in CUED_VARIANTS},
            "gate_tracking_passes": tracking["point"] is not None
            and tracking["point"] >= GATES.behavioral_tracking_min,
        }

    prior_rows = [r for rows_for_prompt in prior.values() for r in rows_for_prompt]
    new_rows = read_jsonl(generations_path) if jobs and generations_path.exists() else []
    all_rows = prior_rows + new_rows
    n_trunc = sum(r["truncated"] for r in all_rows)
    n_parse_fail = sum(not r["parse_ok"] for r in all_rows)

    if not config.report_only:
        append_jsonl(out / "tracking_results.jsonl", rows)
    report_path = write_json(
        out / ("interim_tracking_report.json" if config.report_only else "tracking_report.json"),
        {
            "run_id": out.name,
            "git_hash": get_git_hash(),
            "is_interim": config.report_only,
            "arms": arms,
            "cue_name": cue_name,
            "run": run.model_dump(),
            "resumed_from": resume_paths,
            "n_generations": len(all_rows),
            "n_generations_planned": n_planned * run.n_samples,
            "n_generations_prior": len(prior_rows),
            "n_generations_new": len(new_rows),
            "generations_by_arm": dict(Counter(r["arm"] for r in all_rows)),
            "truncation_rate": round(n_trunc / len(all_rows), 4),
            "parse_fail_rate": round(n_parse_fail / len(all_rows), 4),
            "parse_method": dict(Counter(r["parse_method"] for r in all_rows)),
            "gate_truncation_passes": n_trunc / len(all_rows) < GATES.truncation_rate_max,
            "gate_behavioral_tracking_min": GATES.behavioral_tracking_min,
            "by_arm": per_arm,
        },
    )

    print(f"Report: {report_path}")
    if new_rows:
        print(f"Generations: {generations_path}")
    print(f"Generations: {len(all_rows)} of {n_planned * run.n_samples} planned ({len(prior_rows)} resumed)")
    print(f"Truncation rate: {round(n_trunc / len(all_rows), 4)}, gate <{GATES.truncation_rate_max}")
    print(f"Parse failure rate: {round(n_parse_fail / len(all_rows), 4)}")
    for arm, d in per_arm.items():
        tracking, heldout = d["tracking_all_placements"], d["tracking_heldout"]
        print(
            f"{arm}: n={d['n_items_complete']}/{d['n_items_requested']}, tracking all placements {tracking['point']} {tracking['ci']} "
            f"(gate >={GATES.behavioral_tracking_min}: {d['gate_tracking_passes']}), heldout {heldout['point']} {heldout['ci']}, "
            f"per placement {({v: p['point'] for v, p in d['per_placement'].items()})}"
        )


main()
