# judge stage: reads each sweep think block, asks whether it credits the planted scoring condition and
# which option letter it says that condition names, then assigns the four cells with metrics.scoring.
# no gpu; this calls an external judge api through openrouter.
# uv run -m scripts.run_judge --generations_paths results/raw/sweep_v1/generations.jsonl,results/raw/sweep_v1_resume/generations.jsonl --output_dir results/raw/judge_v1 --judge_model openai/gpt-4.1-mini --seed 42

import asyncio
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import CUED_VARIANTS, GATES, SCORING
from src.generation.judge import JUDGE_SYSTEM, build_judge_messages, parse_judgment, span_match_ok
from src.metrics.bootstrap import ScoredItem, bootstrap_cells, false_attribution_rate, sensitivity_grid
from src.metrics.scoring import Judgment, Trace, assign_cell
from src.utils.cache import cached_completion
from src.utils.io import append_jsonl, get_git_hash, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    # comma separated. a resumed sweep splits its rows across the prior run's file and the resume's,
    # so the full sweep is the union of both and judging either alone silently drops arms.
    generations_paths: str = ""
    output_dir: str = "results/raw/judge_v1"
    judge_model: str = ""  # falls back to JUDGE_MODEL in .env
    arms: str = ""  # comma separated; defaults to every arm present in the generations
    max_concurrent: int = 20
    num_tasks: int | None = None  # cap on items, for a cheap smoke run before the full spend
    cache_dir: str = "cache/judge"
    # deterministic annotation. the judge is an instrument, so it is read at temperature 0 and the
    # reading must not move between the hand check and the run it is checking.
    temperature: float = 0.0
    seed: int = 42


async def judge_rows(rows: list[dict], client, model: str, config: Config) -> list[dict]:
    semaphore = asyncio.Semaphore(config.max_concurrent)
    progress = tqdm(total=len(rows), desc="judging")

    async def judge_one(row: dict) -> dict:
        async with semaphore:
            messages = build_judge_messages(row["question"], row["options"], row["think_text"])
            text = await cached_completion(
                client,
                model=model,
                messages=messages,
                cache_dir=config.cache_dir,
                temperature=config.temperature,
                response_format={"type": "json_object"},
            )
        progress.update(1)
        judgment, error = parse_judgment(text)
        if error:
            LOGGER.error(f"judge parse failed for {row['item_id']}/{row['arm']}/{row['variant']}: {error}")
        return {
            "item_id": row["item_id"],
            "arm": row["arm"],
            "variant": row["variant"],
            "sample_idx": row["sample_idx"],
            "cued_option": row["cued_option"],
            "parsed_answer": row["parsed_answer"],
            "judge_model": model,
            "judge_ok": judgment is not None,
            "judge_error": error,
            "raw_response": None if judgment else text[:500],
            **(judgment.model_dump() if judgment else {}),
            "span_match_ok": span_match_ok(judgment, row["think_text"]) if judgment else None,
        }

    results = await asyncio.gather(*[judge_one(r) for r in rows])
    progress.close()
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    load_dotenv()
    config = simple_parsing.parse(Config)
    if not config.generations_paths:
        raise ValueError("--generations_paths is required")

    out = Path(config.output_dir)
    judgments_path = out / "judgments.jsonl"
    if judgments_path.exists():
        raise FileExistsError(f"{judgments_path} exists; the write appends, so use a new output_dir")

    model = config.judge_model or os.environ.get("JUDGE_MODEL", "")
    if not model:
        raise ValueError("--judge_model is required, or set JUDGE_MODEL in .env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set; copy .env.example to .env and fill it in")

    paths = [p.strip() for p in config.generations_paths.split(",") if p.strip()]
    rows = [r for path in paths for r in read_jsonl(path)]
    keys = Counter((r["arm"], r["item_id"], r["variant"], r["sample_idx"]) for r in rows)
    duplicated = [k for k, n in keys.items() if n > 1]
    if duplicated:
        raise ValueError(f"{len(duplicated)} generations appear in more than one input file, e.g. {duplicated[:3]}")
    arms = [a.strip() for a in config.arms.split(",") if a.strip()] or sorted({r["arm"] for r in rows})
    item_ids = sorted({r["item_id"] for r in rows})
    if config.num_tasks:
        item_ids = item_ids[: config.num_tasks]
    keep = set(item_ids)

    # truncated traces have no closing think block and are excluded from every cell by score_variant,
    # so judging them is spend with nowhere to land. they are still counted in the report.
    n_truncated = sum(r["truncated"] for r in rows if r["item_id"] in keep and r["arm"] in arms)
    to_judge = [r for r in rows if r["item_id"] in keep and r["arm"] in arms and not r["truncated"] and r["think_text"]]
    LOGGER.info(
        f"judging {len(to_judge)} traces over {len(item_ids)} items and {len(arms)} arms with {model}, "
        f"{n_truncated} truncated traces skipped"
    )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        default_headers={
            k: v
            for k, v in {
                "HTTP-Referer": os.environ.get("OPENROUTER_APP_URL", ""),
                "X-Title": os.environ.get("OPENROUTER_APP_TITLE", ""),
            }.items()
            if v
        },
    )
    judgments = asyncio.run(judge_rows(to_judge, client, model, config))
    append_jsonl(judgments_path, judgments)

    by_trace = {(j["arm"], j["item_id"], j["variant"], j["sample_idx"]): j for j in judgments}
    traces: dict[tuple[str, str], dict[str, list[Trace]]] = defaultdict(lambda: defaultdict(list))
    v0_judgments: list[Judgment] = []
    for r in rows:
        if r["item_id"] not in keep or r["arm"] not in arms:
            continue
        j = by_trace.get((r["arm"], r["item_id"], r["variant"], r["sample_idx"]))
        judgment = (
            Judgment(**{k: j[k] for k in ("mentions_cue", "named_option", "attributes_answer_to_cue", "evidence_span")})
            if j and j["judge_ok"]
            else None
        )
        if r["variant"] == "V0":
            if judgment and not r["truncated"]:
                v0_judgments.append(judgment)
            continue
        traces[(r["arm"], r["item_id"])][r["variant"]].append(
            Trace(parsed_answer=r["parsed_answer"], parse_ok=r["parse_ok"], truncated=r["truncated"], judgment=judgment)
        )

    cell_rows, per_arm = [], {}
    variants_by_item = {r["item_id"]: r["distractor_map"] for r in rows if r["variant"] == "V1"}
    for arm in arms:
        scored = [
            ScoredItem(item_id=i, traces_by_variant=traces[(arm, i)], cued=variants_by_item[i])
            for i in item_ids
            if all(v in traces.get((arm, i), {}) for v in CUED_VARIANTS)
        ]
        results = {s.item_id: assign_cell(s.traces_by_variant, s.cued, SCORING) for s in scored}
        cell_rows += [{"arm": arm, "item_id": i, **r.model_dump(exclude={"per_variant"})} for i, r in results.items()]
        per_arm[arm] = {
            "n_items": len(scored),
            "bootstrap": bootstrap_cells([r.cell for r in results.values()]),
            "sensitivity_grid": sensitivity_grid(scored, SCORING),
        }

    n_ok = sum(j["judge_ok"] for j in judgments)
    spans = [j["span_match_ok"] for j in judgments if j["span_match_ok"] is not None]
    false_attrib = false_attribution_rate(v0_judgments)

    append_jsonl(out / "cell_results.jsonl", cell_rows)
    report_path = write_json(
        out / "judge_report.json",
        {
            "run_id": out.name,
            "git_hash": get_git_hash(),
            "generations_paths": paths,
            "n_generations_read": len(rows),
            "judge_model": model,
            "judge_temperature": config.temperature,
            "judge_system_prompt": JUDGE_SYSTEM,
            "arms": arms,
            "scoring": SCORING.model_dump(),
            "n_traces_judged": len(judgments),
            "n_traces_truncated_skipped": n_truncated,
            "judge_parse_ok_rate": round(n_ok / len(judgments), 4) if judgments else None,
            "judge_errors": dict(Counter(j["judge_error"] for j in judgments if j["judge_error"])),
            "span_match_rate": round(sum(spans) / len(spans), 4) if spans else None,
            "false_attribution": {
                "n_v0_traces": len(v0_judgments),
                "rate": false_attrib,
                "gate_max": GATES.false_attribution_max,
                "gate_passes": false_attrib is not None and false_attrib < GATES.false_attribution_max,
            },
            "by_arm": per_arm,
        },
    )

    print(f"Judgments: {judgments_path}")
    print(f"Cells: {out / 'cell_results.jsonl'}")
    print(f"Report: {report_path}")
    print(f"Judged {len(judgments)} traces, parse ok {round(n_ok / len(judgments), 4) if judgments else None}")
    print(
        f"False attribution on V0: {false_attrib} over {len(v0_judgments)} traces, "
        f"gate <{GATES.false_attribution_max}: {false_attrib is not None and false_attrib < GATES.false_attribution_max}"
    )
    for arm, d in per_arm.items():
        b = d["bootstrap"]
        cells = ", ".join(f"{c} {b['cells'][c]['count']}" for c in SCORING.cells)
        cf = b["corrected_faithfulness"]
        print(f"{arm}: n={d['n_items']}, {cells}, corrected faithfulness {cf['point']} {cf['ci']}")
    print(
        f"\nAnalyse with: uv run -m scripts.analyze_cells --judge_dir {out} "
        f"--tracking_results_path results/raw/sweep_v1_resume/tracking_results.jsonl --output_dir results/analysis/cells_v1"
    )


main()
