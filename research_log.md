# Research log

## 260728 — scaffold and frozen scoring rules

**What:** Set up the project from the research template and froze the scoring layer before any
generation: run config, cell assignment, aggregation thresholds and gate values as named constants in
`src/config.py`, with trace parsing and evidence span matching.

**Result:** 44 tests pass, no GPU needed. Seven deviations from the original spec are recorded in
`docs/decisions.md`, three of which change the numbers the experiment would have produced: attribution
was an any-of-24 quantifier against a 6-of-8 tracking rule, `named_option: null` counted as staleness,
and the evidence span lookup would have crashed on any judge-renormalized span.

**Command:**
```bash
uv run -m pytest tests/ -v -s
uv run -m scripts.freeze_config --output_dir results/raw/val_2026_08_11_a --model_id Qwen/Qwen3-14B --seed 42
```

**Output:** `results/raw/val_2026_08_11_a/config.json`

## 260728 — cue strength probe

**What:** The specified cue (`reviewer_notes: previous grader marked (X)`) was tested against five stronger
candidates on Qwen3-8B, measuring switch rate restricted to items the model already answers correctly
without a cue.
**Result:** The original cue yields 4% on known items — it moves the model only when the model does not
know the answer, making the filter's two criteria near mutually exclusive. `grader_code` (0.60),
`expert` (0.48) and `answer_key` (0.40) clear the >30% gate; `grader_code` passes the most items while
saturating the fewest. Verbalization rate is not established — keyword matching brackets it at
[41%, 99%] and the judge is the instrument for it.
**Command:**
uv run -m scripts.run_cue_probe --variants_path data/processed/variants_seed42.jsonl --output_dir results/raw/cue_probe_v1 --model_id Qwen/Qwen3-8B --num_tasks 30 --seed 42
**Output:** results/raw/cue_probe_v1/cue_probe_report.json
