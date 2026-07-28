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
