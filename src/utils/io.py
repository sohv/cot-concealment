# structured output helpers: incremental jsonl during generation, indented json for config dumps,
# parquet conversion at the end of a stage.

import json
import subprocess
from pathlib import Path
from typing import Any

FLOAT_PLACES = 4


def round_floats(obj: Any, places: int = FLOAT_PLACES) -> Any:
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, dict):
        return {k: round_floats(v, places) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, places) for v in obj]
    return obj


def append_jsonl(path: str | Path, records: list[dict]) -> Path:
    """appends one json object per line so a crash mid stage does not cost the generations already made."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(round_floats(record)) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_floats(obj), indent=2, default=str))
    return path


def jsonl_to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> Path:
    import pandas as pd

    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(read_jsonl(jsonl_path)).to_parquet(parquet_path, index=False)
    return parquet_path


def get_git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]
