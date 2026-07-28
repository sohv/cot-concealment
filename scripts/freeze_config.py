# dumps the frozen run config, scoring thresholds and gates to config.json before the first generation.
# uv run -m scripts.freeze_config --output_dir results/val_2026_08_11_a --seed 42 --model_id Qwen/Qwen3-14B

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.config import GATES, RUN, SCORING
from src.utils.io import get_git_hash, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    output_dir: str = "results"
    model_id: str = "Qwen/Qwen3-14B"
    revision: str = ""
    engine_version: str = ""
    run_id: str = ""
    seed: int = 42


def main():
    config = simple_parsing.parse(Config)
    run = RUN.model_copy(
        update={
            "model": config.model_id,
            "revision": config.revision,
            "engine_version": config.engine_version,
            "seed": config.seed,
        }
    )

    if not run.revision:
        LOGGER.warning("revision is empty; pin the model commit hash before generating")
    if not run.engine_version:
        LOGGER.warning("engine_version is empty; pin the vllm version before generating")

    payload = {
        "run_id": config.run_id or Path(config.output_dir).name,
        "git_hash": get_git_hash(),
        "run": run.model_dump(),
        "scoring": SCORING.model_dump(),
        "gates": GATES.model_dump(),
    }
    path = write_json(Path(config.output_dir) / "config.json", payload)
    print(f"Frozen config written to {path}")


main()
