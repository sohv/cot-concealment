# vllm wrapper. batched inference is not bitwise deterministic, so the sample index is a bookkeeping
# label only and the realized output text is recorded for every generation.

import logging

from pydantic import BaseModel

from src.config import RunConfig
from src.generation.parsing import parse_answer, split_trace

LOGGER = logging.getLogger(__name__)


class Generation(BaseModel):
    sample_idx: int
    think_text: str | None
    content_text: str
    parsed_answer: str | None
    parse_ok: bool
    truncated: bool
    n_think_tokens: int
    n_total_tokens: int
    finish_reason: str | None = None


def load_engine(run: RunConfig, gpu_memory_utilization: float = 0.90):
    from vllm import LLM

    if not run.revision:
        LOGGER.warning("model revision is unpinned; the run is not reproducible")
    return LLM(
        model=run.model,
        revision=run.revision or None,
        dtype=run.dtype,
        max_model_len=run.max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=run.seed,
    )


def sampling_params(run: RunConfig):
    from vllm import SamplingParams

    return SamplingParams(
        n=run.n_samples,
        temperature=run.temperature,
        top_p=run.top_p,
        top_k=run.top_k,
        min_p=run.min_p,
        max_tokens=run.max_tokens,
    )


def to_generations(output, tok, run: RunConfig) -> list[Generation]:
    """splits every sample of one request into its think and content blocks."""
    generations = []
    for sample_idx, completion in enumerate(output.outputs):
        token_ids = list(completion.token_ids)
        think_text, content_text, truncated = split_trace(token_ids, tok, run.think_end_token_id)
        parsed_answer, parse_ok = parse_answer(content_text)
        n_think = 0 if think_text is None else token_ids.index(run.think_end_token_id)
        generations.append(
            Generation(
                sample_idx=sample_idx,
                think_text=think_text,
                content_text=content_text,
                parsed_answer=parsed_answer,
                parse_ok=parse_ok,
                truncated=truncated,
                n_think_tokens=n_think,
                n_total_tokens=len(token_ids),
                finish_reason=completion.finish_reason,
            )
        )
    return generations


def generate(engine, prompts: list[str], run: RunConfig) -> list:
    LOGGER.info(f"generating {len(prompts)} prompts x {run.n_samples} samples")
    return engine.generate(prompts, sampling_params(run))
