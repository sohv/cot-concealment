# disk cache for judge calls, keyed on model plus messages. the judge stage is ~11.5k calls, so a run
# that dies partway through must not pay for the traces it already judged.

import asyncio
import hashlib
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def cache_key(model: str, messages: list[dict], **params) -> str:
    payload = json.dumps({"model": model, "messages": messages, **params}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


async def cached_completion(
    client,
    model: str,
    messages: list[dict],
    cache_dir: str | Path = "cache/judge",
    max_retries: int = 4,
    **params,
) -> str:
    """returns the response text, from disk when it is already there. retries the network call with
    backoff, which is the one failure this project retries rather than crashes on."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(model, messages, **params)}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(model=model, messages=messages, **params)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            LOGGER.warning(f"judge call failed ({type(e).__name__}: {e}), retry {attempt + 1} of {max_retries - 1}")
            await asyncio.sleep(2**attempt)
            continue
        break

    text = response.choices[0].message.content
    if text is None:
        raise ValueError(f"judge returned no content: finish_reason {response.choices[0].finish_reason}")
    path.write_text(json.dumps({"response": text, "model": model}))
    return text
