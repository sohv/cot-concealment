# builds the four variants per item and the per item distractor mapping. the mapping is derived from
# seed plus item_id so it is reproducible from the config alone, never from ambient rng state.

import hashlib
import random

from pydantic import BaseModel, model_validator

from src.config import CUED_VARIANTS, OPTIONS

ARITHMETIC_MARKERS = ("mathematic", "arithmetic", "algebra", "statistics", "physics", "accounting")


class Item(BaseModel):
    item_id: str
    source: str
    question: str
    options: dict[str, str]
    correct: str

    @model_validator(mode="after")
    def check(self):
        if set(self.options) != set(OPTIONS):
            raise ValueError(f"{self.item_id}: options must be exactly {OPTIONS}")
        if self.correct not in OPTIONS:
            raise ValueError(f"{self.item_id}: correct {self.correct!r} is not an option letter")
        if any(m in self.source.lower() for m in ARITHMETIC_MARKERS):
            raise ValueError(
                f"{self.item_id}: arithmetic source {self.source!r} gives a route to the answer independent of the hint"
            )
        return self


class Variant(BaseModel):
    item_id: str
    source: str
    question: str
    options: dict[str, str]
    correct: str
    distractor_map: dict[str, str]
    variant: str
    cued_option: str | None


def distractor_map(item_id: str, correct: str, seed: int) -> dict[str, str]:
    """assigns the three wrong options to V1, V2 and V3 in an order that depends on the item, so a
    position artefact in the source data does not become a placement effect."""
    distractors = [letter for letter in OPTIONS if letter != correct]
    digest = hashlib.md5(f"{seed}:{item_id}".encode()).hexdigest()
    random.Random(int(digest, 16)).shuffle(distractors)
    return dict(zip(CUED_VARIANTS, distractors))


def build_variants(item: Item, seed: int) -> list[Variant]:
    mapping = distractor_map(item.item_id, item.correct, seed)
    base = item.model_dump() | {"distractor_map": mapping}
    return [Variant(**base, variant=v, cued_option=mapping.get(v)) for v in ("V0",) + CUED_VARIANTS]
