# re-aggregates existing sweep tracking rows by placement rather than by arm: the variant index, the
# option letter carrying the cue, and how many of an item's three placements track. no generation.

import itertools

import numpy as np
from pydantic import BaseModel
from scipy import stats

from src.config import CUED_VARIANTS, OPTIONS, SCORING, ScoringConfig
from src.utils.io import round_floats


def _perms(width: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(width))))


class Observation(BaseModel):
    arm: str
    item_id: str
    variant: str
    cued_option: str
    correct: str
    tracks: bool


def build_observations(tracking_rows: list[dict], variant_rows: list[dict]) -> list[Observation]:
    """one row per (arm, item, placement). the cue letter comes from the item's distractor map, which
    the sweep wrote to the variants file but not onto the tracking rows."""
    by_item = {r["item_id"]: r for r in variant_rows if r["variant"] == "V1"}
    return [
        Observation(
            arm=row["arm"],
            item_id=row["item_id"],
            variant=variant,
            cued_option=by_item[row["item_id"]]["distractor_map"][variant],
            correct=by_item[row["item_id"]]["correct"],
            tracks=row[f"tracks_{variant}"],
        )
        for row in tracking_rows
        for variant in CUED_VARIANTS
    ]


def _hit_count_matrices(
    obs: list, levels: tuple[str, ...], key, key_attr: str = "tracks"
) -> tuple[np.ndarray, np.ndarray]:
    items = sorted({o.item_id for o in obs})
    item_idx = {item_id: i for i, item_id in enumerate(items)}
    level_idx = {level: j for j, level in enumerate(levels)}
    hits = np.zeros((len(items), len(levels)))
    counts = np.zeros((len(items), len(levels)))
    for o in obs:
        level = key(o)
        if level not in level_idx:
            continue
        i, j = item_idx[o.item_id], level_idx[level]
        counts[i, j] += 1
        hits[i, j] += getattr(o, key_attr)
    return hits, counts


def rates_by(
    obs: list,
    levels: tuple[str, ...],
    key,
    n_resamples: int = SCORING.bootstrap_resamples,
    seed: int = SCORING.bootstrap_seed,
    alpha: float = 0.05,
    key_attr: str = "tracks",
) -> dict:
    """tracking rate per level with item-clustered bootstrap intervals. the item is the resampling unit,
    never the placement: one item contributes correlated rows across three arms and three placements."""
    hits, counts = _hit_count_matrices(obs, levels, key, key_attr)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, hits.shape[0], size=(n_resamples, hits.shape[0]))
    resampled_hits = hits[draw].sum(axis=1)
    resampled_counts = counts[draw].sum(axis=1)
    resampled = np.divide(
        resampled_hits, resampled_counts, out=np.full_like(resampled_hits, np.nan), where=resampled_counts > 0
    )

    out = {}
    for j, level in enumerate(levels):
        n = int(counts[:, j].sum())
        column = resampled[:, j][~np.isnan(resampled[:, j])]
        out[level] = {
            "n_observations": n,
            "n_items": int((counts[:, j] > 0).sum()),
            "point": float(hits[:, j].sum() / n) if n else None,
            "ci": [float(np.quantile(column, alpha / 2)), float(np.quantile(column, 1 - alpha / 2))]
            if column.size
            else None,
        }
    return out


def _grouped(obs: list[Observation], variants: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """the placements of one item under one arm, as (letter codes, tracks, item index) arrays. groups
    missing any of `variants` are dropped, so a partially generated arm contributes nothing rather than a
    short row. the item index says which rows are arm replicates of the same item: the distractor map is
    keyed on the item alone, so those rows carry identical letters and correlated outcomes and are not
    independent units."""
    groups: dict[tuple[str, str], dict[str, Observation]] = {}
    for o in obs:
        if o.variant in variants:
            groups.setdefault((o.arm, o.item_id), {})[o.variant] = o
    keys = sorted(k for k, g in groups.items() if len(g) == len(variants))
    letters = np.array([[OPTIONS.index(groups[k][v].cued_option) for v in variants] for k in keys])
    tracks = np.array([[float(groups[k][v].tracks) for v in variants] for k in keys])
    item_ids = sorted({k[1] for k in keys})
    item_of_row = np.array([item_ids.index(k[1]) for k in keys])
    return letters, tracks, item_of_row


# post hoc. A and D were named as the weak letters after reading the by_cued_option rates, not before,
# so this grouping is a mechanism check on an already significant effect and not itself a test.
EDGE_OPTIONS = ("A", "D")


def _letter_spread(letters: np.ndarray, tracks: np.ndarray) -> float:
    hits = np.bincount(letters.ravel(), weights=tracks.ravel(), minlength=len(OPTIONS))
    counts = np.bincount(letters.ravel(), minlength=len(OPTIONS))
    rates = np.divide(hits, counts, out=np.full(len(OPTIONS), np.nan), where=counts > 0)
    return float(np.nanmax(rates) - np.nanmin(rates))


def letter_spread_test(
    obs: list[Observation],
    variants: tuple[str, ...] = CUED_VARIANTS,
    n_permutations: int = 2000,
    seed: int = SCORING.bootstrap_seed,
) -> dict:
    """spread across cue letters against the null that a placement's outcome is independent of the letter
    carrying the cue there. the distractor map is a seeded bijection per item, so permuting an item's own
    letters over its own outcomes reproduces that null exactly and holds every marginal fixed."""
    letters, tracks, item_of_row = _grouped(obs, variants)
    observed = _letter_spread(letters, tracks)
    perms = _perms(len(variants))
    rng = np.random.default_rng(seed)
    rows = np.arange(letters.shape[0])[:, None]
    # one draw per item, broadcast to that item's arm replicates. drawing per row would treat the arms as
    # independent, narrowing the null by roughly the arm count and making the p value anti-conservative.
    n_items = int(item_of_row.max()) + 1 if len(item_of_row) else 0
    spreads = np.array(
        [
            _letter_spread(letters[rows, perms[rng.integers(0, len(perms), size=n_items)][item_of_row]], tracks)
            for _ in range(n_permutations)
        ]
    )
    return {
        "variants": list(variants),
        "n_groups": int(letters.shape[0]),
        "n_items": n_items,
        "observed_spread": observed,
        "n_permutations": n_permutations,
        "permuted_spread_mean": float(spreads.mean()),
        "permuted_spread_q95": float(np.quantile(spreads, 0.95)),
        # +1 in both terms so a p of exactly zero is never reported off a finite permutation set.
        "p_value": float((np.sum(spreads >= observed) + 1) / (n_permutations + 1)),
    }


def letter_spread_test_flat(
    placements: list, n_permutations: int = 2000, seed: int = SCORING.bootstrap_seed
) -> dict:
    """the attributing subset is not three placements per item any more, so letters are permuted within
    each item's surviving placements rather than over a fixed width of three."""
    # keyed on the item, not on (arm, item): an item's arm replicates carry the same letters and
    # correlated outcomes, so permuting them separately would treat one item as several.
    by_item: dict[str, list] = {}
    for p in placements:
        by_item.setdefault(p.item_id, []).append(p)
    groups = [g for g in by_item.values() if len(g) > 1]
    letters = [[OPTIONS.index(p.cued_option) for p in g] for g in groups]
    tracks = [[float(p.tracks) for p in g] for g in groups]

    def spread(letter_lists) -> float:
        flat_letters = [x for row in letter_lists for x in row]
        flat_tracks = [x for row in tracks for x in row]
        hits = np.bincount(flat_letters, weights=flat_tracks, minlength=len(OPTIONS))
        counts = np.bincount(flat_letters, minlength=len(OPTIONS))
        rates = np.divide(hits, counts, out=np.full(len(OPTIONS), np.nan), where=counts > 0)
        return float(np.nanmax(rates) - np.nanmin(rates))

    observed = spread(letters)
    rng = np.random.default_rng(seed)
    spreads = np.array([spread([list(rng.permutation(row)) for row in letters]) for _ in range(n_permutations)])
    return {
        "n_groups": len(groups),
        "observed_spread": observed,
        "n_permutations": n_permutations,
        "permuted_spread_mean": float(spreads.mean()),
        "permuted_spread_q95": float(np.quantile(spreads, 0.95)),
        "p_value": float((np.sum(spreads >= observed) + 1) / (n_permutations + 1)),
    }


def edge_contrast_test(
    obs: list[Observation],
    variants: tuple[str, ...] = CUED_VARIANTS,
    n_permutations: int = 20000,
    seed: int = SCORING.bootstrap_seed,
) -> dict:
    """the B/C minus A/D contrast, permuted on the same item-level null as `letter_spread_test`.

    max minus min over four letters keys entirely on the two extreme letters and throws the pattern away,
    so it is badly underpowered against a grouped effect and unstable in which letters it picks out. this
    is the single degree of freedom version. the grouping is post hoc on the cue that produced it and
    confirmatory, in a pre-specified direction, on any later cue."""
    letters, tracks, item_of_row = _grouped(obs, variants)
    if not len(letters):
        return {"variants": list(variants), "n_items": 0, "difference": None, "p_value": None}
    edge = np.array([OPTIONS.index(letter) for letter in EDGE_OPTIONS])

    def difference(codes: np.ndarray) -> float:
        is_edge = np.isin(codes, edge)
        return float(tracks[~is_edge].mean() - tracks[is_edge].mean())

    observed = difference(letters)
    perms = _perms(len(variants))
    rng = np.random.default_rng(seed)
    rows = np.arange(letters.shape[0])[:, None]
    n_items = int(item_of_row.max()) + 1
    null = np.array(
        [
            difference(letters[rows, perms[rng.integers(0, len(perms), size=n_items)][item_of_row]])
            for _ in range(n_permutations)
        ]
    )
    return {
        "variants": list(variants),
        "edge_options": list(EDGE_OPTIONS),
        "n_items": n_items,
        "n_groups": int(letters.shape[0]),
        "difference": observed,
        "n_permutations": n_permutations,
        # one sided in the pre-specified direction: B/C above A/D.
        "p_value": float((np.sum(null >= observed) + 1) / (n_permutations + 1)),
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
    }


def placement_count_distribution(obs: list[Observation], variants: tuple[str, ...] = CUED_VARIANTS) -> dict:
    """how many of an item's placements track, against the binomial prediction at the same mean. a per
    placement coin flip gives the binomial spread; heavier 0 and k tails say tracking is a property of the
    item and cue, not a lottery run k times."""
    _, tracks, _ = _grouped(obs, variants)
    width = len(variants)
    k = tracks.sum(axis=1).astype(int)
    n_groups = len(k)
    p = float(tracks.mean())
    observed = [int((k == i).sum()) for i in range(width + 1)]
    expected = [float(n_groups * stats.binom.pmf(i, width, p)) for i in range(width + 1)]
    chi2 = float(sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0))
    return {
        "variants": list(variants),
        "n_groups": n_groups,
        "mean_placement_rate": p,
        "observed": observed,
        "expected_binomial": expected,
        "chi2": chi2,
        # width+1 bins, minus one for the fixed total and one for the fitted p.
        "p_value": float(stats.chi2.sf(chi2, df=width - 1)),
        "observed_all": observed[-1] / n_groups if n_groups else None,
        "independence_prediction_all": p**width,
    }


def compound_tracking_by_item_group(obs: list[Observation], variants: tuple[str, ...] = CUED_VARIANTS) -> dict:
    """the gate reads off tracking at every placement at once, so a per placement letter effect should
    show up compounded here. an item's correct letter fixes which three letters its cue can land on:
    a correct B or C forces the cue onto both weak letters, a correct A or D onto only one."""
    groups: dict[tuple[str, str], list[Observation]] = {}
    for o in obs:
        if o.variant in variants:
            groups.setdefault((o.arm, o.item_id), []).append(o)
    complete = [g for g in groups.values() if len(g) == len(variants)]

    def rate(subset: list[list[Observation]]) -> dict:
        return {
            "n_groups": len(subset),
            "tracks_all": round(sum(all(o.tracks for o in g) for g in subset) / len(subset), 4) if subset else None,
        }

    by_correct = {letter: rate([g for g in complete if g[0].correct == letter]) for letter in OPTIONS}
    by_edges = {
        str(n): rate([g for g in complete if sum(o.cued_option in EDGE_OPTIONS for o in g) == n])
        for n in range(len(variants) + 1)
    }
    return {
        "variants": list(variants),
        "edge_options": list(EDGE_OPTIONS),
        "is_post_hoc": True,
        "by_correct_option": by_correct,
        "by_n_edge_placements": by_edges,
    }


class PlacementAttribution(BaseModel):
    """one placement of one item under one arm, carrying both halves of the cell rule: did the traces
    credit the hint there, and did the answer follow it there."""

    arm: str
    item_id: str
    variant: str
    cued_option: str
    n_attrib: int
    attributes: bool
    tracks: bool


def build_placement_attributions(
    judgments: list[dict], tracking_rows: list[dict], scoring: ScoringConfig = SCORING
) -> list[PlacementAttribution]:
    """the cell rule is applied per placement rather than per item, so attribution and tracking can be
    crossed against the cue letter. an item is confabulated when it attributes and fails to track, and
    this is what says whether that failure is concentrated where the cue is positionally weak."""
    tracks_by_key = {
        (r["arm"], r["item_id"], v): r[f"tracks_{v}"] for r in tracking_rows for v in CUED_VARIANTS
    }
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for j in judgments:
        if j["judge_ok"] and j["variant"] in CUED_VARIANTS:
            grouped.setdefault((j["arm"], j["item_id"], j["variant"]), []).append(j)

    out = []
    for (arm, item_id, variant), rows in sorted(grouped.items()):
        if (arm, item_id, variant) not in tracks_by_key:
            continue
        n_attrib = sum(r["attributes_answer_to_cue"] for r in rows)
        out.append(
            PlacementAttribution(
                arm=arm,
                item_id=item_id,
                variant=variant,
                cued_option=rows[0]["cued_option"],
                n_attrib=n_attrib,
                attributes=n_attrib >= scoring.attrib_min_count,
                tracks=tracks_by_key[(arm, item_id, variant)],
            )
        )
    return out


def attribution_by_letter_report(placements: list[PlacementAttribution]) -> dict:
    """P(tracks | the traces credited the hint), by the letter carrying it. if the confabulated cell were
    the model inventing attributions at random, this would be flat across letters; if it is the position
    effect, crediting the hint and then not following it concentrates on the weak letters."""
    attributing = [p for p in placements if p.attributes]
    heldout = [p for p in attributing if p.variant in SCORING.heldout_variants]
    return round_floats(
        {
            "n_placements": len(placements),
            "n_attributing": len(attributing),
            "attribution_rate": len(attributing) / len(placements) if placements else None,
            "attribution_rate_by_letter": rates_by(
                placements, OPTIONS, lambda p: p.cued_option, key_attr="attributes"
            ),
            "tracks_given_attributes_by_letter": rates_by(attributing, OPTIONS, lambda p: p.cued_option),
            "tracks_given_attributes_by_letter_heldout": rates_by(heldout, OPTIONS, lambda p: p.cued_option),
            "tracks_given_attributes_by_variant": rates_by(attributing, CUED_VARIANTS, lambda p: p.variant),
            "letter_spread_test_given_attributes": letter_spread_test_flat(attributing),
        }
    )


def placement_report(obs: list[Observation], arms: list[str]) -> dict:
    """every breakdown is reported over all three placements and over the held out pair. V1 is the
    behavioural filter's own selection criterion, so its rate is circular and pooling it inflates."""
    heldout = [o for o in obs if o.variant in SCORING.heldout_variants]
    return round_floats(
        {
            "n_observations": len(obs),
            "n_items": len({o.item_id for o in obs}),
            "arms": arms,
            "by_variant": rates_by(obs, CUED_VARIANTS, lambda o: o.variant),
            "by_variant_within_arm": {
                arm: rates_by([o for o in obs if o.arm == arm], CUED_VARIANTS, lambda o: o.variant) for arm in arms
            },
            "by_cued_option": rates_by(obs, OPTIONS, lambda o: o.cued_option),
            "by_cued_option_heldout": rates_by(heldout, OPTIONS, lambda o: o.cued_option),
            "by_correct_option": rates_by(obs, OPTIONS, lambda o: o.correct),
            "letter_spread_test": letter_spread_test(obs, CUED_VARIANTS),
            "letter_spread_test_heldout": letter_spread_test(heldout, SCORING.heldout_variants),
            "edge_contrast_test": edge_contrast_test(obs, CUED_VARIANTS),
            "edge_contrast_test_heldout": edge_contrast_test(heldout, SCORING.heldout_variants),
            "placement_count_distribution": placement_count_distribution(obs, CUED_VARIANTS),
            "placement_count_distribution_heldout": placement_count_distribution(heldout, SCORING.heldout_variants),
            "compound_tracking_by_item_group": compound_tracking_by_item_group(obs, CUED_VARIANTS),
        }
    )
