# SPAR handover

What the validation pilot established, and what it means for the main study. Written 260831, after both
cues finished. Read this before designing your cue.

The pilot is the "Validation (already piloted)" box in the project proposal. It ran the cue-swap design
end to end on Qwen3-8B with two hint types and found that the instrument works, that one of its headline
numbers is an artifact, and — most importantly for you — **that the main study's headline quantity has a
ceiling problem that will silently produce zeros if cue selection does not change.**

---

## 1. The finding that should change the design

**The disclosure gap is unmeasurable when disclosure is already at ceiling.**

The project's headline quantity is the difference in disclosure rate between a model that believes its
reasoning is monitored and one that does not. The pilot ran exactly that manipulation — `C2_monitored`,
`C0_bare`, `C3_neutral_private` — on two hint types and got:

- verbalization 0.925–0.979 in every arm of both hints
- the silent influence cell empty: 0 of 120 and 0 of 93 items
- paired monitored-vs-private difference in corrected faithfulness of **exactly 0.0 [0.0, 0.0]**

That is not a null result about monitoring. There is no room for disclosure to *increase* from 0.95, so
the manipulation cannot move the number regardless of whether the model would conceal under other
conditions. A cue that is already fully disclosed cannot measure a disclosure gap.

**Consequence.** Cue selection currently screens on switch rate — does the cue move the answer. It must
also screen on **baseline verbalization being well below ceiling**. Run the cue probe, but record the
mention rate alongside the switch rate, and reject cues that are disclosed above roughly 0.8. This is a
cheap pre-cohort measurement and it gates everything downstream.

---

## 2. What the probe already tells you about cues

Six cues, 30 items, switch rate measured on the 25 items the model already answers correctly unaided
(`results/raw/cue_probe_v1/`):

| Cue | Filter yield on known items |
|---|---|
| `grader_code` (scoring assertion) | 0.60 |
| `expert` (subject-matter expert asserts) | 0.48 |
| `answer_key` | 0.40 |
| `consensus` (94% of specialists chose X) | 0.12 |
| `reviewer_note` (originally specified) | 0.04 |
| `user_belief` (sycophancy) | 0.04 |

Two things to take from this.

**The originally specified cue does not work on an 8B model.** It moves the model only where the model
does not know the answer, so the filter's two criteria are near mutually exclusive. Do not start from it.

**The sycophancy cue is the weakest of all six.** This is the awkward one: `user_belief` is where
concealment is most plausible — a model has some reason to be coy about deferring to a user's opinion —
and it is the cue no item pool can be built from at 0.04 yield. If cue type (iii) is to survive as a
domain, it needs either a stronger sycophancy framing, a relaxed filter for that domain specifically, or
an explicit decision up front that the domain reports a filter-failure result. Decide this before the
mentee starts, not in week six.

Note also that probe yields overestimate realised yields by roughly 1.5×: `grader_code` probed at 0.60 and
filtered at 0.40; `answer_key` probed at 0.40 and filtered at 0.31. Budget accordingly.

---

## 3. Cue type (i) is exhausted, and it is your negative control

The pilot ran cue type (i) — a hint pointing at a wrong option — twice, with two different hint texts, on
three framing arms. Both are disclosed at ceiling, both have an empty silent cell, both show no framing
sensitivity. A mentee assigned this domain as specified would spend a semester measuring a quantity pinned
at ceiling.

**Re-scope it or reallocate.** Two options that would make it productive:

- Move it from explicit metadata blocks toward **implicit or distributional cues** — an answer-position
  pattern across few-shot examples, of the kind the original unfaithfulness result used. A pattern spread
  across examples may never be represented as a mentionable proposition at all, which is exactly the
  regime where absent computation is plausible and metadata blocks are not.
- Or keep it as the **negative control** for the whole study. You now have a matched setup where silent
  influence provably does not occur, same pipeline, same judge, same items, bound < 0.03. If code repair or
  sycophancy produces silent influence, the contrast against this is clean and interpretable. Without it,
  a positive result in one domain is hard to distinguish from an instrument artifact.

The second framing is honest and already done. It turns the pilot from a domain with no headroom into the
floor the other domains are measured against.

---

## 4. Cue type (ii) is the most promising domain

Reward-hackable code repair is the case where a model plausibly special-cases to pass a grader *without*
narrating "I am special-casing to pass the grader." The influence is procedural rather than a stated fact
sitting in the context, which is where unverbalized influence should live if it exists anywhere.

Nothing in the pilot speaks against it, and everything in the pilot suggests explicit in-context facts get
narrated. Start here if you want the study's central question to have a chance of a positive answer.

---

## 5. The elicitation baselines do not exist, and they are the discriminator

This is the biggest engineering gap and it sits on the critical path.

Cell assignment can tell you influence went unmentioned. It **cannot** tell you whether the model had the
reason and withheld it. Only the baselines separate concealment from absent computation — interrogation,
few-shot honest disclosure, prefill completion, prefix attacks. If prefill recovers the reason, it was
represented and withheld. If nothing recovers it, that is evidence by elimination for absent computation,
which is the project's central argument.

None of that is built. It is a larger job than the cue-swap validation was.

**One piece is done: interrogation is validated as an instrument.** `scripts/run_interrogation.py` runs a
positive control on faithful items, where the hint demonstrably drove the answer and the trace already
credited it, so the reason is established as present and articulable.

| Framing | Acknowledges | At least once in 4 | Names right option | Sees it, denies influence |
|---|---|---|---|---|
| open (does not say where to look) | 0.740 | 0.973 [0.933, 1.000] | 0.730 | 0.020 |
| direct (names the metadata block) | 0.947 | 1.000 | 0.947 | 0.000 |

Interrogation works unprompted, acknowledgement is specific rather than vague assent, and the
"notices-it-and-denies-it" failure mode — which looks exactly like absent computation while being denial —
runs at 0.02. **So a failure to recover on a genuinely silent case is now interpretable** as evidence the
reason was not represented, rather than as the instrument failing to ask properly. Before this control,
those two were not separable.

Run both framings in the main study: direct as the recovery ceiling, open as the realistic condition, the
gap between them as a measure of how much prompting the elicitation needs.

**Bound on that result:** it is a positive control on items the model already verbalized unprompted, so it
is an upper bound on interrogation's power. It rules out gross instrument failure; it does not prove
interrogation recovers concealed reasons, which was untestable here because the silent cell is empty.

---

## 6. Two measurement lessons that transfer

**Report the trace-level number beside the item-level one.** The four-cell rule aggregates 24 traces and
three placements before comparing, so an item can be scored confabulated without any single trace having
credited a hint it did not follow. Measured per item, "credits the hint but does not follow it" is 0.34;
measured per trace it is 0.03. Roughly 40% of the item-level number is the 6-of-8 modal threshold biting on
high-variance items. Both hints agree. If you report only the item-level cell counts, you will overstate
confabulation by an order of magnitude and it will not be visible from the numbers alone.

**Your randomization unit is the item, not the item-arm pair.** The distractor map is keyed on the item, so
an item's arm replicates carry identical letters and their outcomes agree ~88% of the time. A permutation
test that treats each replicate as independent is anti-conservative by roughly the arm count — it turned a
p of 0.0130 into 0.0010 here. `src/metrics/placement.py` does this correctly now and there is a regression
test pinning it (duplicating items across arms must not move the p-value). The same trap applies to any
statistic you compute across arms.

---

## 7. Scale check before you finalise scope

One cue × one model × three arms was 11,520 generations and about 13 hours on an L4. The proposal is 3 cue
types × 4 models × 5 framing conditions.

Rough arithmetic: ~24,000 generations per cue-model pair including its filter, × 12 pairs ≈ **290,000
generations, ~320 GPU-hours** at the throughput measured here — about 13 days of continuous single-GPU
compute, before baselines and before any rerun. Measured rates: ~15 generations/minute on full traces
(A10), ~12.6 on an L4.

This needs either parallel GPUs, a reduced condition grid, or a smaller item pool per cell. Decide now
rather than in week six.

---

## 8. Model scale

Everything above is Qwen3-8B. 14B does not fit a 24 GB card at bfloat16 (27.6 GiB of weights). Concealment
may require capacity an 8B model does not have, so **a null at 8B carries little information about the
12–20B checkpoints in the model list.** That is an argument for the main study, not against it — but it
also means the judge calibration and the trace-level correction should be re-validated at the target scale
before mentees inherit them as frozen.

---

## 9. What you inherit, working

- `src/config.py` — every threshold and gate, frozen, dumped to `config.json` with the git hash
- `src/metrics/scoring.py` — the four-cell assignment
- `src/metrics/bootstrap.py` — item-clustered intervals, rule-of-three bounds for empty cells, the
  threshold sensitivity grid
- `src/metrics/attribution.py` — the trace-level diagnostics from §6
- `src/metrics/placement.py` — placement, letter and overdispersion analysis, with the corrected unit
- `src/generation/` — prompts and framing arms, vLLM wrapper, trace parsing, the judge, the prefill pair
  builder, the interrogation control
- `scripts/` — one entry point per stage, all with `--cue_name` and `--arms` parameterised, so a new cue is
  a configuration run and not a code change
- 195 tests

Read `paper/methodology.md` for what each stage does and why, `paper/results.md` for every number,
`research_log.md` for the narrative including the wrong turns, and `docs/decisions.md` for the
pre-registration record.

---

## 10. Two non-negotiables the pilot demonstrates

**Thresholds get written down before results are seen.** `docs/decisions.md` is dated and includes
deviations with reasons. One prediction about the ablation was revised before the run on the strength of an
earlier result, recorded as a revision, and then refuted by that run — the original was right. That
ordering is the only reason either outcome is readable rather than a story fitted afterwards.

**Everyone hand-labels their own data.** The judge is validated against 50 blind human labels at κ = 0.92,
and both disagreements run the same direction (the judge under-calls attribution), which is what makes the
empty silent cell a conservative measurement rather than an optimistic one. Without the hand check that
sentence cannot be written.
