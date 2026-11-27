# unsupervised-grounding

Code for Allomorph's first project, **Predicting the Limits of Unsupervised Grounding**.

The full proposal is in [`docs/proposal.md`](docs/proposal.md). Read it before making design decisions. If anything here conflicts with the proposal, the proposal wins; flag the conflict rather than silently picking one.

The same proposal is also published as a Word document on the website, at `public/proposals/predicting-the-limits-of-unsupervised-grounding.docx` in the `allomorphlabs.com` repo. The two must stay in step: if `docs/proposal.md` changes, say so, so the Word version can be updated too.

## The question in one paragraph

A small simulated world produces two independent data streams: descriptions in an invented language, and silent grid "videos". One model is trained on each. The two are then aligned with no paired data, and each mapping is scored against the known ground truth. The central prediction is that the remaining errors follow the world's symmetries (its automorphisms) exactly: interchangeable entities get swapped, nothing else does, and breaking a symmetry removes its confusion and changes nothing else. Near-symmetries are confused in proportion to how nearly they hold. For two models of the same data this limit is now proven (Syrota et al., 2026); our two streams share a world but no data, so the proof doesn't cover them, and the graded near-symmetry prediction is new.

## Hypotheses

| | Claim | Falsified if |
|---|---|---|
| H1 | The language model encodes world state | Probes and the continuation test do no better than the grammar-only and scrambled-world controls |
| H2 | The two models' representations align with no paired data | Symmetry-adjusted accuracy doesn't beat the co-occurrence baseline and the scrambled-world control |
| H3 | Remaining errors follow the world's symmetries, and near-symmetries are confused in proportion to how nearly they hold | Errors above the noise floor fall outside the predicted symmetries and near-symmetries, confusion doesn't rise as violation rate falls, or a confusion persists after its symmetry is broken beyond the registered amount |
| H4 | Alignment degrades gracefully as the two worlds diverge, including when each describes things the other lacks | Accuracy collapses at small divergence, or falls as worlds get richer |

H3 is the main contribution, and H2 must hold for it to be tested. H4's partial-overlap case is the second new test.

## Rules from the proposal

These are design commitments. Don't relax any of them without asking first.

- **No pretrained weights or tokenisers.** Every model is trained from scratch on its own stream.
- **The invented language must not resemble English.** Each word for an action, object or place is generated at random and kept only if it is at least 3 edits from the English words for its meaning, and at least 2 edits from each of the 20,000 most common English words. This check is automated and runs whenever the vocabulary changes.
- **No leakage between streams.** The language and observation streams come from independently generated episodes, with randomised lengths. Nothing about episode boundaries, lengths or timing may line up between them.
- **Renderer checks.** A symmetry in the simulator can be broken by a renderer, for example through different word lengths or frequencies for interchangeable agents, or fixed screen positions. Each symmetry must be verified to survive both renderers before its prediction is registered.
- **Pre-registration comes first.** Predictions, thresholds and tests are fixed and published before the main conditions run. Don't write analysis code that looks at main-condition results before then.
- **The alignment method is frozen first.** The method and its hyperparameters are tuned on the pilot world and worlds derived from it, and checked on a separate calibration world, used for nothing else. They're then frozen before the main conditions run, which use new worlds, not pilot variants.
- **The unit of analysis is an independent pair.** That means one independently sampled corpus pair, one trained model pair and its alignment runs. Restarts within a pair are summarised, never counted as independent samples.
- **Report everything.** That includes full confusion matrices, the objective diagnostic, every control and both observation formats. Null results are reported too.

## Controls

Every main result needs these alongside it:

- **Co-occurrence alignment:** the same alignment on raw word and state statistics, with no neural model.
- **Lookup-table world model:** the frame model replaced by counted transitions, to measure what the learned world model adds.
- **Scrambled world:** the same entities with randomised causal rules.
- **Grammar only:** invented syntax with random words and no world behind it.
- **Symmetry-free world:** sets the noise floor for H3.
- **Cipher of a language stream:** the upper bound on alignment, recovered by classic substitution decipherment; the same-kind-of-data case the 2026 proof covers.
- **Identical worlds:** the easiest point on the H4 overlap sweep.

## Terms

Use these consistently in code and docs:

- **Automorphism / symmetry group:** relabellings of agents, objects and places that commute with the dynamics. Where the dynamics are random, they must also preserve transition probabilities.
- **Near-symmetry:** a relabelling that almost commutes with the dynamics.
- **Violation rate:** how often relabelling and stepping disagree. This is the primary score for near-symmetries.
- **Symmetry cost:** how much less likely the world's own trajectories become when relabelled. The secondary score for near-symmetries, after Xu & Kanan (2026).
- **Noise floor:** the confusion rate measured on the symmetry-free world.
- **Swap rate:** for an exact symmetry, how often the two entities are swapped across independent pairs. The expected value is 50%, tested for equivalence with pre-registered bounds.

## Work plan

The project runs in five gated phases. Don't start a phase until the previous phase's gate is met.

1. **Pilot:** two interchangeable agents among a handful of distinguishable objects and places, with state-vector observations only. *Gate:* alignment beats the co-occurrence baseline, and variance estimates are in hand.
2. **Literature and novelty check.** *Gate:* the design is confirmed as novel, or adjusted.
3. **World, streams and pre-registration.** *Gate:* the pre-registration is published.
4. **Train and align.** *Gate:* the H1 sanity check passes. If it fails, scale up the model or the world rather than stopping.
5. **Sweeps and write-up.**

The compute budget is roughly 2,000–3,000 GPU-hours on a single GPU. Keep tests and examples to tiny configurations that run in seconds, and never launch long training runs without asking.

## Performance and GPU time

A full search takes hours, so time spent making runs faster, and choosing the cheapest run that answers a question, pays for itself many times over. Twenty minutes testing a speed-up can save hours on every run after it.

- **Pick the cheapest test that answers the question.** A model's own measures take minutes, and a fit under the true matching on a tuning pair takes under an hour. Keep full searches for confirming a finished, frozen change on fresh pairs. Stop a running search whose method is already set to change.
- **Measure before optimising.** Time where a run actually goes, from its logs or a short profile at real sizes, before changing code. Synthetic tests can mislead: a real split step fitted 26 times more unit views than a synthetic one of the same data size (experiment log, entry 91).
- **Exact speed-ups first.** A change that gives bit-identical results needs only a check, not new experiments. Verify it against the committed code on the same inputs: on the CPU, then on the GPU at a real search's size, with the old code run twice to see what reproduces itself. A change that differs by rounding is a method change: test it like one, or don't make it.
- **Switch it on.** A verified speed-up behind a setting does nothing until the setting is on. Check the defaults and configs that searches actually use (entry 91 found two exact speed-ups that had never been enabled).
- **Usual suspects:** many tiny GPU operations (launch overhead; CUDA graphs help), waiting for the GPU inside loops (`.item()`, `float()`, `.tolist()`, `if tensor.any()`), recomputing what doesn't change (histories, fixed matrices), and GPU memory filling and spilling into system memory (decision 25).
- **Keep the GPU busy, one job at a time.** Queue work through `runs/queue/`, keep a short job queued behind a long one, watch long runs for stalls and memory spill, and run CPU-side checks with the GPU hidden so they can't disturb a run.

## Technical decisions

Engineering choices (stack, tokenisation, word lists, package layout, how rules are enforced) are recorded in [`docs/technical-decisions.md`](docs/technical-decisions.md), each marked Decided, Proposed or Open. Read it before writing code, and follow Decided entries.

Keep it up to date and concise. When a decision is made, changed or reopened in conversation, edit its entry and the summary table in the same change. Don't add dates or a history section; git records those. Don't treat a Proposed or Open entry as settled; ask first.

In short, so far: Python 3.15, PyTorch and uv; one package, `grounding`; morpheme-level tokens; `wordfreq` for the common-English list; glosses drafted with WordNet into a committed file; tuning on the pilot world is allowed. Every run records its config, random seeds and git commit, and can be rerun from those alone.

## Writing and naming

- UK English in docs and comments.
- For identifiers, files and repo names, prefer short words spelled the same in UK and US English. Avoid pairs like colour/color and optimise/optimize in code names.
- Write docs so an intelligent outsider can follow them: explain the idea before the jargon.
- Never invent results, numbers or citations. Mark unknowns as `[TBD]`.

## Documentation

- `README.md` is the public front page. Keep it short and readable by outsiders, and keep its links to the website and to `docs/proposal.md` working.
- `docs/proposal.md` is the design of record. Don't edit it to match the code; if the code needs to depart from it, raise that first.
- `docs/technical-decisions.md` records engineering choices the proposal leaves open. It must never contradict the proposal; if a decision would, raise it as a conflict.

