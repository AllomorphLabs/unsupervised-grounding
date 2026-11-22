# Technical decisions

Engineering choices the [proposal](proposal.md) leaves open. Where the two disagree, the proposal wins.

Status is **Decided** (code follows it), **Proposed** (current plan, not yet confirmed) or **Open**. When a decision changes, edit its entry; git keeps the history.

| # | Decision | Status |
|---|---|---|
| 1 | Python 3.15, PyTorch, uv | Decided |
| 2 | Morpheme-level tokens | Decided |
| 3 | `wordfreq` for the common-English word list | Decided |
| 4 | Tuning on the pilot world is allowed | Decided |
| 5 | One installable package, `grounding` | Decided |
| 6 | Proposal rules enforced in code and tests | Decided |
| 7 | English glosses drafted with WordNet; stems and inflected forms both checked | Decided |
| 8 | Exact transition tables, with no built-in asymmetry | Decided |
| 9 | Near-symmetries scored on the states the data visits, with one pre-registered main measure | Decided |
| 10 | Pilot grammar: SOV, case and tense suffixes, every change said | Decided |
| 11 | Nothing about entity order can bias the swap rate | Decided |

## 1. Python 3.15, PyTorch, uv

uv manages the environment, and the Python version is pinned in `.python-version` and `pyproject.toml`. The proposal's starting codebases (Othello-GPT, MUSE, vec2vec) are PyTorch, and torch, numpy and regex all publish Python 3.15 wheels, including for Windows.

- The torch wheels on PyPI for Windows are CPU-only. GPU training needs the CUDA build from PyTorch's package index, added as a uv index when first needed.
- Tests and examples use tiny configurations that run in seconds.

## 2. Morpheme-level tokens

We write the language, so the renderer knows where every stem and suffix begins and ends. It outputs morphemes with boundaries marked, and each morpheme becomes one token: `sorrun-ek` ("go" + past) → `[sorrun] [-ek]`. Nothing is learned, so the rule against pretrained tokenisers is met trivially.

- **Not whole words:** `sorrun-ek` and `sorrun-ai` would become unrelated tokens. That would spread each concept across several tokens, leaving alignment no single embedding per concept.
- **Not characters:** shared letters between words could break a symmetry between interchangeable entities, and sequences would be several times longer.

Tokenising is a separate thin layer, so a character-level robustness check can be added later on the same corpora. The choice must be stated in the pre-registration.

## 3. `wordfreq` for the common-English word list

The 20,000 most common English words come from `wordfreq.top_n_list("en", 20000)`. The package stopped updating in 2024, so the list is stable. Its version is pinned in the lockfile and recorded with each vocabulary. The data is CC-BY-SA 4.0, so the README must credit it once the check is in use.

## 4. Tuning on the pilot world is allowed

The pilot comes before pre-registration, and the freezing rule applies from phase 4. The pilot world must never be reused as a main-condition world or as the calibration world.

## 5. One installable package, `grounding`

```
pyproject.toml
src/grounding/
  sim/          # world state, rules, stepping; world specs; symmetry search
  render/
    lang/       # vocabulary generator, grammar, English-distance check
    state/      # state-vector observations (pilot)
    grid/       # grid "video" observations (phase 3)
  data/         # corpus generation, independent seeding, splits, hidden labels
  models/       # language model, next-state predictor
  align/        # co-occurrence baseline, entity matching, later methods
  analysis/     # scoring, confusion matrices, objective diagnostic, statistics
  runs/         # run records, condition registry
configs/        # world and condition configs
prereg/         # the pre-registration
scripts/        # one-off tools run by hand
tests/          # mirrors src/grounding/, e.g. tests/sim/test_symmetry.py
runs/           # outputs (not committed)
```

## 6. Proposal rules enforced in code and tests

- **Independent streams:** each corpus pair has one root seed. The two streams get separate child seeds via `numpy.random.SeedSequence.spawn`, and each samples its own episode lengths. A test checks that no boundaries or lengths line up.
- **Hidden ground truth:** true states are stored in a separate labels file. Training loaders never open it; only `analysis/` does.
- **Symmetry tests:** small worlds are searched exhaustively against exact transition probabilities. Tests assert each world's expected group, for example {identity, agent swap} for the pilot world and the identity alone for the symmetry-free world.
- **Renderer checks:** for every symmetry, relabelling a trace and then rendering it must give the same statistics as rendering it and then relabelling.
- **Pre-registration firewall:** results of main conditions can't be loaded until `prereg/LOCK` records the commit at which the pre-registration was published. The calibration world is flagged so it can be used only to check the frozen method.
- **Unit of analysis:** outputs are stored as `runs/<condition>/<pair>/<restart>/`, and restarts are summarised within each pair before any statistics.
- **Reproducibility:** every run records its config, seeds, git commit, whether there were uncommitted changes, and package versions.

## 7. English glosses and what the edit-distance check covers

**Glosses.** Each concept, with its chosen WordNet sense and any hand-written extras (irregular forms such as "went", and words WordNet misses), is listed in `configs/concepts.yaml`. `scripts/draft_glosses.py`, run by hand, expands each sense into its single-word synonyms plus regular inflections and writes `configs/glosses.yaml`, which is committed and reviewed as a diff. Fixes go in `concepts.yaml`, never in the generated file. The vocabulary check reads only `glosses.yaml`, so tests, CI and the package never depend on WordNet.

- **Lexicon:** Princeton WordNet 3.0, via the `wn` package (`omw-en:1.4`), a development-only dependency.
- **No named individuals:** the script refuses any sense WordNet marks as a named person or place (for example, "Bush" the president), so only ordinary-word senses can be chosen.
- **Over-generation is harmless:** regular rules produce some non-words ("goed"). An extra gloss can only reject more candidates, which costs almost nothing.

**Scope.** Both stems and full inflected forms (`sorrun` and `sorrunek`) are checked:

- Words for actions, objects and places must be at least 3 edits from their glosses and at least 2 edits from each common English word.
- The proposal's rule doesn't cover grammatical suffixes like `-ek`; any two-letter string is likely within 1 edit of some English word. Agent names have no English gloss but are still checked against the common-word list.

## 8. Exact transition tables, with no built-in asymmetry

The simulator lists every possible (event, next state) outcome of a state with its exact probability, as a fraction. Random stepping only samples from that list, so the symmetry search and the simulator can't drift apart, and exact symmetries are tested by exact equality, not within a tolerance.

- **Events are part of the outcome.** The language stream describes events ("A drops the glass"), not just states, so a relabelling must map events as well as states.
- **Nothing in the mechanics may tell interchangeable entities apart.** Each step, one agent chosen uniformly at random acts; a fixed turn order would make the agents distinguishable. Any future tie-breaks must be random for the same reason.
- **Episode starts must be symmetric too.** As well as commuting with the dynamics, an automorphism must map the set of episode starts onto itself. Otherwise the data wouldn't be symmetric even though the rules are. This adds a condition to the proposal's definition rather than changing it.

## 9. Scoring near-symmetries

A near-symmetry is scored by comparing the real world's transitions with those of the relabelled world, state by state, and averaging over states (`sim/symmetry.py`, `sim/visits.py`).

- **Averaged over the states the data visits.** Each state is weighted by its expected share of all steps in the corpus, computed exactly from the episode-start and episode-length distributions. Counting reachable states equally would make the score depend on how the state space happens to be laid out, not on what the models see. Weights are always passed explicitly, so the choice can't be made by accident.
- **One main measure.** The violation rate is the weighted mean total variation distance: the expected share of steps on which the real and relabelled worlds disagree, matching the proposal's "how often relabelling and stepping disagree". Only this measure is used for the pre-registered rank-correlation test.
- **Reported as extras.** The evidence rate (weighted mean KL divergence: how much each step tells the two labellings apart, infinite when one world can do something the other can't), and the violation rate under equal weighting as a sensitivity check.
- **Same episode lengths in both streams.** Both streams draw lengths from the same distribution, sampled independently, so a symmetry has one violation rate rather than one per stream. The distribution itself is chosen when `data/` is built.
- **Only differences the streams can show.** Both renderers should expose every event, so that violations in the world are violations in the data. Where a renderer hides something, the violation rate is also computed on what that stream reveals, and both are reported.
- **Graded series.** As well as the proposal's rank correlation across all near-symmetries in a world, each symmetry studied is also broken in graded steps (habit weights of 1, 1.1, 1.25, 1.5, 2 and 4, for example) to give a dose-response curve. Habit weights are exact fractions for this reason. This adds to the proposal's test rather than replacing it.

## 10. Pilot grammar

The pilot language is built as the full language will be, so the morpheme tokens and the renderer checks are exercised from the start (`render/lang/`).

- **Word order** is subject-object-verb, unlike English: `A glass-obj take-now`.
- **Suffixes** mark each noun's role (`obj`, `to` a place, `at` a place) and each verb's tense (`now`, `past`). Subjects are unmarked.
- **Every change is said.** An episode opens by saying where every agent and object starts. Each step gives the event, then any side effects: something breaking, an agent becoming hungry. A test rebuilds the full state after every step from the sentences alone.
- **Tense carries meaning.** Steps are narrated in the present. After a step, with a set probability, a past-tense sentence repeats an earlier event from the same episode, so a model tracking the world must learn that past-tense sentences change nothing.
- **No accidental asymmetry.** The opening sentences are shuffled, and every random choice is drawn the same way whatever entities are involved. A test checks that relabelling an episode relabels its sentences and changes nothing else, for any relabelling.
- **Uniform word shapes.** Every stem has the shape consonant-vowel-consonant-vowel-consonant and every suffix vowel-consonant, so interchangeable entities can't differ in the length or shape of their names, even under a later character-level check.

## 11. Nothing about entity order can bias the swap rate

When two mappings fit equally well, as under an exact symmetry, alignment methods break the tie by taking the first best option, which means the lowest index. If agent A always had the lower token number or the first state-vector slot, every pair could resolve the tie the same way, and the swap rate would land near 0% or 100% even with perfect models. So:

1. **A fresh vocabulary per pair**, drawn from the pair's language-stream seed. The English check runs on every draw.
2. **Random token order per pair.** Token numbers come from a seeded shuffle, never from concept order.
3. **Random slot order per pair.** Agent, object and place slots in the state vectors are shuffled with the observation-stream seed.
4. **Random entity order in every alignment restart**, so a tie within a run doesn't fall to whichever entity comes first.
5. **Every shuffle is recorded** with its pair. Analysis maps results back through them before scoring, and every run stays reproducible from its seeds.

A test checks this works: on the tiny world, an aligner that ties perfectly between the two agents gives a swap rate near 50% across seeds, and near 0% or 100% with the shuffles turned off.
