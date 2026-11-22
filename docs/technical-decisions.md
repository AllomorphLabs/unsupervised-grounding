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
tests/
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

**Glosses.** A script in `scripts/`, run by hand, maps each concept to its chosen WordNet sense. It expands that sense into synonyms, adds inflected forms ("went", "rivers") and writes the result to a committed file, `configs/glosses.yaml`, which we then review and edit by hand. The vocabulary check reads only that file, so tests, CI and the package never depend on WordNet. WordNet (via `wn` and Open English WordNet) is a development-only dependency.

**Scope.** Both stems and full inflected forms (`sorrun` and `sorrunek`) are checked:

- Words for actions, objects and places must be at least 3 edits from their glosses and at least 2 edits from each common English word.
- The proposal's rule doesn't cover grammatical suffixes like `-ek`; any two-letter string is likely within 1 edit of some English word. Agent names have no English gloss but are still checked against the common-word list.
