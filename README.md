# unsupervised-grounding

**When an AI decodes a language with no dictionary, some mistakes are unavoidable. This project predicts exactly which ones, before the experiment runs, and then checks.**

This repository holds the code for *Predicting the Limits of Unsupervised Grounding*, the first project from [Allomorph](https://allomorphlabs.com), an independent research organisation working on hard, open problems in computer science.

## The idea

We build a small simulated world and describe it in two ways that never meet: sentences in an invented language, and silent grid "videos". One AI learns only from the sentences, and another only from the videos. We then decode the sentences using what the second AI has learned, with no dictionary and no paired examples. Because we built the world, every answer can be checked against the truth.

The central prediction is that the mistakes are predictable in advance. Where two things in the world play exactly the same role, like twin characters who do everything together, no amount of pattern-matching can tell them apart. Give one of them a distinctive habit, and the confusion should disappear while nothing else changes. Where two things are only *almost* alike, they should be mixed up some of the time, and more often the closer they are. Testing this gives a precise answer to how much of what words mean can be recovered from patterns alone, and a map of what never can be.

There's an interactive demo of the idea on the website: <https://allomorphlabs.com/research/unsupervised-grounding/>.

## Status

The pilot is complete. On three independent pairs, aligning the two models with no paired data decodes 59.6% of held-out sentences, on average, into exactly the right world state, against 0.8% for raw word counts. Every attempt, including the ones that failed, is recorded in the [experiment log](docs/alignment-attempts.md).

The first pass of the literature and novelty check is also complete. Two 2026 papers prove and demonstrate the symmetry limit for simpler settings, so the proposal now tests whether it carries over to ours, and adds predictions for near-symmetries and for worlds that only partly overlap.

Phase 3 is under way: grid video, and the pre-registration. Both observation formats now have their structure found from their own statistics, with nothing hand-written. Told which word is which, the method decodes 37–47% of grid-video frames exactly right; whether its search finds that matching with no hints is being tested.

## Running the code

You need [uv](https://docs.astral.sh/uv/), which installs the right Python version (3.15) and every dependency for you:

```
uv sync
uv run pytest
```

To build the pilot's paired data streams (written to `runs/`):

```
uv run python -m grounding.data.make configs/pilot.yaml --pairs 0-9
```

To train a pair's language and observation models (up to an hour and 10 minutes on a GPU):

```
uv run python -m grounding.models.train runs/pilot/pair-000 --stream lang --out runs/pilot/pair-000/models/lang-ps
uv run python -m grounding.models.train runs/pilot/pair-000 --stream obs --out runs/pilot/pair-000/models/obs-joint
```

To align a pair's two models (decipherment, about 80 minutes on a GPU) and score the result against the ground truth:

```
uv run python -m grounding.align.decipher runs/pilot/pair-000
uv run python -m grounding.analysis.score runs/pilot/pair-000 --kind decipher
```

The earlier representation-matching aligner and the co-occurrence baseline are still available through `grounding.align.run` (`--kind model` and `--kind raw`).

Grid video, the proposal's main observation condition, is in progress (phase 3). The same steps work on a grid pair, with the frame structure and the words' kinds found from each stream alone:

```
uv run python -m grounding.data.make configs/pilot-grid.yaml --pairs 0
uv run python -m grounding.models.train runs/pilot-grid/pair-000 --stream obs --config configs/pilot-grid-models.yaml --out runs/pilot-grid/pair-000/models/obs-joint
uv run python -m grounding.align.decipher runs/pilot-grid/pair-000 --config configs/pilot-grid-decipher.yaml
```

## Documents

- **[The full proposal](docs/proposal.md)**: hypotheses, experimental design, controls, analysis plan and work plan.
- The same proposal as a Word document: [allomorphlabs.com/proposals/predicting-the-limits-of-unsupervised-grounding.docx](https://allomorphlabs.com/proposals/predicting-the-limits-of-unsupervised-grounding.docx)
- **[Technical decisions](docs/technical-decisions.md)**: the engineering choices behind the code, such as language, tooling and how the invented language is split into tokens, and the reasons for each.
- **[Experiment log](docs/alignment-attempts.md)**: every alignment attempt in the pilot, in plain language: why it was tried, what was done, what happened and what it means.
- **[Literature review](docs/literature-review.md)**: the novelty check, with every source found, how much of it was read and what it means for the design.

## Principles

- **Pre-registered.** Predictions, thresholds and tests are published before the main experiments run.
- **Open.** The simulator, renderers, datasets, trained models and results will be published here, including null results.
- **From scratch.** No pretrained weights or tokenisers are used, so nothing outside the experiment can leak into it.

## Credits

- The invented language is checked against English using word-frequency data from [wordfreq](https://github.com/rspeer/wordfreq) by Robyn Speer, licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- English glosses are drafted from [Princeton WordNet 3.0](https://wordnet.princeton.edu/) (Princeton University, "About WordNet", 2010), used under the [WordNet licence](https://wordnet.princeton.edu/license-and-commercial-use).

## Links

- Website: <https://allomorphlabs.com>
- Allomorph on GitHub: <https://github.com/allomorphlabs>
