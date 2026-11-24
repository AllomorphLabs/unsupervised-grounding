# unsupervised-grounding

**Can an AI that has only ever read words be connected to the world those words describe, and where must that connection fail?**

This repository holds the code for *Predicting the Limits of Unsupervised Grounding*, the first project from [Allomorph](https://allomorphlabs.com), an independent research organisation working on hard, open problems in computer science.

## The idea

We build a small simulated world and describe it in two ways that never meet: sentences in an invented language, and silent grid "videos". One AI learns only from the sentences, and another only from the videos. We then try to match up what each has learned, with no dictionary and no paired examples. Because we built the world, every match can be checked against the truth.

The central prediction is that the mistakes are predictable in advance. Where two things in the world play exactly the same role, like twin characters who do everything together, no amount of pattern-matching can tell them apart. Give one of them a distinctive habit, and the confusion should disappear while nothing else changes. Testing this gives a precise answer to how much meaning a text-only AI can be shown to hold, and a map of what it can never pin down.

There's an interactive demo of the idea on the website: <https://allomorphlabs.com/#project>.

## Status

Pilot in progress. The simulated world, the invented language, the two data streams and the two models are built and tested. On the first pilot pair, aligning the two models with no paired data decodes about three quarters of held-out sentences into exactly the right world state; the method is now being confirmed on fresh pairs. Every attempt so far, including the ones that failed, is recorded in the [experiment log](docs/alignment-attempts.md).

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

## Documents

- **[The full proposal](docs/proposal.md)**: hypotheses, experimental design, controls, analysis plan and work plan.
- The same proposal as a Word document: [allomorphlabs.com/proposals/predicting-the-limits-of-unsupervised-grounding.docx](https://allomorphlabs.com/proposals/predicting-the-limits-of-unsupervised-grounding.docx)
- **[Technical decisions](docs/technical-decisions.md)**: the engineering choices behind the code, such as language, tooling and how the invented language is split into tokens, and the reasons for each.
- **[Experiment log](docs/alignment-attempts.md)**: every alignment attempt in the pilot, in plain language: why it was tried, what was done, what happened and what it means.

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
