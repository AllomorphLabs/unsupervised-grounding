# unsupervised-grounding

**Can an AI that has only ever read words be connected to the world those words describe, and where must that connection fail?**

This repository holds the code for *Predicting the Limits of Unsupervised Grounding*, the first project from [Allomorph](https://allomorphlabs.com), an independent research organisation working on hard, open problems in computer science.

## The idea

We build a small simulated world and describe it in two ways that never meet: sentences in an invented language, and silent grid "videos". One AI learns only from the sentences, and another only from the videos. We then try to match up what each has learned, with no dictionary and no paired examples. Because we built the world, every match can be checked against the truth.

The central prediction is that the mistakes are predictable in advance. Where two things in the world play exactly the same role, like twin characters who do everything together, no amount of pattern-matching can tell them apart. Give one of them a distinctive habit, and the confusion should disappear while nothing else changes. Testing this gives a precise answer to how much meaning a text-only AI can be shown to hold, and a map of what it can never pin down.

There's an interactive demo of the idea on the website: <https://allomorphlabs.com/#project>.

## Status

Proposal stage. Code will be added here as the pilot begins.

## Documents

- **[The full proposal](docs/proposal.md)**: hypotheses, experimental design, controls, analysis plan and work plan.
- The same proposal as a Word document: [allomorphlabs.com/proposals/predicting-the-limits-of-unsupervised-grounding.docx](https://allomorphlabs.com/proposals/predicting-the-limits-of-unsupervised-grounding.docx)
- **[Technical decisions](docs/technical-decisions.md)**: the engineering choices behind the code, such as language, tooling and how the invented language is split into tokens, and the reasons for each.

## Principles

- **Pre-registered.** Predictions, thresholds and tests are published before the main experiments run.
- **Open.** The simulator, renderers, datasets, trained models and results will be published here, including null results.
- **From scratch.** No pretrained weights or tokenisers are used, so nothing outside the experiment can leak into it.

## Links

- Website: <https://allomorphlabs.com>
- Allomorph on GitHub: <https://github.com/allomorphlabs>
