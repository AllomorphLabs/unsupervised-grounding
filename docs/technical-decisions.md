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
| 12 | Corpus pairs: episode lengths, the actor mark and hidden ground truth | Decided |
| 13 | Pilot models: a small causal transformer for each stream | Decided |
| 14 | Pilot alignment: on world states, judged by the target stream's dynamics | Decided |
| 15 | A medium language model for the pilot | Decided |
| 16 | Why alignment must use the world's dynamics | Decided |
| 17 | A predictive-state head on the language model | Decided |
| 18 | The frame model predicts whole next frames | Decided |
| 19 | Alignment by decipherment on the two trained models | Decided |

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
- **Hidden ground truth:** everything that reveals what a token or slot means is stored under each pair's `hidden/` directory. Training loaders never open it; only `analysis/labels.py` does, and a test fails if any other module reaches for it.
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
- **Same episode lengths in both streams.** Both streams draw lengths from the same distribution, sampled independently, so a symmetry has one violation rate rather than one per stream. The distribution is set in decision 12.
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

## 12. Corpus pairs

`python -m grounding.data.make configs/pilot.yaml --pairs 0-9` builds pairs 0 to 9 (root seeds 0 to 9) under `runs/pilot/pair-<k>/data/` (`data/corpus.py`).

- **Episode lengths** are uniform from 20 to 60 steps in both streams, drawn independently per stream.
- **Observation frames** are the start state, then the state after each step. Each frame also marks the agent that acted on that step, so every event, including a wait, is visible in both streams, as decision 9 asks.
- **Held-out episodes:** 10% of each stream's episodes, chosen at random, are kept back for scoring.
- **Layout:** the public files (`lang.npz`, `lang_tokens.json`, `obs.npz`) are all training code may read. Sentence labels, true states, the vocabulary and the slot shuffle are under `hidden/`. Each pair also gets a `run.json` run record.
- **Pilot size:** 50,000 episodes per stream (about 15 million tokens and 2 million frames), a starting point to be revised by the pilot's power analysis. With 5,000 episodes, the first trial pair's models memorised their training episodes (see decision 13).

## 13. Pilot models

`python -m grounding.models.train runs/pilot/pair-000 --stream lang` (or `--stream obs`) trains one model of a pair, with settings from `configs/pilot-models.yaml` (`models/`).

- **One architecture for both streams:** a causal transformer with 4 layers, width 128 and 4 heads, about 0.85M parameters, reading one whole episode at a time. The proposal's range starts at a few million parameters; this smaller size is for the pilot only, and the work plan scales up if H1 fails. Decision 15 enlarges the language model.
- **Language model:** predicts the next token.
- **Frame model:** predicts each of the next frame's values as an independent yes/no; decision 18 replaces this with whole frames. It isn't told which values form a block, so the structure of the world isn't handed to it.
- **Budget:** 20,000 steps or 10 minutes per model, whichever comes first. On the workstation GPU that is about 10 minutes for the language model and 5 for the frame model.
- **Best model kept:** 10% of the training episodes are set aside for validation, and the model kept is the one from the step with the lowest validation loss. In the first trial pair, held-out loss rose while training loss kept falling, which means memorisation. The held-out episodes are reserved for scoring: they play no part in training or in choosing the model, and their loss is reported once, at the end.
- **Seeds:** each model's training seed is drawn from its own part of its stream's seed, so every pair and stream trains independently. Runs are exactly reproducible on the CPU; on the GPU they may differ in the last digits.
- **PyTorch builds:** the CUDA 13.4 build on Windows (the workstation's RTX 3070 Ti, 8 GB) and the CPU build on Linux (CI). To revisit when training moves to a Linux GPU machine.

## 14. Pilot alignment

`python -m grounding.align.run runs/pilot/pair-000 --kind model` aligns a pair, and `python -m grounding.analysis.score runs/pilot/pair-000 --kind model` scores it, with settings from `configs/pilot-align.yaml` (`align/`, `analysis/score.py`). `--kind raw` runs the co-occurrence baseline.

- **What is aligned:** each model's picture of the world at each moment. That is the language model's hidden state at every sentence end, and the frame model's hidden state at every frame, from the last layer. Only public structure is used to choose these points. Opening sentences and recaps have no exact counterpart among the frames.
- **Method:** a candidate mapping is judged by how plausible the mapped episodes are to the target stream's trained model (its next-step loss), and the search looks for the mapping with the lowest loss. No labels and no pairs are used. Several restarts are run per pair; the one kept has the lowest judge loss, a score that needs no labels. Every restart is saved and scored. Decision 16 gives the reasons. The search method and the form of the map are being developed on a ladder of controls (frames to frames, then the cipher control, then language to observations) and are frozen before the main conditions.
- **Retired:** the first method, vec2vec-style matching of distributions (after Jha et al., 2025), failed every control. Its code stays until the new aligner replaces it.
- **Superseded for language to frames by decision 19.** A map trained against the judge alone can game it, by inventing or by freezing (decision 16). The judge's insight, that dynamics identify the mapping, carries over into decision 19.
- **Co-occurrence baseline:** the same method on raw statistics, with no neural model. For language, the token counts of the current sentence and the share of each token so far in the episode; for observations, the raw frame. So any gain over the baseline comes from the learned models, not the method. How to judge raw statistics without a trained model (for example a count-based model of the target stream) is Open.
- **Sampling:** alignment uses 5,000 training episodes per side, and scoring 1,000 held-out episodes per side. Both are drawn from the pair's alignment seed and shared by the model alignment and the baseline.
- **Scoring:** each held-out sentence end is mapped across, and the true state of its nearest held-out frame is the prediction. Reported: exact accuracy, accuracy up to symmetry, agreement with each symmetry's relabelling of the truth (which shows whether a pair swapped the agents), accuracy per part of the state, and all of these again for shuffled predictions as chance.
- **Firewall:** scoring refuses any condition not registered in `runs/conditions.py`, and any main condition until `prereg/LOCK` exists.
- **Entity order (decision 11, point 4):** each restart draws its own random starting mapping and its own random search order, so no tie can fall to whichever entity comes first.

## 15. A medium language model for the pilot

From the first pilot run (notes in `runs/pilot-notes.md`, not committed).

The 0.86M-parameter language model predicts well (96.8% of its next-word probability falls on legal actions) but holds the world state only weakly: a two-layer probe on its hidden states, pooled over the last 30 positions, reads places at 0.69 and object locations at 0.59-0.69. A 4.9M-parameter model (6 layers, width 256, 8 heads), trained for an hour with bf16 mixed precision and learning rate 1e-3, predicts no better (validation loss 0.406 against 0.407) but holds the state more explicitly: places 0.76, objects 0.76-0.84. This size is inside the proposal's range.

The language model's settings are in `configs/pilot-lang.yaml`, which training uses by default for the language stream. The frame model keeps decision 13's size: it already holds the state perfectly.

## 16. Why alignment must use the world's dynamics

From the first pilot run; decision 14 follows from it.

- **vec2vec fails even on the easiest control:** raw frames of one pair against raw frames of another, which differ only in slot order. Every restart scored at chance, with the two distributions matched but the wrong mapping found. On real hidden states it was also at chance, while a map fitted with true pairs showed the signal was there.
- **Why:** single moments don't show which places are next to which. In the static distribution, places with similar visit rates are interchangeable, a far larger group than the world's automorphisms. Gromov-Wasserstein matching of single frames shows this directly, mapping places by clean but wrong permutations (bush and cave swapped).
- **What works on the easy control:** judging a candidate mapping by how plausible the mapped episodes are to a trained model of the target stream. The true mapping and its agent-swapped twin score exactly the same (0.1156) and far better than random ones (0.68-0.80); a pairwise-swap search finds the exact mapping, up to the agent swap, in the restart with the best score.
- **The objective is right for language too.** Ideal mappings of language points, checked against the truth, score 0.105 under the judge; every single wrong swap of places or objects scores worse (0.118-0.274), and only the agent swap ties. Opening sentences and recaps don't hurt it.
- **But the judge can be gamed in two ways.** A flexible map can invent plausible trajectories that ignore its input, and static trajectories are very plausible to a next-step model. So the map must be constrained: with linear readouts, a correct map scores 0.153 and one with two objects swapped 0.220, while with small networks both score 0.153.
- **Still to solve:** finding a constrained language-to-observation map without labels. Gradient search, re-initialising parts of the map, matching slow components of the dynamics, and discrete summaries in the frame's format were all tried and stayed at chance, apart from one agent's place in some restarts.

## 17. A predictive-state head on the language model

From the second pilot session. The language model also has to predict the future from a summary: at each sentence end, a 64-number vector must predict the next 48 tokens on its own, through a small two-layer decoder, as well as the usual next token. What comes next depends on where everything is, so the summary is pushed to hold the whole state. It is still trained only on its own stream.

- **Why:** next-token training gave good predictions but a weak picture of the world (decision 15: places 0.76, objects 0.76-0.84). With the head, trained from scratch on pair 0 for an hour, two-layer probes on the summary read agent places, hunger, the stone and the apple at 0.97, and the glass at 0.77 (0.93 from the last layer's hidden state). A map fitted with the truth then gets 68% of frames exactly right, against 5% before.
- **Settings:** `future`, `summary` and `future_layers` in the model config (`configs/pilot-lang.yaml`); `future: 0` turns the head off. Summaries are taken at sentence ends, which are public structure. The loss is the sum of the next-token and future losses, and the best step is chosen by that sum.
- **Glass:** it lags in the summary, probably because a broken glass's place never affects what happens next.

## 18. The frame model predicts whole next frames

From the third pilot session. Alignment must run on two trained models, one per stream, so the
aligner takes its picture of how frames change from the trained frame model. For that it needs
the probability of a whole next frame and valid samples of one. Independent yes/no values can't
give either: they can't say "if agent A acted, then A's place changed, not B's".

- **Head:** a masked network (after MADE, Germain et al., 2015) reads the transformer's context
  and predicts the next frame's values one after another, each given the values before it. The
  product of its yes/no probabilities is the probability of the whole frame. It is still trained
  only on the frame stream, from scratch, with the same per-value loss.
- **Settings:** `joint` and `joint_hidden` in the model config, on in `configs/pilot-models.yaml`.
  `model.log_prob_next` scores a candidate next frame; `model.sample_next` draws one.
- **Still not told the structure:** the value order is the frame's own; the head isn't told
  which values form a block.

## 19. Alignment by decipherment on the two trained models

From the third pilot session (experiment log, entries 21-46). `python -m grounding.align.decipher
runs/pilot/pair-001` aligns a pair, and `python -m grounding.analysis.score runs/pilot/pair-001
--kind decipher` scores it, with settings from `configs/pilot-decipher.yaml` (`align/decipher.py`,
`align/structure.py`).

- **Idea:** the language is treated as a code for a hidden sequence of frames. The fit must explain
  every sentence, so it can't ignore its input or freeze, the two ways a map trained against a
  judge can cheat.
- **Two trained models:** the frame model proposes next frames (its joint head, decision 18), and
  the language model's 64-number summaries (decision 17) weigh them, as a Gaussian linear in the
  frame, softened by a power of 0.2. The words weigh them too, through shared role tables (see
  below). No table of the frame data's transitions is used. Without the summaries, alignment on
  pair 1 falls to chance (entries 43 and 45), so the language model carries it.
- **Each sentence states one fact** (`emission: sentence`, entry 44): a sentence picks one
  value that is on, or one change, and all its words come from that fact's table, the background,
  or the acting agent's place. Explaining each word separately, from any fact, left objects at
  chance on pair 1.
- **Summary inputs** (`summary_features`): the frame's values, or also their pairwise products.
  Chosen by the rule in entry 46. Products fitted to the filter's own early guesses confirmed
  its mistakes when the old word model was in use (entry 43).
- **Inference:** a particle filter, 32 guesses per language episode. Until a guess first moves,
  each episode keeps exact odds over every start frame. EM refits the tables, and from iteration 5
  the Gaussian, to the guesses' paths. The mixture weights are fixed after iteration 8.
- **Structure from each stream alone:** frame values are grouped into entities (agents' places,
  hunger and acted flags; objects' "at", "held", "broken" and "gone" values), and words into
  agents, objects, places and verbs, before the streams are compared (`align/structure.py`).
- **Shared role tables:** each kind of frame feature has one word table written in roles (this
  agent, another place, ...), plus a role for the acting agent's place. A *matching* turns roles
  into words, so a wrong matching can't be patched up by free tables.
- **Choosing the matching:** all matchings are ranked by how far apart the two streams' raw
  frequency profiles are. The top 6 are each fitted, and the fit that best explains 128 held-out
  selection episodes (words plus summaries, including the Gaussian's width term) is kept. Agents
  are fixed in one order, since their swap is a symmetry. **Known limitation for H3:** which
  agent word lands on which frame agent is then decided by arbitrary numbering, not by the
  evidence, so a 50% swap rate across pairs is close to guaranteed by construction. The main
  phase should search both agent orders and test that they tie.
- **The share of sentences that advance the frame** (0.23 stay probability in pair 0) comes from
  each stream's mean episode length: a population statistic, not a pairing.
- **Frozen after pair 0** (75% of frames exactly right, true matching selected). Pair 1 then
  failed on objects (entry 42), and the word model and summary inputs were changed in response
  (entries 43-46), so pair 1 is a tuning pair, not a confirmation. Pair 2 is run once with the
  method frozen again, only if it selects the true matching on pairs 0 and 1.
- **A departure from the proposal.** The proposal names representation-level methods
  (vec2vec-style, cycle-consistent, quadratic assignment), and its working definition maps the
  language model's representations onto the grounded model's. Those methods failed here
  (entries 1-20). Decipherment uses the frame model as a generator of frame histories rather than
  for its hidden representations, and uses the raw words as well as the language model's
  summaries. This must be raised as a change to the proposal before the main phase.
- **No neural model baseline:** `frame_model: counts` replaces the frame model with counted
  transitions from the frame stream (nearest seen frame where unseen), for the co-occurrence
  control.
