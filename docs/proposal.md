# Predicting the Limits of Unsupervised Grounding

20 November 2026 · Shane · revised 24 November 2026 after the pilot

*Also published as a Word document on the Allomorph website: [predicting-the-limits-of-unsupervised-grounding.docx](https://allomorphlabs.com/proposals/predicting-the-limits-of-unsupervised-grounding.docx). The two versions have the same content; if you change one, change the other.*

*Revised after the pilot, before any pre-registration. The alignment method has changed from matching representations to decipherment, and one control has been added. The section [Pilot results and revisions](#pilot-results-and-revisions) lists every change and the reasons for it.*

## In plain terms

This project asks whether an AI that has only ever read words can be connected to the world those words describe, and it predicts exactly where that connection must fail.

**The puzzle.** Modern AI language models learn by reading enormous amounts of text and guessing the next word, over and over. They never see, hear or touch anything. Yet they often behave as if they understand what they're talking about. That raises an old question in a new form: can you learn what words *mean* just from how they're arranged? Or is that like learning a foreign language from a library with no pictures, where you could master the grammar but never know that one word means "tree"?

**Why it's hard to settle.** With real language and the real world, we can't check the answer. We never have a complete list of what every word refers to, and we can't control what the world contains. So the debate has mostly been argued with thought experiments.

**Our approach.** We build a small simulated world ourselves: little agents moving between places, picking things up, dropping and breaking them. We describe it in an invented language that shares nothing with English. One AI learns only from those descriptions. A second AI never sees a word: it learns only by watching silent, video-like recordings of the same world. Then, without telling either AI anything about the other, we try to match up their internal pictures of the world. Because we built the world, we know the right answer and can check every match.

**The catch: some things can't be told apart.** Say our world contains two agents who behave in exactly the same way: same habits, same routes, same everything. The text-reading AI knows them only as the words "Zib" and "Mov". The video-watching AI only sees two figures, figure 1 and figure 2. To connect the two AIs, we have to work out which name goes with which figure. The only clue is behaviour. If the text says Zib always goes to the river, look for the figure that always goes to the river. But if Zib and Mov do exactly the same things, that clue is useless. "Zib is figure 1" and "Zib is figure 2" fit the evidence equally well, so the match is a coin toss, however clever the AIs are. That isn't a flaw in either AI. It's a limit on what patterns can ever reveal.

**Why this matters.** Now give Zib one habit Mov doesn't have: he visits the river every day at noon. The text-reading AI picks this up from sentences about Zib, and the video-watching AI sees one figure at the river at noon. Now only one pairing fits, and the match should get it right. That gives us a sharp test. Because we designed the world, we know in advance every pair of things that is truly identical. So we can predict exactly which mix-ups the matching will make, and which it must never make. Then we give one member of each pair a distinguishing habit and check that the mix-up disappears and nothing else changes. If the mistakes appear only where predicted, and vanish exactly when we break the tie, the two AIs have captured the same world. That would show they've recovered the world's structure as fully as patterns alone allow.

**What we hope to show.** If the predictions hold, we'll have a precise, measurable answer to how much meaning a word-only learner can be shown to hold, and a map of what it can never pin down. The method also bears on a practical question: whether we could ever translate communication from another species, such as whales, without a dictionary.

## Abstract

We propose pre-registered predictions of exactly which mapping errors between a form-only language model and a perceiving model of the same world are unavoidable, derived from the world's symmetry group and tested by breaking symmetries one at a time.

We build a small simulated world and produce two independent data streams from it: descriptions in an invented language, and non-linguistic observations rendered as grid "videos". A language model is trained on the first and a predictive world model on the second, from different episodes. We then align the two with no paired data and score every mapping against the simulator's ground truth.

Structure can identify meaning only up to the world's automorphisms. Because we build the world, we compute those automorphisms in advance, predict the residual alignment errors they imply, and test whether errors appear exactly where predicted and vanish when the corresponding symmetry is broken. The result is a controlled, quantitative measure of how much meaning can be recovered from a form-only learner, and of what remains undetermined.

## Background and motivation

Language models trained only to predict the next token show abilities that look like understanding. Why gradient descent on text produces this, and whether it amounts to meaning, are both open questions.

**The world-model hypothesis.** Predicting text well may eventually require modelling the process that generated it. Othello-GPT supports this: trained only on move sequences, with no rules or board, it developed an internal board representation that could be edited to change its predictions (Li et al., 2023).

**The Platonic Representation Hypothesis.** Huh et al. (2024) argue that models trained on different data and modalities converge on a shared statistical model of the world behind the data. Unpaired translation between the embedding spaces of different text models (Jha et al., 2025) and blind matching of vision and language embeddings (Schnaus et al., 2025) support a strong version of this claim.

**The form-versus-meaning objection.** Bender & Koller (2020) argue that a system trained only on linguistic form cannot learn meaning, because meaning is the relation between form and something outside language. Others answer that meaning can arise from conceptual role, the relations between words (Piantadosi & Hill, 2022).

**The structuralist limit.** Even if structure carries meaning, it cannot carry all of it. Newman (1928) observed that pure structure is nearly empty unless its relations are fixed. Where they are fixed, structure still determines its referents only up to isomorphism: any relabelling that preserves every relation fits equally well.

These positions are hard to separate with real data. In natural text and images we never know the true referents, the world's true symmetries, or how much two modalities overlap. A synthetic world gives us all three.

## Research question and claim

**Question.** Does a model trained only on the form of a language acquire representations that can be matched, without supervision, to a perceiving model's representations of the same world? If so, how much meaning is fixed, and what is left undetermined?

**Claim to be tested.** Gradient descent on form alone recovers structure that aligns with perception-derived structure, up to and only up to the world's symmetries.

**Working definition.** A form-only model's representations carry meaning, in the sense tested here, if an unsupervised procedure can use them, together with a grounded model of the world, to map held-out sentences onto the world states that generated them. Any remaining ambiguity should match the world's symmetries. The mapping uses no labels, no probes trained on world facts and no paired examples.

**What this does and does not test.** The language model never sees the world, but the alignment step has access to both models. The experiment therefore tests whether meaning can be *recovered* from a form-only learner by matching it to a grounded learner, not whether text alone constitutes meaning. It is a cross-modal test of the Platonic hypothesis, in a setting where every quantity is known.

## Hypotheses

H3 is the central contribution and H2 is its precondition. H1 checks the premise, and H4 maps the conditions under which the result holds.

| \#  | Hypothesis                                                                                 | Test                                                                                                                                     | Falsified if                                                                                                                              |
|-----|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| H1  | The language model encodes world state                                                     | Linear probes plus intervention tests on hidden activations                                                                              | State probes do no better than on the grammar-only control, or causal-consequence probes do no better than on the scrambled-world control |
| H2  | Language-model representations align with world-model representations, with no paired data | Unsupervised alignment; held-out sentences scored by whether they map to their true world state, both raw and modulo the symmetry group  | Symmetry-adjusted accuracy does not exceed the co-occurrence baseline and the scrambled-world control                                     |
| H3  | Residual alignment errors follow the world's symmetries                                    | Compute the symmetry group; compare predicted and observed confusion patterns across independent corpora; break symmetries one at a time | Model-attributable errors above the noise floor fall outside predicted symmetries, or persist after a symmetry is broken                  |
| H4  | Alignment degrades gracefully as the described and observed worlds diverge                 | Sweep world overlap and world richness                                                                                                   | Accuracy collapses at small divergence, or falls as worlds become richer                                                                  |

H4 also tests a prediction of Goldwasser et al. (2023): given common ground, richer systems should be easier, not harder, to translate without parallel data.

## Meaning up to symmetry

Structure alone can identify referents only up to the world's automorphisms: relabellings of entities that preserve every relation and every causal rule. In a built world this group can be computed exactly, which turns a philosophical limit into a pre-registered prediction.

**Computing the symmetries.** Treat the world as a transition system. An automorphism is a relabelling of agents, objects and places that commutes with the dynamics: relabelling and then stepping gives the same result as stepping and then relabelling. Where dynamics are stochastic, the relabelling must preserve transition probabilities, not just outcomes. Small worlds can be searched exhaustively, and static relations can use graph-automorphism tools such as nauty.

**Near-symmetries.** A relabelling that almost commutes with the dynamics is scored by its violation rate: how often relabelling and stepping disagree. We pre-register a monotonic prediction: the lower a near-symmetry's violation rate, the more often alignment confuses the entities it exchanges. This is tested by rank correlation across all near-symmetries in a world.

**Predictions for exact symmetries.**

- A single alignment outputs one mapping, so an exact symmetry shows up across independent runs, not within one. Two interchangeable agents should be swapped in about half of runs and never partly confused within a run.

- The swap must be free to happen. If the alignment search fixed the order of interchangeable entities and broke ties by their numbering, a 50% swap rate would follow from that numbering, not from the evidence. The search therefore covers every order, and the fit of the two tied mappings is reported for each pair.

- "Independent" means independently sampled corpora and training seeds. A finite corpus never preserves a symmetry perfectly, so one model pair may consistently favour one mapping by chance. Restarting alignment on a single pair measures optimiser behaviour, which is reported separately.

- A mirror-symmetric layout should swap "left" and "right" together, as a pair, along with the objects on each side.

- Above a noise floor, every systematic confusion should correspond to a symmetry or near-symmetry, and every symmetry should appear as confusions. The noise floor is measured on a world built to have no symmetries.

**Interventions.** Break one symmetry at a time: give one agent a distinct habit, or make one side of the map behave differently. The matching confusion should disappear and nothing else should change. How much asymmetry is needed before alignment resolves the ambiguity is a measurement in its own right.

**Renderer checks.** A symmetry in the simulator can be broken by the renderers: different word lengths or frequencies for interchangeable agents, or fixed screen positions in the observation stream. Each symmetry is verified to survive both renderers before its prediction is registered.

## Experimental design

One simulated world produces two independent streams, language and observation, from different episodes. Each trains its own model, and the two models are aligned with no paired data. They share nothing but the world behind their data.

1.  **Build the world.** A small simulator of agents, objects, places, properties and causal rules (dropping glass breaks it; agents eat when hungry). Symmetries are explicit and switchable: interchangeable agents, mirror-symmetric layouts, objects with identical roles.

2.  **Language stream.** A renderer turns events into an invented language with its own vocabulary and grammar (word order, morphology, tense marked by suffixes). Nothing in it resembles English: each word for an action, object or place is generated at random and kept only if it is at least three edits away from the English words for its meaning and at least two edits away from each of the 20,000 most common English words. For example, “zib sorrun mov teshi vudo” means “Zib and Mov go to the river”, and “gorvi pukat” means “eat at the berry bush”.

3.  **Observation stream.** Episodes rendered as small grid "videos" with no words: the main condition. Structured state vectors serve as an easier, more symbolic comparison condition.

4.  **Train.** Small transformers, from a few million to about 100M parameters: a language model on the language stream and a next-frame predictor on the observation stream. At least 10 independently sampled corpora and seeds per condition, with the final number set by a power analysis from the pilot.

5.  **Align, by decipherment.** Read the language stream as a code for a hidden sequence of world states, as statistical decipherment reads an unknown script (Knight et al., 2006; Ravi & Knight, 2011). The world model proposes how the state can change from moment to moment. The words, and the language model's summary of each sentence, weigh each proposed history. Expectation-maximisation learns how each kind of fact is put into words, under a candidate *matching* of words to entities. The matching kept is the one whose fit best explains held-out sentences. No labels or paired examples are used at any point. The candidate matchings cover every relabelling, including the world's symmetries, so a tie between two symmetric matchings is measured rather than broken by how the entities happen to be numbered. Each alignment is restarted several times.

6.  **Score.** For each held-out sentence, check whether the world state it is decoded to is the true state that generated it. Report accuracy both raw and modulo the symmetry group, and record the full confusion pattern.

7.  **Diagnose.** Because the true mapping is known, compute the alignment objective (the fit to held-out sentences) for both the true mapping and the mapping found. If the found mapping scores better than the truth, the models genuinely encode the world differently. If it scores the same, the ambiguity is genuine, as predicted for exact symmetries. If it scores worse, the optimiser failed. Only the first two kinds of error count as evidence about H3.

8.  **Sweep.** Break symmetries one at a time; vary the overlap between the described and observed worlds, world richness, corpus size and model size.

**Starting points.** Othello-GPT's public code for probing and interventions; the statistical decipherment literature (Knight et al., 2006; Snyder, Barzilay & Knight, 2010; Ravi & Knight, 2011) for alignment.

## Controls, baselines and threats to validity

Each control rules out one cheap explanation for a positive result.

| Control or baseline         | What it is                                                                             | What it rules out                                             |
|-----------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Co-occurrence alignment     | The same decipherment on raw word and state statistics, with no neural model           | Success that needs no learned model                           |
| Lookup-table world model    | The world model replaced by counts of the state changes seen in the observation stream | Success that needs no learned model of perception             |
| Scrambled world             | Same entities, randomised causal rules                                                 | Alignment from co-occurrence rather than causal structure     |
| Grammar only                | Invented syntax with random words and no world behind it                               | Alignment driven by syntax or formatting; should be at chance |
| Symmetry-free world         | A world built to have no automorphisms                                                 | Sets the noise floor for H3                                   |
| Cipher of a language stream | Word-for-word substitution of a second corpus in the same language over the same world | Upper bound for alignment under near-perfect isomorphism      |
| Identical worlds            | Both streams from the same world, different episodes                                   | Easiest point of the H4 overlap sweep                         |

**Threats to validity.**

- **Leakage between streams.** Shared episode boundaries, lengths or timing could let alignment succeed trivially. Episodes are generated independently and lengths randomised.

- **Experimenter-supplied grounding.** Both renderers are written by the experimenter. Grid videos reduce this relative to state vectors; results are reported for both, and the gap between them is itself informative.

- **Probes that do the work.** A powerful probe can find structure the model doesn't use. H1 uses linear probes and relies on interventions.

- **Alignment-method overfitting.** The alignment procedure and its hyperparameters are fixed on the cipher control, then frozen before the main conditions are run. Because the cipher pairs two language models, the frozen procedure is also checked on a separate calibration world, used for nothing else, with both language and observation streams.

- **Optimiser failure mistaken for model failure.** Handled by the objective diagnostic in the design.

- **An aligner that does the work.** Decipherment is itself a model of how words express facts, and a strong one could succeed with little help from the trained models. The co-occurrence and lookup-table controls run the identical decipherment with one or both trained models taken out, so they measure what each model contributes.

- **Structure supplied to the aligner.** Before the streams are compared, each one's units are grouped into kinds: words into agents, objects, places and verbs, and state values into entities. This uses each stream on its own, with no pairing and no ground truth. In the pilot, however, the grouping is written for this world's formats, which is experimenter-supplied structure. Replacing it with a general procedure is [TBD]; until then it is reported as a limitation.

- **Renderer-induced asymmetry.** Handled by the renderer checks.

- **Lexical leakage.** If invented words looked like English, a pretrained component or a person inspecting the data could pair words with referents by spelling rather than by structure. No pretrained weights or tokenisers are used, and every word form passes the edit-distance check in the language renderer.

## Analysis plan and success criteria

All predictions, thresholds, tests and analyses are pre-registered before the main conditions are run.

- **Unit of analysis.** One independently sampled corpus pair, one trained model pair and its alignment runs. Alignment restarts within a pair are summarised, not counted as independent.

- **Sample size.** At least 10 independent pairs per condition. The pilot provides variance estimates for a power analysis targeting 80% power at the pre-registered minimum effect size. The exact-symmetry conditions need a larger sample: showing a swap rate within roughly 35–65% of the expected 50% takes about 30–50 independent pairs per condition. These conditions use small, cheap worlds to keep the added compute modest.

- **H2 success.** Symmetry-adjusted held-out accuracy exceeds both the co-occurrence baseline and the scrambled-world control. This is tested with a one-sided permutation test across independent pairs, with a pre-registered minimum effect size and Holm correction across conditions.

- **H3 success.** Model-attributable confusions above the noise floor are predicted by the computed symmetry group. This is measured by agreement between predicted and observed confusion matrices against a permutation null. Each symmetry-breaking intervention removes its predicted confusion and leaves the others unchanged within equivalence bounds.

- **Exact-symmetry check.** For each exact symmetry, the swap rate across independent pairs is tested for equivalence to 50% with two one-sided binomial tests and pre-registered bounds (failing to reject 50% is not evidence for it), and within-run partial confusion is checked to be absent.

- **Near-symmetry check.** Rank correlation between violation rate and confusion rate is tested against the pre-registered negative direction.

- **H4.** Accuracy is reported as a curve against world overlap and world richness, with uncertainty across independent pairs. No single threshold: the shape of the curve is the result.

- **What each model adds.** Accuracy with the lookup-table world model is reported beside every H2 result, as a measure of what the learned world model adds. It is not part of the H2 test.

- **Reporting.** Full confusion matrices, the objective diagnostic, every control and both observation conditions are reported, including null results.

## Expected outcomes and interpretation

Every outcome is informative; only the first supports the claim in full.

| Alignment (H2)                                 | Symmetry match (H3)               | Interpretation                                                                                                                                                                       |
|------------------------------------------------|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Beats all baselines                            | Errors match symmetries           | Form-only structure aligns with perception up to symmetry in closed worlds; strong support for the cross-modal Platonic hypothesis and a measured version of the structuralist limit |
| Beats all baselines                            | Errors don't match                | Models align, but residual ambiguity comes from the models rather than the world; the objective diagnostic shows where the representations diverge                                   |
| Partial: some entity types align, others don't | Errors match within aligned types | Meaning is recovered for some kinds of entity; which kinds, and why, becomes the main finding                                                                                        |
| Beats all baselines, but the lookup-table world model does as well | Tested as above | The language model's structure aligns with the world, but a memorised table of what was seen is as good a partner as a learned world model; a richer world is needed to test the perceiving model's part |
| Matches the co-occurrence baseline only        | Not tested                        | Meaning is recoverable from raw statistics; learned world models add nothing at this scale                                                                                           |
| Fails                                          | Not tested                        | If H1 holds, language models build world models in formats that don't line up with perception; evidence against the Platonic hypothesis                                              |

If H1 itself fails, the language model has not built a world model at this scale, and alignment results should be treated with caution. Alignment that succeeds without a detectable world model would itself be informative, most likely pointing to the co-occurrence explanation. The next step is then a larger model or a richer world, as the work plan sets out.

The H4 curve, showing how fast alignment degrades as the described and observed worlds diverge, is the most practical output. It bears directly on unsupervised translation of animal communication, where overlap with the human world is unknown.

## Relation to prior work

The contribution is the combination of a controlled world, ground-truth scoring and pre-registered symmetry predictions. Each ingredient has precedent; to our knowledge the combination does not.

| Line of work                                                                                                             | What it shows                                                                  | What this project adds                                                                          |
|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| World representations in sequence models (Li et al., 2023; Li, Nye & Andreas, 2021; Jin & Rinard, 2024)                  | Models trained on form encode world state, found with supervised probes        | Recovers the mapping without supervision, by alignment with a perceiving model                  |
| Representational convergence (Huh et al., 2024; Jha et al., 2025)                                                        | Models converge on shared structure; text embeddings translate without pairs   | Tests convergence across language and perception with known ground truth                        |
| Blind vision-language matching (Schnaus et al., 2025)                                                                    | Real vision and language embeddings can often be matched without parallel data | Known referents, known symmetries and controlled overlap; predicts which errors are unavoidable |
| Unsupervised translation (Conneau et al., 2018; Lample et al., 2018; Artetxe et al., 2018; Søgaard, Ruder & Vulić, 2018) | Works when spaces are near-isomorphic; fails when they are not                 | Measures the failure curve directly by sweeping world overlap                                   |
| Theory of unsupervised translation (Goldwasser et al., 2023)                                                             | Translation without parallel data is possible given common ground              | Tests the prediction with trained neural models in a world where the common ground is known     |
| Statistical decipherment (Knight et al., 2006; Snyder, Barzilay & Knight, 2010; Ravi & Knight, 2011)                     | Ciphers, lost scripts and foreign text can be read by fitting a model of the hidden source | The hidden source is a world state proposed by a separately trained perceiving model, and the residual errors are predicted from the world's symmetries |
| Meaning from form (Bender & Koller, 2020; Piantadosi & Hill, 2022; Patel & Pavlick, 2022; Abdou et al., 2021)            | Argument and evidence on whether text-only models acquire meaning              | Measures how much is fixed and what is left undetermined                                        |
| Structural realism (Newman, 1928)                                                                                        | Structure determines its referents only up to isomorphism                      | Computes that isomorphism class in advance and tests it empirically                             |

The closest single comparison is Schnaus et al. (2025). Their result on real data motivates this project, which supplies the controlled setting in which residual errors can be predicted and explained.

## Scope, limitations and open questions

The results apply to closed synthetic worlds at small scale. Extending them to real language and perception is a separate argument, though a clean positive result would shift the burden of proof in the form-versus-meaning debate.

**Limitations.**

- Both data streams are rendered by the experimenter, so the world model's "perception" is itself a designed encoding.

- The alignment step sees both models, so the experiment tests whether meaning can be recovered, not whether a text-only system possesses it unaided.

- Small worlds may lack the richness that makes alignment possible at scale. Null results at small scale are weaker evidence than positive ones.

- In a world small enough to see nearly every situation, a lookup table of observed changes is almost as good a partner for alignment as a learned world model. The pilot showed exactly this, so the perceiving model's contribution can only be tested properly in richer worlds.

**Open questions.**

- How large and rich must a world be before results transfer to real language models?

- Which near-symmetries matter in practice, and is violation rate the right measure of their strength?

- Do larger models resolve near-symmetries that smaller ones confuse, and is that better world modelling or exploitation of renderer artefacts?

- Which observation format best balances perceptual realism against tractability?

## Pilot results and revisions

The pilot (phase 1) is complete, and both parts of its gate are met. The full record, including every failed attempt, is in the project's [experiment log](alignment-attempts.md).

**Setting.** Two interchangeable agents, three objects and four places on a path. Observations are 45 yes/no state values per moment, in shuffled order. Each stream trains its own small transformer from scratch. The method was developed on two corpus pairs. Because it was changed after the second pair was seen, that pair counts as tuning, not as a confirmation. The method was then frozen and run once on a third, untouched pair.

**Results.** The frozen method, which never sees the ground truth, found the true word-to-entity matching on all three pairs, up to the predicted agent swap. Accuracy is the share of held-out sentences decoded to exactly the right world state, with all 45 values correct, up to the world's symmetry (here, the order of the two agents).

| Pair                   | The method | Lookup-table world model | Co-occurrence (no neural model) |
|------------------------|------------|--------------------------|---------------------------------|
| 0 (development)        | 64.7%      | 61.9%                    | 0.8%                            |
| 1 (tuning)             | 44.0%      | 32.9%                    | 1.0%                            |
| 2 (confirmation)       | 70.1%      | 63.0%                    | 0.6%                            |
| Mean                   | 59.6%      | 52.6%                    | 0.8%                            |

Chance is about 0.02%. Without the language model's sentence summaries, alignment falls to chance, so the language model is essential. The learned world model adds 3–11 points over the lookup table on every pair. On the confirmation pair, raw word counts alone ranked a wrong matching first.

**Variance.** Across three restarts of each pair, accuracy varies by a standard deviation of about 2.7 points within a pair, against about 15.8 points between pairs. As the analysis plan assumes, the independent pair is the unit that matters. The power analysis that uses these figures is done in phase 3 ([TBD]).

**What changed in this proposal, and why.**

- **Alignment method.** The original design matched the two models' representations directly: adversarial and cycle-consistent mapping (vec2vec-style) and quadratic-assignment matching. In the pilot these failed even on the easiest control, two observation streams that differ only in the order of their values. The reason is that single moments don't reveal which places are next to which, so many wrong mappings fit the static picture as well as the true one. What identifies the mapping is the dynamics, meaning how states change over time. Decipherment uses the dynamics directly (design step 5).
- **Working definition.** The definition used to say the language model's representations are mapped onto the grounded model's representations. Decipherment instead uses the grounded model as a source of possible world histories, and the language model's summaries and the words as evidence for them. The definition now says this.
- **New control.** The lookup-table world model, added because the pilot showed that a small world can be aligned nearly as well without a learned perceiving model.
- **Agent order.** The pilot's search fixed the order of the two interchangeable agents, so its 50% swap rate would be automatic. The design now requires the search to cover every order (see the predictions for exact symmetries). In the pilot the two orders' fits tied to within 0.01.
- **Threats to validity.** Two added: an aligner that does the work, and structure supplied to the aligner.

## Work plan and resources

The work runs in five gated phases over about eight months, all at single-GPU scale. Durations and compute are initial estimates, to be revised after the pilot.

1.  **Pilot (2 weeks; complete).** Two interchangeable agents among a handful of distinguishable objects and places (so there is something to align beyond the symmetric pair), state-vector observations only. Gate: alignment beats the co-occurrence baseline, and variance estimates are in hand for the power analysis.

2.  **Literature and novelty check (2 weeks).** Detailed review of Goldwasser et al. (2023) and Schnaus et al. (2025), alongside the probing, unsupervised-translation and decipherment literature. Gate: the design is confirmed as novel or adjusted.

3.  **World, streams and pre-registration (6 weeks).** Build the simulator with switchable symmetries, both renderers and all controls. Compute each world's symmetry group, run the renderer checks, and pre-register predictions, thresholds and tests. Gate: pre-registration published.

4.  **Train and align (10 weeks).** Fix and freeze the alignment method on the cipher control, then run the identical-worlds condition, the main conditions and all controls. Gate: the H1 sanity check passes. If it fails, scale up the model or the world rather than stopping.

5.  **Sweeps and write-up (12 weeks).** Symmetry-breaking interventions, the overlap and richness sweeps, analysis, and publication of code, data and results, including nulls.

| Resource            | Estimate                                                                                                                                                                                                                  |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model pairs trained | ~1,500 across conditions, controls and sweeps                                                                                                                                                                             |
| Compute             | ~2,000–3,000 GPU-hours on a single modern GPU, or a small cloud budget; on one GPU this is about 12–18 weeks of continuous running, so it must run near full-time through the train and sweep phases, or use a second GPU |
| People              | One researcher full-time, with part-time input on alignment methods and philosophy of language                                                                                                                            |
| Outputs             | Open-source simulator and renderers, pre-registration, datasets, trained models and a paper                                                                                                                               |

## References

- [Abdou, Kulmizev, Hershcovich, Frank, Pavlick & Søgaard (2021), Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color](https://aclanthology.org/2021.conll-1.9/), CoNLL 2021

- [Artetxe, Labaka, Agirre & Cho (2018), Unsupervised Neural Machine Translation](https://arxiv.org/abs/1710.11041), ICLR 2018

- [Bender & Koller (2020), Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data](https://aclanthology.org/2020.acl-main.463.pdf), ACL 2020

- [Conneau, Lample, Ranzato, Denoyer & Jégou (2018), Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087), ICLR 2018

- [Goldwasser, Gruber, Kalai & Paradise (2023), A Theory of Unsupervised Translation Motivated by Understanding Animal Communication](https://arxiv.org/abs/2211.11081)

- [Huh, Cheung, Wang & Isola (2024), The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)

- [Jha, Zhang, Shmatikov & Morris (2025), Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540)

- [Jin & Rinard (2024), Emergent Representations of Program Semantics in Language Models Trained on Programs](https://proceedings.mlr.press/v235/jin24e.html), ICML 2024

- [Knight, Nair, Rathod & Yamada (2006), Unsupervised Analysis for Decipherment Problems](https://aclanthology.org/P06-2065/), COLING/ACL 2006

- [Lample, Conneau, Denoyer & Ranzato (2018), Unsupervised Machine Translation Using Monolingual Corpora Only](https://arxiv.org/abs/1711.00043), ICLR 2018

- [Li et al. (2023), Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382); code: [othello_world](https://github.com/likenneth/othello_world)

- [Li, Nye & Andreas (2021), Implicit Representations of Meaning in Neural Language Models](https://aclanthology.org/2021.acl-long.143/), ACL 2021

- [Newman, M. H. A. (1928), Mr. Russell's "Causal Theory of Perception"](https://doi.org/10.1093/mind/XXXVII.146.137), *Mind* 37(146), 137–148

- [Patel & Pavlick (2022), Mapping Language Models to Grounded Conceptual Spaces](https://iclr.cc/virtual/2022/poster/5992), ICLR 2022

- [Piantadosi & Hill (2022), Meaning without Reference in Large Language Models](https://arxiv.org/abs/2208.02957)

- [Ravi & Knight (2011), Deciphering Foreign Language](https://aclanthology.org/P11-1002/), ACL 2011

- [Schnaus, Araslanov & Cremers (2025), It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](https://arxiv.org/abs/2503.24129)

- [Snyder, Barzilay & Knight (2010), A Statistical Model for Lost Language Decipherment](https://aclanthology.org/P10-1107/), ACL 2010

- [Søgaard, Ruder & Vulić (2018), On the Limitations of Unsupervised Bilingual Dictionary Induction](https://aclanthology.org/P18-1072/), ACL 2018
