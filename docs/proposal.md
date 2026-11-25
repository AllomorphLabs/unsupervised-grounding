# Predicting the Limits of Unsupervised Grounding

25 November 2026 · Shane · second version

*Also published as a Word document on the Allomorph website: [predicting-the-limits-of-unsupervised-grounding.docx](https://allomorphlabs.com/proposals/predicting-the-limits-of-unsupervised-grounding.docx).*

*This is the second version of the proposal. It replaces the first, of 20 November 2026. It draws on a completed pilot and a literature check. [How this proposal came to be](#how-this-proposal-came-to-be) explains what the first version proposed and what changed.*

## In plain terms

**In one sentence.** When an AI works out what an unknown language means with no dictionary, some mistakes are unavoidable. This project predicts exactly which ones, before the experiment runs, and then checks.

**The set-up.** We build a small simulated world. In the pilot it has two agents walking between four places, picking up a stone, an apple and a glass, eating when hungry, and breaking the glass if they drop it. Two separate AIs learn about this world, and they never share any data:

- a *reader*, which only reads descriptions of the world in a language we invented, whose words look nothing like English; and
- a *watcher*, which only watches silent recordings of the world, with no words at all.

We then decode the reader's language using what the watcher knows: which word means which agent, object and place, and what each sentence says happened. There's no dictionary and no translated examples. Because we built the world, we can check every answer.

**This part already works.** In the pilot, the decoding got about 60% of new sentences exactly right, with every fact correct, against under 1% for a method that only counts words.

**The question now: where must it fail?** The pilot's two agents follow exactly the same rules. The reader knows them as two words, Tofem and Bikop; the watcher sees two figures. Nothing in their behaviour says which word goes with which figure, so the best any method can do is guess. That isn't a weakness of these AIs: no amount of data or cleverness could fix it. Because we built the world, we can list every such tie in advance, which lets us make four sharp predictions:

1. **Exact ties are a coin toss.** Across many independent runs, Tofem should be matched to each figure about half the time, and things with no tie should never be systematically mixed up.
2. **Near-ties are mixed up some of the time, predictably.** Two things that are almost, but not quite, interchangeable should be confused occasionally, and more often the more alike they are. We measure how alike they are from the world's rules, before training any AI.
3. **Breaking a tie removes its mix-up, and nothing else.** Give Tofem a habit Bikop doesn't have, and that confusion should disappear while everything else stays the same.
4. **Partly different worlds degrade gradually.** When the language describes things the recordings never show, or the other way round, decoding should get steadily worse, not collapse.

**What's new.** In 2026, researchers proved that the first prediction holds when two AIs learn from the same kind of data, such as two models of the same photographs. Our reader and watcher learn from different kinds of data, words and recordings, which share a world but nothing else, so the proof doesn't cover them. As far as we know, nobody has tested the second or fourth prediction.

**Why it matters.** It gives a precise answer to an old question: how much of what words mean can be recovered from patterns alone, and what can never be pinned down. It's also a rehearsal for decoding a communication system nobody has a dictionary for, such as whale song. There, knowing in advance which mistakes are unavoidable would tell us which translations to trust.

## Abstract

We test whether the errors of unsupervised grounding can be predicted in advance from the symmetries of the world being described.

We build a small simulated world and produce two independent data streams from it: descriptions in an invented language, and non-linguistic observations rendered as grid "videos". A language model is trained on the first and a predictive world model on the second, from different episodes. We then align the two with no paired data, by decipherment: the language is read as a code for world states proposed by the world model. Every mapping is scored against the simulator's ground truth.

Structure can identify meaning only up to the world's automorphisms, the relabellings that leave every rule unchanged. This limit has recently been proven for two models of the same data (Syrota et al., 2026) and demonstrated on static stimuli (Xu & Kanan, 2026). Our setting lies outside those results: the two models learn from different modalities that share a world but no data, and the symmetries come from the world's dynamics rather than the geometry of a data set. Because we build the world, we compute its symmetries and near-symmetries in advance and pre-register the errors they imply. We test three predictions. Errors appear exactly where the symmetries predict. Near-symmetries are confused in proportion to how nearly they hold. And each confusion vanishes when its symmetry is broken by more than a registered amount. We also measure how alignment degrades when each stream describes things the other lacks, a case unpaired-translation theory leaves open (Goldwasser et al., 2023).

A pilot has shown that the alignment works: on three independent pairs of models, it decodes 59.6% of held-out sentences into exactly the right world state, against 0.8% for the same method without the trained models.

## Background and motivation

Language models trained only to predict the next token show abilities that look like understanding. Why gradient descent on text produces this, and whether it amounts to meaning, are both open questions.

**The world-model hypothesis.** Predicting text well may eventually require modelling the process that generated it. Othello-GPT supports this: trained only on move sequences, with no rules or board, it developed an internal board representation that could be edited to change its predictions (Li et al., 2023).

**The Platonic Representation Hypothesis.** Huh et al. (2024) argue that models trained on different data and modalities converge on a shared statistical model of the world behind the data. Unpaired translation between the embedding spaces of different text models (Jha et al., 2025) and blind matching of vision and language embeddings (Maniparambil et al., 2024; Schnaus et al., 2025, 2026) support a strong version of this claim. The hypothesis is contested, though. Once similarity measures are calibrated for model scale, much of the apparent convergence disappears, and only local neighbourhood structure is shared across modalities (Gröger et al., 2026). Which items relate to which converges more than distances between them do (You et al., 2026). Scoring against a known ground truth, as here, avoids the calibration problem altogether.

**The form-versus-meaning objection.** Bender & Koller (2020) argue that a system trained only on linguistic form cannot learn meaning, because meaning is the relation between form and something outside language. Others answer that meaning can arise from conceptual role, the relations between words (Piantadosi & Hill, 2022).

**The structuralist limit.** Even if structure carries meaning, it cannot carry all of it. Newman (1928) observed that pure structure is nearly empty unless its relations are fixed. Where they are fixed, structure still determines its referents only up to isomorphism: any relabelling that preserves every relation fits equally well.

**Identifiability up to symmetry.** This limit now has a formal version in machine learning. Syrota et al. (2026) prove that aligning two models of the same data without pairs recovers the true correspondence if and only if the data's distribution breaks every symmetry of the data; otherwise the correspondence is fixed only up to those symmetries. Xu & Kanan (2026) show the same limit experimentally, in comparisons between models' responses to the same stimuli. More data and more restarts don't remove it, but stimuli chosen to break the symmetry do. Gromov-Wasserstein matching has the same blind spot (Demetci et al., 2024).

**Why a built world.** These positions are hard to separate with real data. In natural text and images we never know the true referents, the world's true symmetries, or how much two modalities overlap. A synthetic world gives us all three. It also lets us test the formal limit where its proofs don't reach: across two modalities that share a world but no data.

## Research question and claim

**Question.** Does a model trained only on the form of a language acquire representations that can be matched, without supervision, to a perceiving model of the same world? If so, how much meaning is fixed, and what is left undetermined?

**Claim to be tested.** Gradient descent on form alone recovers structure that matches perception-derived structure, up to and only up to the world's symmetries. Where a symmetry nearly holds, alignment confuses the entities it exchanges, and more often the more nearly it holds.

**Working definition.** A form-only model carries meaning, in the sense tested here, if an unsupervised procedure can use it, together with a grounded model of the world, to map held-out sentences onto the world states that generated them. The grounded model proposes possible world histories; the form-only model's representations and the words themselves are the evidence for choosing between them. Any remaining ambiguity should match the world's symmetries. The mapping uses no labels, no probes trained on world facts and no paired examples.

**What this does and does not test.** The language model never sees the world, but the alignment step has access to both models. The experiment therefore tests whether meaning can be *recovered* from a form-only learner by matching it to a grounded learner, not whether text alone constitutes meaning. It is a cross-modal test of the Platonic hypothesis, in a setting where every quantity is known.

## Hypotheses

H3 is the central contribution and H2 is its precondition. H1 checks the premise, and H4 maps the conditions under which the result holds.

| \#  | Hypothesis | Test | Falsified if |
|-----|------------|------|--------------|
| H1  | The language model encodes world state | Linear probes, intervention tests on hidden activations, and a distinguishing-continuation test | State probes do no better than on the grammar-only control, or causal-consequence probes and the continuation test do no better than on the scrambled-world control |
| H2  | The language model can be aligned with the world model, with no paired data | Decipherment; held-out sentences scored by whether they decode to their true world state, both raw and up to the symmetry group | Accuracy up to symmetry does not exceed the co-occurrence baseline and the scrambled-world control |
| H3  | Residual alignment errors follow the world's symmetries, and near-symmetries are confused in proportion to how nearly they hold | Compute the symmetries and near-symmetries; compare predicted and observed confusions across independent pairs; rank near-symmetries by violation rate; break symmetries one at a time, by registered amounts | Model-attributable errors above the noise floor fall outside the predicted symmetries and near-symmetries; or confusion does not rise as violation rate falls; or a confusion persists after its symmetry is broken beyond the registered amount |
| H4  | Alignment degrades gracefully as the described and observed worlds diverge, including when each describes things the other lacks | Sweep world overlap and world richness | Accuracy collapses at small divergence, or falls as worlds become richer |

**What is new.** H3 has two parts. Its exact-symmetry part tests whether a limit proven for two models of the same data carries over to two modalities that share only a world. Its graded part, which predicts how often near-symmetries are confused, has no precedent we know of. H4's partial-overlap sweep is the second new test. Goldwasser et al. (2023) predict that, given common ground, richer systems should be easier to translate without parallel data, not harder. Their theory assumes that one world contains the other, and leaves partial overlap open. Unsupervised translation of languages invented by agents finds that more varied meanings make translation easier (Levy et al., 2025), which supports the "richer is easier" side.

## Meaning up to symmetry

Structure alone can identify referents only up to the world's automorphisms: relabellings of entities that preserve every relation and every causal rule. In a built world these can be computed exactly, which turns a philosophical limit into a pre-registered prediction.

**What is already proven, and what isn't.** For two models of the same data, the limit is a theorem: unpaired alignment is identifiable exactly when the data breaks every symmetry, and otherwise only up to the symmetries (Syrota et al., 2026). Our setting differs in three ways, so the theorem doesn't settle our predictions:

- **Different modalities.** The theorem needs both models to describe one shared data set, even if they see different samples of it. Here one model reads sentences and the other watches grid videos. They share a world but no data, so the theorem's assumptions fail.
- **Symmetries from dynamics.** The theorem's symmetries are static: they preserve distances between data points and the distribution of the data. Ours are relabellings that commute with how the world changes over time.
- **Near-symmetries.** The theorem says nothing about symmetries that nearly hold. Its one real-data example is a near-symmetry, the left-right flip of faces: every failed alignment was a mirror image, but the failure rate isn't predicted or related to how nearly the flip holds. We make that prediction.

**Computing the symmetries.** Treat the world as a transition system. An automorphism is a relabelling of agents, objects and places that commutes with the dynamics: relabelling and then stepping gives the same result as stepping and then relabelling. Where dynamics are random, the relabelling must preserve transition probabilities, not just outcomes. This is the standard notion of an automorphism of a Markov decision process (Ravindran & Barto, 2003). Small worlds can be searched exhaustively, and static relations can use graph-automorphism tools such as nauty.

**Near-symmetries.** A relabelling that almost commutes with the dynamics gets two scores, both computed from the simulator before any model is trained:

- its **violation rate**, how often relabelling and stepping disagree, which counts how often it goes wrong; and
- its **symmetry cost**, how much less likely the world's own trajectories become when they're relabelled, which measures how badly. This follows the "catastrophic cost" of Xu & Kanan (2026).

We pre-register a monotonic prediction: the lower a near-symmetry's violation rate, the more often alignment confuses the entities it exchanges. Violation rate is the primary predictor and symmetry cost the secondary one.

A near-symmetry can be confused in two ways, and both are counted. Within a pair, some sentences decode to the relabelled state while the mapping found is right. Across pairs, the mapping found can be the relabelled one, with every sentence affected. The pilot world shows the second: its path of places is symmetric end to end except for feeding at one place, and an alignment that couldn't read hunger well chose the mirrored places. The confusion rate of a near-symmetry is therefore reported at both levels, the second in the same way as an exact symmetry's swap rate.

**Predictions for exact symmetries.**

- A single alignment outputs one mapping, so an exact symmetry shows up across independent pairs, not within one. Two interchangeable agents should be swapped in about half of pairs, and never partly confused within one.
- The swap must be free to happen. If the alignment search fixed the order of interchangeable entities and broke ties by their numbering, a 50% swap rate would follow from that numbering, not from the evidence. The search therefore covers every order, and the fit of the tied mappings is reported for each pair.
- "Independent" means independently sampled corpora and training seeds. A finite corpus never preserves a symmetry perfectly, so one model pair may consistently favour one mapping by chance. Restarting alignment on a single pair measures optimiser behaviour, which is reported separately.
- A mirror-symmetric layout should swap "left" and "right" together, as a pair, along with the objects on each side.
- Above a noise floor, every systematic confusion should correspond to a symmetry or near-symmetry, and every symmetry should appear as confusions. The noise floor is measured on a world built to have no symmetries.

**Interventions.** Break one symmetry at a time: give one agent a distinct habit, or make one side of the map behave differently. Each intervention registers in advance how strongly it breaks the symmetry, as the violation rate of the relabelling it breaks. A symmetry broken only slightly becomes a near-symmetry, and identifiability results show that breaking a symmetry doesn't guarantee recovery (Syrota et al., 2026, Appendix C). So the prediction is graded. Above a registered violation rate, the matching confusion should disappear and nothing else should change. Below it, the confusion should follow the near-symmetry prediction. How much asymmetry is needed before alignment resolves the ambiguity is a measurement in its own right.

**Renderer checks.** A symmetry in the simulator can be broken by the renderers: different word lengths or frequencies for interchangeable agents, or fixed screen positions in the observation stream. Each symmetry is verified to survive both renderers before its prediction is registered. In grid video, the random choice of colours for each pair is what keeps interchangeable agents interchangeable.

## Experimental design

One simulated world produces two independent streams, language and observation, from different episodes. Each trains its own model, and the two models are aligned with no paired data. They share nothing but the world behind their data.

1. **Build the world.** A small simulator of agents, objects, places, properties and causal rules (dropping glass breaks it; agents eat when hungry). Symmetries are explicit and switchable: interchangeable agents, mirror-symmetric layouts, objects with identical roles.

2. **Language stream.** A renderer turns events into an invented language with its own vocabulary and grammar (word order, word endings, tense). Nothing in it resembles English: each word for an action, object or place is generated at random and kept only if it is at least three edits away from the English words for its meaning and at least two edits away from each of the 20,000 most common English words. Words are split into stems and endings, and the models read these as tokens. Two real sentences from the pilot: "tofem ledol -af rizud -az" (Tofem goes to the bush) and "tofem tevuv -ev kipon -az" (Tofem takes the stone).

3. **Observation stream.** Episodes rendered as small grid "videos" with no words: the main condition. Each place is a patch of the image, each thing is drawn in a random cell of its patch, a carried object shares its carrier's cell, and nothing marks which agent acted. Which colour draws which entity is assigned at random for each pair, so interchangeable agents differ only by an arbitrary colour. Structured state vectors, frames of yes/no facts in a shuffled order with no labels, serve as an easier, more symbolic comparison condition. The two streams come from independently generated episodes with randomised lengths, so nothing about episode boundaries, lengths or timing lines up between them.

4. **Train.** Each model is a small transformer trained from scratch, with no pretrained weights or tokenisers, from a few million to about 100 million parameters. At least 10 independently sampled pairs per condition, with the final number set by a power analysis.
   - The **language model** predicts the next token. At each sentence end it also compresses its state into a short summary, which must predict the next few dozen tokens on its own through a small decoder used only in training. This pushes the summary to hold the state of the world.
   - The **world model** predicts the next frame. It predicts each fact of the frame in turn, given the facts before it, so it gives the probability of a whole frame and can propose realistic ones. Work on world models learned from unlabelled game video (Bruce et al., 2024; Schmidt & Jiang, 2024) informs its design for grid videos.

5. **Align, by decipherment.** Read the language stream as a code for a hidden sequence of world states, as statistical decipherment reads an unknown script (Knight et al., 2006; Ravi & Knight, 2011; Luo, Cao & Barzilay, 2019), and as grounded language learning maps each sentence to a record of the world (Liang, Jordan & Klein, 2009). No labels or paired examples are used at any point, and each alignment is restarted several times.
   - The world model proposes how the state can change from moment to moment.
   - The words, and the language model's summary of each sentence, weigh each proposed history.
   - Expectation-maximisation learns how each kind of fact is put into words, under a candidate *matching* of words to entities. The candidates cover every relabelling, including the world's symmetries.
   - The matching kept is the one whose fit best explains held-out sentences.

6. **Score.** For each held-out sentence, check whether the world state it is decoded to is the true state that generated it, with every fact correct. Report accuracy both raw and up to the symmetry group, and record the full confusion pattern.

7. **Diagnose.** Because the true mapping is known, compute the alignment objective (the fit to held-out sentences) for both the true mapping and the mapping found, each fitted by the same procedure with the same restarts, so that a difference reflects the mappings rather than where one fit happened to settle. If the found mapping scores better than the truth, the models genuinely encode the world differently. If it scores the same, the ambiguity is genuine, as predicted for exact symmetries. If it scores worse, the optimiser failed. Only the first two kinds of error count as evidence about H3.

8. **Sweep.** Break symmetries one at a time, by registered amounts. Vary the overlap between the described and observed worlds, including worlds where each stream has entities or events the other lacks. Vary world richness, corpus size and model size.

**Freezing the method.** The alignment method and its hyperparameters are tuned on the cipher control, then checked on a separate calibration world, used for nothing else, with both language and observation streams. They are then frozen before any main condition runs.

**Starting points.** The pilot's code, which already implements the simulator, the invented language, state-vector frames, both models and decipherment; Othello-GPT's public code for probing and interventions.

## Controls

Each control rules out one cheap explanation for a positive result.

| Control or baseline | What it is | What it rules out |
|---------------------|------------|-------------------|
| Co-occurrence alignment | The same decipherment on raw word and state statistics, with no neural model | Success that needs no learned model |
| Lookup-table world model | The world model replaced by counts of the state changes seen in the observation stream | Success that needs no learned model of perception |
| Scrambled world | Same entities, randomised causal rules | Alignment from co-occurrence rather than causal structure |
| Grammar only | Invented syntax with random words and no world behind it | Alignment driven by syntax or formatting; should be at chance |
| Symmetry-free world | A world built to have no automorphisms | Sets the noise floor for H3 |
| Cipher of a language stream | Word-for-word substitution of a second corpus in the same language over the same world | Upper bound for alignment under near-perfect isomorphism |
| Identical worlds | Both streams from the same world, different episodes | Easiest point of the H4 overlap sweep |

## Threats to validity

- **Leakage between streams.** Shared episode boundaries, lengths or timing could let alignment succeed trivially. Episodes are generated independently and lengths randomised.
- **Experimenter-supplied grounding.** Both renderers are written by the experimenter. Grid videos reduce this relative to state vectors; results are reported for both, and the gap between them is itself informative.
- **Probes that mislead.** A powerful probe can find structure the model doesn't use. Probes can also mislead the other way: a model can pass probe tests while holding an incoherent world model (Vafa et al., 2024), and a linear probe can miss a world model stored in a different frame of reference (Nanda, Lee & Wattenberg, 2023). The pilot met the second case. H1 therefore combines linear probes, interventions, and a distinguishing-continuation test after Vafa et al. (2024): world states that differ should lead the language model to different continuations, and states that are the same should not. Because every state is known, this test is cheap here.
- **An aligner that does the work.** Decipherment is itself a model of how words express facts, and a strong one could succeed with little help from the trained models. The co-occurrence and lookup-table controls run the identical decipherment with one or both trained models taken out, so they measure what each model contributes.
- **Structure supplied to the aligner.** Before the streams are compared, each one's units are grouped into kinds: words into classes that name things and places, and frame values into things, places and appearances. In the pilot this grouping was written for that world's formats. It is now found by general procedures from each stream's own statistics, for both observation formats, with no pairing and no ground truth, and is checked against the hidden key only after the fact. The procedures still assume what kinds of thing a world contains (things at places, that can look different, be carried or disappear) and that each sentence states one fact; these assumptions are reported with the results.
- **Impossible candidate frames.** A world model can propose frames that couldn't occur, such as a thing drawn twice. The aligner gives no weight to candidates that break the structure found in the observation stream itself, and the share of valid proposals is reported for every world model.
- **Alignment-method overfitting.** Handled by freezing the method on the cipher control and checking it on the calibration world.
- **Optimiser failure mistaken for model failure.** Handled by the objective diagnostic.
- **Renderer-induced asymmetry.** Handled by the renderer checks.
- **Lexical leakage.** If invented words looked like English, a pretrained component or a person inspecting the data could pair words with referents by spelling rather than by structure. No pretrained weights or tokenisers are used, and every word form passes the edit-distance check.

## Analysis plan and success criteria

All predictions, thresholds, tests and analyses are pre-registered before the main conditions are run, following the pre-registration trials at NeurIPS (2020 and 2021).

- **Unit of analysis.** One independently sampled corpus pair, one trained model pair and its alignment runs. Alignment restarts within a pair are summarised, not counted as independent.
- **Sample size.** At least 10 independent pairs per condition, with the final number set by a power analysis at 80% power for the pre-registered minimum effect size. The pilot found that results vary far more between pairs than between restarts within a pair (see [The pilot](#the-pilot)), which this analysis will use. The exact-symmetry conditions need a larger sample: showing a swap rate within roughly 35–65% of the expected 50% takes about 30–50 independent pairs per condition. These conditions use small, cheap worlds to keep the added compute modest.
- **H1.** State probes are compared with the grammar-only control, and causal-consequence probes with the scrambled-world control. The continuation test measures how often the language model's predicted continuations tell apart world states that differ, and fail to tell apart states that are the same, against both controls.
- **H2.** Accuracy up to symmetry on held-out sentences exceeds both the co-occurrence baseline and the scrambled-world control. This is tested with a one-sided permutation test across independent pairs, with a pre-registered minimum effect size and Holm correction across conditions.
- **H3, exact symmetries.** For each exact symmetry, the swap rate across independent pairs is tested for equivalence to 50% with two one-sided binomial tests and pre-registered bounds (failing to reject 50% is not evidence for it), and partial confusion within a pair is checked to be absent.
- **H3, near-symmetries.** The rank correlation between violation rate and confusion rate is tested against the pre-registered negative direction, with confusion measured both within pairs and as the share of pairs whose mapping found is the relabelled one. The same test is run for symmetry cost, and the two predictors are compared.
- **H3, overall.** Model-attributable confusions above the noise floor are predicted by the computed symmetries and near-symmetries. This is measured by agreement between predicted and observed confusion matrices against a permutation null.
- **H3, interventions.** Each intervention above its registered violation rate removes its predicted confusion and leaves the others unchanged within equivalence bounds. Below that rate, its confusion is tested as a near-symmetry.
- **H4.** Accuracy is reported as a curve against world overlap and world richness, with uncertainty across independent pairs. There is no single threshold: the shape of the curve is the result.
- **What each model adds.** Accuracy with the lookup-table world model is reported beside every H2 result, as a measure of what the learned world model adds. It is not part of the H2 test.
- **Reporting.** Full confusion matrices, the objective diagnostic, every control and both observation formats are reported, including null results.

## Expected outcomes and interpretation

Every outcome is informative; only the first supports the claim in full.

| Alignment (H2) | Symmetry match (H3) | Interpretation |
|----------------|---------------------|----------------|
| Beats all baselines | Errors match symmetries and near-symmetries | Form-only structure aligns with perception up to symmetry in closed worlds. The proven identifiability limit carries over across modalities and to dynamics, and extends to near-symmetries; strong support for the cross-modal Platonic hypothesis and a measured version of the structuralist limit |
| Beats all baselines | Errors match exact symmetries, but near-symmetry confusion doesn't track violation rate | The limit carries over for exact symmetries, but how nearly a symmetry holds doesn't predict its confusion; symmetry cost, or properties of the models, may explain it instead |
| Beats all baselines | Errors don't match | Residual ambiguity comes from the models rather than the world, so the proven limit doesn't carry over to this setting. The objective diagnostic shows where the representations diverge, and the theorem's assumptions show where to look |
| Partial: some entity types align, others don't | Errors match within aligned types | Meaning is recovered for some kinds of entity; which kinds, and why, becomes the main finding |
| Beats all baselines, but the lookup-table world model does as well | Tested as above | The language model's structure aligns with the world, but a memorised table of what was seen is as good a partner as a learned world model; a richer world is needed to test the perceiving model's part |
| Matches the co-occurrence baseline only | Not tested | Meaning is recoverable from raw statistics; learned models add nothing at this scale |
| Fails | Not tested | If H1 holds, language models build world models in formats that don't line up with perception; evidence against the Platonic hypothesis |

If H1 itself fails, the language model has not built a world model at this scale, and alignment results should be treated with caution. Alignment that succeeds without a detectable world model would itself be informative, most likely pointing to the co-occurrence explanation. The next step is then a larger model or a richer world, as the work plan sets out.

The H4 curve, showing how fast alignment degrades as the described and observed worlds diverge, is the most practical output. It bears directly on unsupervised translation of animal communication, where overlap with the human world is unknown.

## The pilot

The pilot is complete, and it met its gate: alignment beats the co-occurrence baseline, and variance estimates are in hand. The full record, including every failed attempt, is in the project's [experiment log](alignment-attempts.md).

**Setting.** Two interchangeable agents, three objects and four places on a path, with state-vector observations: 45 yes/no facts per moment, in a shuffled order. Each stream trains its own small transformer from scratch. The language model has about 6.5 million parameters, a quarter of them in the decoder used only in training; the world model has about 1 million. The method was developed on two corpus pairs. Because it was changed after the second pair was seen, that pair counts as tuning, not as a confirmation. The method was then frozen and run once on a third, untouched pair.

**Results.** The frozen method, which never sees the ground truth, found the true word-to-entity matching on all three pairs, up to the predicted swap of the two agents. Accuracy is the share of held-out sentences decoded to exactly the right world state, with all 45 facts correct, up to the world's symmetry (here, the order of the two agents).

| Pair | The method | Lookup-table world model | Co-occurrence (no neural model) |
|------|------------|--------------------------|---------------------------------|
| 0 (development) | 64.7% | 61.9% | 0.8% |
| 1 (tuning) | 44.0% | 32.9% | 1.0% |
| 2 (confirmation) | 70.1% | 63.0% | 0.6% |
| Mean | 59.6% | 52.6% | 0.8% |

Chance is about 0.02%. Without the language model's sentence summaries, alignment falls to chance, so the language model is essential. The learned world model adds 3–11 points over the lookup table on every pair. On the confirmation pair, raw word counts alone ranked a wrong matching first.

**Variance.** Across three restarts of each pair, accuracy varies by a standard deviation of about 2.7 points within a pair, against about 15.8 points between pairs. The independent pair, not the restart, is the unit that matters. The power analysis that uses these figures is done in phase 3 ([TBD]).

**What the pilot taught us.**

- **Matching representations directly fails here.** Lining up the two models' internal representations so that their overall shapes match stayed at chance, even on the easiest control. Single moments don't reveal which places are next to which, so many wrong mappings fit the static picture as well as the true one. What identifies the mapping is the dynamics, how states change over time, and decipherment uses them directly.
- **A small world needs little perception.** In a world small enough to see nearly every situation, a lookup table of observed changes is almost as good a partner as a learned world model. The world model's contribution can only be tested properly in richer worlds.
- **Ties must be free to fall either way.** The pilot's search fixed the order of the two agents, so its swap rate can't count as evidence. The two orders' fits tied to within 0.01, as predicted, but the main study searches every order.
- **Probes can under-report.** In one pair, the language model stored object positions in a form a linear probe couldn't read, although the information was there. This is why H1 adds the continuation test.

## How this proposal came to be

The [first version of this proposal](proposal-v1.md), dated 20 November 2026, is kept as a record. It asked the same question and made the same core prediction, that alignment errors follow the world's symmetries. Two things have changed it since.

**The pilot changed the method.** The first version aligned the two models by matching their representations directly: adversarial and cycle-consistent mapping in the style of vec2vec, and quadratic-assignment matching (Schnaus et al., 2025). In the pilot these failed even on the easiest control, for the reason given above. This version aligns by decipherment instead. With it:

- the working definition now says that the grounded model proposes world histories and the language model weighs them, rather than that one model's representations are mapped onto the other's;
- the lookup-table world model was added as a control;
- the search must cover every order of interchangeable entities;
- two threats to validity were added: an aligner that does the work, and structure supplied to the aligner.

**The literature check changed the framing.** The first version presented "meaning up to symmetry" as the project's own idea. In 2026 it was proven for two models of the same data (Syrota et al., 2026) and demonstrated on static stimuli (Xu & Kanan, 2026). The full check, covering 33 sources, is in the project's [literature review](literature-review.md). In response:

- H3 now tests whether that proven limit carries over to a setting outside its assumptions, and adds a graded prediction for near-symmetries;
- each near-symmetry gets a second score, its symmetry cost;
- each symmetry-breaking intervention registers how strongly it breaks its symmetry;
- H1 adds a distinguishing-continuation test;
- H4 explicitly includes worlds where each stream describes things the other lacks.

The core design is unchanged: one built world, two independent streams, two models trained from scratch, alignment with no paired data, ground-truth scoring and pre-registered predictions.

## Relation to prior work

The contribution is the combination of a controlled world, ground-truth scoring and pre-registered symmetry predictions, together with two tests that have no precedent we know of: the graded near-symmetry prediction and a direct test of partial overlap. Each ingredient has precedent, and the principle behind H3 has recently been proven for alignment within one modality.

| Line of work | What it shows | What this project adds |
|--------------|---------------|------------------------|
| Identifiability of unpaired alignment (Syrota et al., 2026; Xu & Kanan, 2026; Demetci et al., 2024) | Unpaired alignment is fixed only up to the data's symmetries: proven for two models of the same data, and shown on static stimuli | Tests the limit across two modalities that share a world but no data, with symmetries defined by dynamics; adds graded predictions for near-symmetries, registered in advance, and swap rates across independent pairs |
| World representations in sequence models (Li et al., 2023; Li, Nye & Andreas, 2021; Jin & Rinard, 2024; Nanda, Lee & Wattenberg, 2023; Vafa et al., 2024) | Models trained on form encode world state, found with supervised probes; probes can over- or under-report it | Recovers the mapping without supervision, by alignment with a perceiving model, and checks H1 with a continuation test |
| Representational convergence (Huh et al., 2024; Jha et al., 2025; Gröger et al., 2026; You et al., 2026) | Models converge on shared structure, though after calibration mainly on local, relational structure | Tests convergence across language and perception with known ground truth, which needs no calibration |
| Blind vision-language matching (Maniparambil et al., 2024; Schnaus et al., 2025, 2026) | Real vision and language embeddings can often be matched with few or no pairs, and when matching works can be predicted from geometry | Known referents, known symmetries and controlled overlap; predicts which errors are unavoidable |
| Unsupervised translation (Conneau et al., 2018; Lample et al., 2018; Artetxe et al., 2018; Søgaard, Ruder & Vulić, 2018) | Works when spaces are near-isomorphic; fails when they are not | Measures the failure curve directly by sweeping world overlap |
| Theory of unsupervised translation (Goldwasser et al., 2023; Levy et al., 2025) | Translation without parallel data is possible given common ground, and easier for richer languages; partial overlap is left open | Tests the prediction with trained neural models in a world where the common ground is known, including the partial-overlap case |
| Statistical decipherment and grounded language learning (Knight et al., 2006; Snyder, Barzilay & Knight, 2010; Ravi & Knight, 2011; Luo, Cao & Barzilay, 2019; Liang, Jordan & Klein, 2009) | Ciphers, lost scripts and foreign text can be read by fitting a model of the hidden source; sentences can be matched to records of the world they describe | The hidden source is a world state proposed by a separately trained perceiving model, and the residual errors are predicted from the world's symmetries |
| Meaning from form (Bender & Koller, 2020; Piantadosi & Hill, 2022; Patel & Pavlick, 2022; Abdou et al., 2021) | Argument and evidence on whether text-only models acquire meaning | Measures how much is fixed and what is left undetermined |
| Structural realism (Newman, 1928) | Structure determines its referents only up to isomorphism | Computes that isomorphism class in advance and tests it empirically |

The closest comparisons are Syrota et al. (2026) and Xu & Kanan (2026) for the principle, and Schnaus et al. (2025, 2026) for the setting. Their results on real data motivate this project, which supplies the controlled setting in which residual errors can be predicted and explained.

## Scope, limitations and open questions

The results apply to closed synthetic worlds at small scale. Extending them to real language and perception is a separate argument, though a clean positive result would shift the burden of proof in the form-versus-meaning debate.

**Limitations.**

- Both data streams are rendered by the experimenter, so the world model's "perception" is itself a designed encoding.
- The alignment step sees both models, so the experiment tests whether meaning can be recovered, not whether a text-only system possesses it unaided.
- Small worlds may lack the richness that makes alignment possible at scale. Null results at small scale are weaker evidence than positive ones.
- The procedures that group each stream's units into kinds assume the general shape of the world and the language (see threats to validity).

**Open questions.**

- How large and rich must a world be before results transfer to real language models?
- Which near-symmetries matter in practice, and is violation rate or symmetry cost the better measure of their strength?
- Do larger models resolve near-symmetries that smaller ones confuse, and is that better world modelling or exploitation of renderer artefacts?
- Which observation format best balances perceptual realism against tractability?
- Does the same limit apply on the observation side alone? A world model that discovers its own actions from video, as Genie does (Bruce et al., 2024), could be tested for whether it confuses symmetric actions in the same way.

## Work plan and resources

The work runs in five gated phases over about eight months, all at single-GPU scale. No phase starts until the previous phase's gate is met.

1. **Pilot (2 weeks; complete).** Two interchangeable agents among a handful of distinguishable objects and places, state-vector observations only. Gate: alignment beats the co-occurrence baseline, and variance estimates are in hand for the power analysis. Met.
2. **Literature and novelty check (2 weeks; complete).** Gate: the design is confirmed as novel, or adjusted. Met: the design was adjusted, as this version describes. A few remaining full-text reads, listed in the literature review, continue alongside phase 3.
3. **World, streams and pre-registration (6 weeks).** Build the full simulator with switchable symmetries, the grid-video renderer and all controls. Compute each world's symmetries, near-symmetries, violation rates and symmetry costs, set the strength of each symmetry-breaking intervention, run the renderer checks and the power analysis, and pre-register predictions, thresholds and tests. Gate: pre-registration published.
4. **Train and align (10 weeks).** Freeze the alignment method on the cipher control and the calibration world, then run the identical-worlds condition, the main conditions and all controls. Gate: the H1 sanity check passes. If it fails, scale up the model or the world rather than stopping.
5. **Sweeps and write-up (12 weeks).** Symmetry-breaking interventions, the overlap and richness sweeps, analysis, and publication of code, data and results, including nulls.

| Resource | Estimate |
|----------|----------|
| Model pairs trained | ~1,500 across conditions, controls and sweeps |
| Compute | ~2,000–3,000 GPU-hours on a single modern GPU, or a small cloud budget; on one GPU this is about 12–18 weeks of continuous running, so it must run near full-time through the train and sweep phases, or use a second GPU |
| People | One researcher full-time, with part-time input on alignment methods and philosophy of language |
| Outputs | Open-source simulator and renderers, pre-registration, datasets, trained models and a paper |

## References

- [Abdou, Kulmizev, Hershcovich, Frank, Pavlick & Søgaard (2021), Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color](https://aclanthology.org/2021.conll-1.9/), CoNLL 2021

- [Artetxe, Labaka, Agirre & Cho (2018), Unsupervised Neural Machine Translation](https://arxiv.org/abs/1710.11041), ICLR 2018

- [Bender & Koller (2020), Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data](https://aclanthology.org/2020.acl-main.463.pdf), ACL 2020

- [Bruce et al. (2024), Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391), ICML 2024

- [Conneau, Lample, Ranzato, Denoyer & Jégou (2018), Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087), ICLR 2018

- [Demetci, Tran, Redko & Singh (2024), Breaking Isometric Ties and Introducing Priors in Gromov-Wasserstein Distances](https://proceedings.mlr.press/v238/demetci24a.html), AISTATS 2024

- [Goldwasser, Gruber, Kalai & Paradise (2023), A Theory of Unsupervised Translation Motivated by Understanding Animal Communication](https://arxiv.org/abs/2211.11081)

- [Gröger, Wen & Brbić (2026), Revisiting the Platonic Representation Hypothesis: An Aristotelian View](https://arxiv.org/abs/2602.14486), ICML 2026

- [Huh, Cheung, Wang & Isola (2024), The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)

- [Jha, Zhang, Shmatikov & Morris (2025), Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540)

- [Jin & Rinard (2024), Emergent Representations of Program Semantics in Language Models Trained on Programs](https://proceedings.mlr.press/v235/jin24e.html), ICML 2024

- [Knight, Nair, Rathod & Yamada (2006), Unsupervised Analysis for Decipherment Problems](https://aclanthology.org/P06-2065/), COLING/ACL 2006

- [Lample, Conneau, Denoyer & Ranzato (2018), Unsupervised Machine Translation Using Monolingual Corpora Only](https://arxiv.org/abs/1711.00043), ICLR 2018

- [Levy, Paradise, Carmeli, Meir, Goldwasser & Belinkov (2025), Unsupervised Translation of Emergent Communication](https://arxiv.org/abs/2502.07552), AAAI 2025

- [Li et al. (2023), Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382); code: [othello_world](https://github.com/likenneth/othello_world)

- [Li, Nye & Andreas (2021), Implicit Representations of Meaning in Neural Language Models](https://aclanthology.org/2021.acl-long.143/), ACL 2021

- [Liang, Jordan & Klein (2009), Learning Semantic Correspondences with Less Supervision](https://aclanthology.org/P09-1011/), ACL 2009

- [Luo, Cao & Barzilay (2019), Neural Decipherment via Minimum-Cost Flow: from Ugaritic to Linear B](https://aclanthology.org/P19-1303/), ACL 2019

- [Maniparambil et al. (2024), Do Vision and Language Encoders Represent the World Similarly?](https://arxiv.org/abs/2401.05224), CVPR 2024

- [Nanda, Lee & Wattenberg (2023), Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941), BlackboxNLP 2023

- NeurIPS pre-registration workshops: [2020](https://neurips.cc/virtual/2020/workshop/16158) and [2021](https://neurips.cc/virtual/2021/workshop/21885)

- [Newman, M. H. A. (1928), Mr. Russell's "Causal Theory of Perception"](https://doi.org/10.1093/mind/XXXVII.146.137), *Mind* 37(146), 137–148

- [Patel & Pavlick (2022), Mapping Language Models to Grounded Conceptual Spaces](https://iclr.cc/virtual/2022/poster/5992), ICLR 2022

- [Piantadosi & Hill (2022), Meaning without Reference in Large Language Models](https://arxiv.org/abs/2208.02957)

- [Ravi & Knight (2011), Deciphering Foreign Language](https://aclanthology.org/P11-1002/), ACL 2011

- [Ravindran & Barto (2003), SMDP Homomorphisms: An Algebraic Approach to Abstraction in Semi-Markov Decision Processes](https://ijcai.org/proceedings/2003/144), IJCAI 2003

- [Schmidt & Jiang (2024), Learning to Act without Actions](https://arxiv.org/abs/2312.10812), ICLR 2024

- [Schnaus, Araslanov & Cremers (2025), It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](https://arxiv.org/abs/2503.24129)

- [Schnaus, Dagès, Cremers, Wang & Isola (2026), Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data](https://arxiv.org/abs/2610.09411), preprint

- [Snyder, Barzilay & Knight (2010), A Statistical Model for Lost Language Decipherment](https://aclanthology.org/P10-1107/), ACL 2010

- [Søgaard, Ruder & Vulić (2018), On the Limitations of Unsupervised Bilingual Dictionary Induction](https://aclanthology.org/P18-1072/), ACL 2018

- [Syrota, Lopez i Serrano, Franck & Hauberg (2026), Identifiable Alignment of Unpaired Representations](https://arxiv.org/abs/2611.11460), preprint

- [Vafa, Chen, Rambachan, Kleinberg & Mullainathan (2024), Evaluating the World Model Implicit in a Generative Model](https://arxiv.org/abs/2406.03689), NeurIPS 2024

- [Xu & Kanan (2026), More Data Cannot Break a Symmetry: Identifiability by Design](https://arxiv.org/abs/2608.27651), preprint

- [You, Jang, Mo & Jung (2026), What Converges in the Platonic Representation Hypothesis? Structure over Geometry](https://arxiv.org/abs/2609.27252), preprint
