# Pilot experiment log

A step-by-step record of every experiment in the pilot (phase 1): what we tried, why, what
happened, and what it means. It exists so the proposal can later be updated with what the pilot
taught us.

**How this file works**

- Entries are numbered and in the order they were run. **Old entries are never rewritten.** If a
  later experiment changes how an earlier result should be read, a new entry says so.
- The only section that gets updated is "Where we are now", just below. Entry titles are a
  summary and may be reworded so that each reads clearly on its own; the entries themselves
  are not.
- Each entry has the same four parts: **Why** (the question or problem behind it), **How** (what
  we did, in plain words first, then the technical specifics: models, settings, data sizes,
  scripts), **Result** and **Meaning**.
- All results are for pilot pair 0 unless stated. Numbers are copied from the run outputs
  (detailed logs are in `runs/pilot-notes.md` and `runs/*.log`, which are not committed).
- The simulator's true states are used **only to score** results, never to fit anything.

## Where we are now

*(Updated as the work moves on. Last updated after experiment 86.)*

**Phase 3, grid video and state vectors with nothing hand-written (entries 56-86).**
- Both observation formats find their structure from their own statistics, and the words'
  kinds come from the language alone; the search leaves the agents' order free.
- **Hunger on grid video is solved under the true matching** (entries 69-76): split moves teach
  the summary model both agents' hunger with no pairing, reliably across move orders.
- **In the pilot world the selection chooses the places' mirror,** a near-symmetry broken only by
  feeding at the bush (violation rate 0.07). That happens on state vectors (entry 71) and on grid
  video (entry 81). Restarting the fits doesn't help: a better fit of the mirror scores higher
  (entry 84).
- **In the lured world** (entry 84), both agents prefer the river and the cave, which raises
  the mirror's violation rate to 0.25. There the grid search rejects every wrong reading of the
  places (entry 85). It confuses only object words that the world barely tells apart (violation
  rates 0.03-0.05). The selected reading is the truth with the stone and apple words exchanged,
  32% of frames exact once that is allowed for; the true reading decodes 43%. **Across the two
  worlds the confusions follow the violation rate,** as H3 predicts.
- **Speed:** a faster filter, a cached context and a CUDA-graph frame sampler reproduce the
  current code exactly on the GPU (entry 83). With them a grid search takes about 4 hours,
  against 8. They are on the `feat/perf` branch, not yet merged.
- **Tuning pairs:** pilot pairs 0-2, grid pair 0 and the lured pairs 0 have been used to make
  choices while looking at the truth, so they can't confirm the method (technical decision 24).
- **State vectors in the lured world: the first full success** (entry 86). With nothing
  hand-written and no hints, the search selects the true reading: 68% of frames exactly right.
  Every wrong reading of the places loses, and by more the more strongly the world breaks it
  (gaps of 0.6, 1.1 and 1.4 at violation rates 0.25, 0.45 and 0.51).
- **Confirmed on both fresh pairs** (entry 87): the frozen method selects the truth on lured pairs
  1 and 2, at 62.9% and 64.0% exact (mean over three pairs 65.0%).
- **Grid video on fresh pair 1** (entry 88): the true matching selected, but with hunger read
  poorly (71-73%), so only 13% of frames are exact.
- **Running:** grid video on fresh pair 2. [TBD]

**The pilot (entries 1-55).** The state of the pilot when it ended:

- Both trained models work. The language model holds the world's state (readable at about 97%),
  and the frame model predicts whole next frames almost perfectly.
- **Pair 0:** the whole pipeline runs without the truth and selects the true matching, decoding
  75% of whole frames exactly right from language alone, with no paired data (experiments
  39-40).
- **Pair 1, a fresh pair, failed at first** (experiment 42): objects at chance. The cause was a
  hidden assumption that held on pair 0 by chance: pair 0's language model stored object
  positions in a form a straight line could read, and pair 1's doesn't. The fix is a word model
  in which each sentence states one fact (experiment 44): pair 1 goes from 0.5% to 43% of frames
  exactly right under the true matching, and pair 0 stays at about 74%.
- **Purity so far:** without the language model's summaries, alignment falls to chance even
  with the better word model (experiments 43 and 45). The trained language model, not raw word
  statistics, carries the alignment.
- **The clean confirmation passed** (experiments 49-51). With the method frozen and no truth
  used, it selects the true matching on pairs 0 and 1, and then on the fresh pair 2, run once:
  **70.1% of held-out frames exactly right** on pair 2 (64.7% on pair 0, 44.0% on pair 1, scored
  as in experiment 55), with the predicted agent swap. On pair 2, raw counts alone would have picked a wrong matching.
- **The pilot's alignment gate is met** (experiments 53 and 55). Mean across the three pairs: 59.6% of
  frames exactly right for the method, 52.6% with a lookup table in place of the frame model, and
  0.8% for the no-neural-model co-occurrence control. Both trained models earn their place, and
  the language model is essential.
- **Variance estimates are in hand** (experiments 54 and 55): restarts within a pair vary by about 2.7
  points (standard deviation), pairs by about 15.8 points around a mean of 60%. Almost all the
  variation is between pairs, as the proposal's unit of analysis assumes. **Both parts of the
  pilot gate are met.**
- **Next:** decide with the user on the open issues below, then the pilot write-up and the
  literature check (phase 2).
- **Open issues for the proposal:**
  - *Change of method: now in the proposal.* The proposal was revised after the pilot to adopt
    decipherment in place of representation-level alignment (which failed in experiments 1-20),
    with a reworded working definition and the lookup-table control added.
  - *The agent swap is fixed by construction.* The search fixes the agents' order, so which agent
    word lands on which frame agent comes from arbitrary numbering. A 50% swap rate across pairs
    would follow automatically, so it can't yet count as evidence for H3. The revised proposal
    requires the search to cover every order; the code still has to be changed to do this.
  - *Hand-built scaffolding.* Structure discovery (one-of-n blocks, the "acted" flags, word types
    from suffixes) and the role tables encode knowledge of how this world and language are
    formatted. No truth goes in, but it limits how general the method is.
  - *Easy on counts.* On pairs 0 and 1, raw word counts already rank the true matching first
    (experiments 19, 37 and 42). On pair 2 they didn't, and the models still found it
    (experiment 51), but the shortlist still relies on counts to cut 144 matchings to 6.
  - *Pair 1 is a tuning pair.* The method was changed after seeing it, so it doesn't count as a
    confirmation. Pair 2 is the confirmation.

## The setup, in brief

- **The world:** two agents (A and B), three objects (glass, stone, apple) and four places on a
  path: river, bush, cave, hill. Agents move between neighbouring places, pick things up, drop
  them (the glass breaks), get hungry, and eat (at the bush, or by eating the apple).
- **Two streams of data, from separate runs of the world:**
  - *Language:* sentences in an invented language describing what happens, such as "A goes to
    the river" or "the glass breaks".
  - *Frames:* a list of 45 yes/no values per moment, such as "A is at the river" or "the stone is
    held by B". The order of the values is shuffled, so their meaning isn't given away.
- **Two models, one per stream,** each trained only on its own stream to predict what comes next.
- **The goal (alignment):** without ever being shown which sentence goes with which frame, work
  out which frame each sentence describes.
- **Symmetry:** A and B behave identically, so no method can tell which agent word means which
  frame agent. Swapping them is the one confusion that *should* remain. The places also have a
  near-symmetry: the path looks the same backwards (river ↔ hill, bush ↔ cave), and only the
  fact that eating happens at the bush breaks it.

## Glossary

- **Probe:** a small model trained *with* the true answers, to check whether information is
  present inside another model. A probe score of 0.97 means the information can be read out 97%
  of the time. Probes are diagnostics only.
- **Chance:** the score you'd get by guessing (for example, shuffling the predictions). A result
  only means something if it clearly beats chance.
- **Held-out:** data kept aside and never used for fitting, so scores on it show whether a
  result generalises.
- **Oracle or ceiling:** the same method fitted using the true answers. It shows the best the
  method could possibly do, and so whether a failure is in the method or in the search.
- **Fit score (log-likelihood per sentence):** how well a model predicts the held-out sentences.
  Closer to zero is better: -4.1 beats -4.2. These numbers can only be compared between runs of
  the same kind of model.
- **Restart:** running the same fitting procedure again from a different random starting point.
  The spread between restarts is the **noise**; a difference smaller than the noise means nothing.
- **EM (expectation-maximisation):** a standard way to fit a model with hidden parts. Guess the
  hidden parts, update the model to fit the guesses, and repeat. It can get stuck in a wrong but
  self-consistent answer (a **local optimum**).
- **Matching:** which invented word means which agent, object and place in the frames. There are
  144 possible matchings: 6 orders for the objects times 24 for the places, with the agents fixed
  because of their symmetry.
- **Particle filter:** a way of tracking a hidden sequence by keeping many guesses ("particles")
  at once. At each step, each guess is extended and weighted by how well it explains the
  evidence, and poor guesses are dropped.

## Experiments

Scripts named below are scratch scripts kept outside the repository; settings that became part
of the package are in `docs/technical-decisions.md`.

### Session 1: first pilot run

**1. First attempt: matching the two models' representations (vec2vec) finds nothing.**
- *Why:* to run the proposal's planned pipeline end to end and see where it stands.
- *How:* we used the proposal's original method, vec2vec. It learns a mapping between the two
  models' internal representations by making their overall distributions look alike.
  - Language model: 4-layer causal transformer, width 128, 0.86M parameters, 10 minutes (14,364
    steps, held-out loss 0.435).
  - Frame model: the same architecture, 20,000 steps in 5.2 minutes (held-out loss 0.116), each
    of the 45 values predicted as an independent yes/no.
  - Aligned: last-layer hidden states at sentence ends against those at frames, using 5,000
    episodes per side.
  - Scoring: 1,000 held-out episodes per side. Each mapped sentence takes the true state of its
    nearest frame.
  - 20 restarts, about 40 seconds each.
- *Result:* chance in all 20 restarts (for example, agent A's place 0.256 against chance 0.258).
- *Meaning:* nothing worked yet. The failure could be in either model or in the aligner;
  experiments 2-9 find out which.

**2. The language model doesn't hold the world's state in a readable form.**
- *Why:* alignment can only work if each model's internal state contains the world's state.
- *How:* probes on each model.
  - Linear probes (logistic regression) per part of the state, trained on alignment-sample
    points and tested on held-out points.
  - An oracle mapping: ridge regression from language states to frames, fitted with true pairs.
  - Layers 2-4 compared.
- *Result:* the frame model holds everything perfectly (1.00). The language model holds little:
  agent places 0.43, objects 0.34-0.43, against guessing the commonest answer at 0.12-0.31. The
  oracle mapping got 0.4% of frames exactly right. Earlier layers were worse.
- *Meaning:* even a perfect aligner would fail; the language side doesn't contain the state in a
  readable form.

**3. The language model tracks the world but keeps no full snapshot of it.**
- *Why:* weak probes could mean the model hasn't learned the world, or that it has learned it but
  doesn't store it where a probe can read it.
- *How:* a behavioural test, in the style of the Othello-GPT legal-move test. Just after a
  sentence's subject (say "A"), we measured how much next-word probability goes to actions that
  are actually possible for A right then, over 20,073 held-out event sentences.
- *Result:* 0.92.
- *Meaning:* it tracks the world well enough to behave correctly, but doesn't keep a full
  snapshot anywhere. It looks up what it needs when it needs it.

**4. Longer training doesn't make the language model store a full snapshot.** (`lang-v2`).
- *Why:* the first model stopped at its 10-minute limit while still improving.
- *How:* the same architecture; learning rate 1e-3 (was 3e-4), warm-up 500 steps, bf16 mixed
  precision (new `amp` option), 30 minutes / 100,000 steps. Best validation loss 0.407 at step
  33,000. Two-layer probes, at subjects and pooled over the last 30 positions.
- *Result:* better behaviour (0.968 on the test in experiment 3). Probes barely improved (pooled:
  places 0.69, objects 0.59-0.69).
- *Meaning:* better prediction doesn't by itself make the model store a full snapshot.

**5. A bigger language model holds more of the state, but not all of it.** (`lang-m`).
- *Why:* larger models are known to form more explicit internal pictures of a world (as in
  Othello-GPT), and the work plan says to scale up if H1 is weak.
- *How:* 6 layers, width 256, 8 heads, about 4.9M parameters (inside the proposal's range);
  learning rate 1e-3, warm-up 1,000, bf16, 60 minutes (66,865 steps). Best validation loss 0.406.
- *Result:* two-layer probes over the last 30 positions: agent places 0.76, objects 0.76-0.84. A
  map fitted with the truth got 8.6% of frames exactly right, up from 0.4%.
- *Meaning:* bigger helps, but still falls well short of a full snapshot (technical decision 15).

**6. Representation matching fails even on the bigger language model.**
- *Why:* the bigger model holds real signal (experiment 5), so the aligner should now have
  something to find.
- *How:* language points pooled over the last 30 positions (new `window` option), 20 restarts.
- *Result:* still chance (A's place 0.213, against chance 0.251).
- *Meaning:* the aligner itself is now the suspect.

**7. Representation matching fails even with perfect representations.**
- *Why:* to test the aligner on a case where the representations are certainly perfect.
- *How:* we aligned one pair's frames to another pair's frames. Both describe the same world
  perfectly, and only the order of the 45 values differs. vec2vec, pilot configuration, raw
  frames on both sides, 10 restarts.
- *Result:* chance in all 10 restarts, though the selection scores were low (0.006-0.018): the
  distributions matched, but the mappings were wrong.
- *Meaning:* **vec2vec can't work here even with perfect representations.** Many wrong mappings
  produce exactly the same overall distribution.

**8. Why matching distributions can't work: snapshots don't show which places are adjacent.**
- *Why:* to see which wrong mappings distribution matching settles on, and why.
- *How:* another distribution-matching method, Gromov-Wasserstein (which matches points by
  comparing distances within each side), on single frames and on pairs of consecutive frames.
- *Result:* places mapped by clean but wrong swaps: bush ↔ cave (seed 3, at 0.96-1.00), a 4-way
  cycle (seed 0). The agents mapped correctly. On pairs of consecutive frames it was muddier.
- *Meaning:* a single snapshot doesn't show which places are next to each other, so places
  visited equally often look interchangeable. Only movement over time can tell them apart.

**9. Adding time to distribution matching helps, but not enough.**
- *Why:* if time is what's missing (experiment 8), adding it might fix distribution matching.
- *How:*
  - vec2vec with extra discriminators on consecutive pairs of frames, 4 restarts; linear maps
    were also tried, and did worse.
  - k-means clusters on each side, then graph matching (FAQ) of the clusters' transition graphs.
- *Result:* the first gave some signal (B-as-A place 0.59, glass 0.31) but didn't solve it. The
  second was at chance.
- *Meaning:* time helps, but adversarial distribution matching is too weak a search, and the two
  sides' clusters don't correspond.

**10. Judging a mapping by the world's dynamics identifies it, up to symmetry.**
- *Why:* if movement over time is what identifies the mapping, use it directly as the score.
- *How:* we translated one pair's frames into the other pair's value order, then asked the other
  pair's trained frame model how believable the translated episodes are. A wrong mapping makes
  impossible things happen, such as an agent jumping between places that aren't neighbours.
  - Judge: pair 0's frame model, scored by its next-frame loss (binary cross-entropy per value)
    on the mapped episodes.
  - Search: pairwise swaps of the 45 values, hill-climbing from a random order, judged on 64
    training episodes; about 20 seconds per restart, 3 restarts.
  - A gradient search over soft permutations (Sinkhorn) only reached 0.45-0.51; random orders
    score 0.68-0.80.
- *Result:* the best restart (judge 0.1180) found the exact mapping, apart from the agent swap.
  The true and agent-swapped orders scored exactly the same (0.1156).
- *Meaning:* **the first method to pass any test.** Dynamics identify the mapping up to, and only
  up to, the world's symmetry, which is the proposal's own claim.

**11. From language to frames, the dynamics judge finds no mapping.**
- *Why:* to apply the method that worked in experiment 10 to the real task.
- *How:* a small neural network (MLP) from `lang-m` pooled points to 45 soft frame values. Loss:
  the judge's loss on the mapped episodes (first 61 points, the frame model's context), plus 10 ×
  MMD (a penalty that pulls the mapped frames' distribution towards real frames). 3 restarts,
  1,500 steps.
- *Result:* chance; judge loss 0.164-0.178.
- *Meaning:* either the judge is wrong for language, or the search can't find the answer;
  experiments 12-14 check which.

### Session 2

**12. The dynamics judge scores language correctly; the search is what fails.**
- *Why:* to separate "the scoring is wrong" from "the search fails".
- *How:* 300 held-out language episodes, mapped to frames using the truth, then scored by the
  judge. Variants used every sentence end, or event sentences only.
- *Result:* true mapping 0.105 (event sentences only: 0.116). Every single wrong swap scored
  worse (places 0.118-0.136, objects 0.157-0.274); only the agent swap tied.
- *Meaning:* the scoring is right; finding a mapping that scores well is the problem.

**13. A cipher control that word frequency alone solves.**
- *Why:* the proposal's upper-bound control: language aligned to a renamed copy of language,
  which should be the easiest real alignment.
- *How:* pair 1's language with its words renamed into pair 0's word numbers, judged by pair 0's
  language model; a pairwise-swap search over the renaming.
- *Result:* true renaming 0.4088, agents swapped 0.4089 (a tie). The swap search from random
  starts stuck at 4.7-5.6 (4-21% of words right). Matching words by frequency rank alone got all
  24 right.
- *Meaning:* with only 24 words, frequency alone solves this control, so it doesn't test the search.

**14. A flexible neural mapping cheats the judge by inventing plausible frames.**
- *Why:* to find out whether a faithful mapping even scores well under the judge.
- *How:* MLP mappings fitted with true frames (binary cross-entropy 0.162), then scored by the judge.
- *Result:* the faithful mapping scored 0.25-0.32 with the judge, *worse* than the unfaithful
  ones from experiment 11 (0.16-0.18), and got only 5% of frames exactly right.
- *Meaning:* a flexible mapping can **invent** believable frame sequences that ignore what the
  language says, and the judge then prefers the invention. The language model was also still too
  weak.

**15. A language model trained to hold a full snapshot of the world.** (`lang-s48`).
- *Why:* the language model was the bottleneck (experiments 2, 5, 14), and it needs to hold the
  whole state at each sentence end, as each frame does.
- *How:* a predictive-state objective. At each sentence end, a 64-number summary must predict the
  next 48 words on its own, through a small 2-layer decoder. To do that it has to record where
  everything is. Training still uses only the language stream.
  - Fine-tuned from `lang-m` for 25 minutes (18,488 steps), with both losses: next word, plus the
    future words from the summary.
  - Next-word loss 0.403 (`lang-m`: 0.406); future loss through the summary 0.42-0.43 per word.
- *Result:* two-layer probes on held-out data: agent places 0.971, hunger 0.967, stone 0.970,
  apple 0.971, glass 0.775. A map fitted with the truth now got **68%** of frames exactly right.
- *Meaning:* the language side is no longer the bottleneck (technical decision 17).

**16. With a better language model, flexible mappings still cheat and constrained ones can't be found.**
- *Why:* with the bottleneck removed, to see whether the judge-based search now works.
- *How:*
  - 1,500 training episodes and 300 held-out.
  - Frame-side groups of values found from frame data alone: sizes 4, 4, 7, 10, 6.
  - MLP map trained against the judge, plus a round trip back to the summary, plus MMD.
  - Variations: hard 0/1 frames (straight-through), swap moves on the outputs, staged training.
  - Basin test: start from the truth, or from the truth with two objects swapped, then train.
- *Result:* chance; at best one agent's place in 2 of 8 restarts. In the basin test, MLP maps
  ended equal (0.153 against 0.154), while linear maps kept the gap (0.153 against 0.220). Linear
  maps from random starts stuck at judge 0.45.
- *Meaning:* flexible mappings cheat; constrained ones can't be found by gradual adjustment.

**17. Matching the slowest-changing patterns doesn't identify the mapping.**
- *Why:* a search-free alternative: slow features describe the world's lasting state, and might
  line up across the two sides.
- *How:* time-lagged component analysis on each side, with components matched by how slowly
  they change.
- *Result:* only the slowest component matched (correlation 0.66). The frame side has many
  components that change almost equally slowly (0.979-0.981), so they can't be told apart.
- *Meaning:* ordering by slowness doesn't identify the mapping in this world.

**18. A second way to cheat the judge: facts that never change look predictable.**
- *Why:* the swap search worked when both sides shared a format (experiment 10), so make the
  language side's summary take the frame's format.
- *How:*
  - On the frozen `lang-s48`: a discrete code with one-of-n groups of sizes 4, 4, 7, 10, 6 plus
    4 yes/no values (Gumbel straight-through), trained only to predict the next 48 words.
  - 20 minutes; future loss 0.444.
  - Then the permutation search from experiment 10.
  - Variations: a penalty on how many groups change between sentences (weights 0.2, 0.02, and
    0.01 with a warm-up), and the judge added to training (weight 1 or 0.1, after 40% of steps).
- *Result:*
  - Groups only partly matched entities: hunger 0.90, agent places 0.54-0.72, objects 0.37-0.40.
  - The penalty either collapsed the code to a constant or left it entangled.
  - With the judge at weight 1, the code **froze**: judge 0.09, *below* the truth's 0.105, while
    its future loss worsened from 0.45 to 0.58.
- *Meaning:* a second way to cheat the judge: things that never change look very predictable.

**19. Raw word counts alone get much of the matching right in this world.**
- *Why:* to see how much of the matching simple statistics give away, which matters for the
  counting baseline.
- *How:* per stream, word counts, frame-value changes and agent arrivals at each place; the
  truth used only to label the table.
- *Result:* objects rank the same on both sides (stone > apple > glass), so counts alone match
  them. Places match only up to the mirror: river 270,281 words / 153,774 arrivals against hill
  271,633 / 155,263; bush 524,436 / 314,239 against cave 446,714 / 313,825.
- *Meaning:* **for the proposal:** in this world a good counting method would get much of the
  matching right, which weakens the planned comparison against a counting baseline.

**20. The predictive-state language model reproduces from scratch.** (`lang-ps`).
- *Why:* experiment 15 fine-tuned an existing model; the result needs to reproduce from scratch,
  inside the package.
- *How:* the head built into the package (`future: 48`, `summary: 64`, `future_layers: 2` in
  `configs/pilot-lang.yaml`); 60 minutes, 38,405 steps; best validation loss 0.826 (next word plus
  future) at step 34,000.
- *Result:* probes: summary 0.97 for agent places, hunger, stone and apple, glass 0.77; last-layer
  hidden state, glass 0.93.
- *Meaning:* experiment 15 reproduces from scratch.

### Session 3: decipherment

From here on the direction changes. Instead of mapping language onto frames (which can cheat by
inventing or freezing), we treat the language as a code describing a hidden sequence of frames,
and ask which frame sequence best explains every word. This is how ciphers and lost scripts have
been deciphered automatically.

**21. Decipherment, first version: the truth doesn't score best.** (`hmm.py`).
- *Why:* a fit that has to explain every word can't ignore its input (no invention), and a frame
  that never changes can't explain changing sentences (no freezing).
- *How:*
  - A hidden Markov model whose hidden states are frames: the 49,272 distinct frames in the frame
    stream's training episodes, with transition probabilities from 389,545 observed transitions.
  - Each sentence either stays on the current frame or takes one step.
  - Emission (in the style of IBM Model 1): each word comes from background word frequencies,
    from one of the frame's "on" values, or from a value the step changed.
  - Fitted by stochastic EM: forward filtering, sampling one path back, re-estimating the tables.
    512 language episodes, 20 iterations, 4 restarts.
- *Result:*
  - A bug first: a smoothing term wasn't multiplied by the emission, so a frequencies-only model
    scored -5.69 against its true -11.8. Fixed.
  - Then the fit preferred wrong answers: random starts reached -10.6 to -10.8, better than tables
    fitted with the true paths (-10.91). EM pushed the stay probability from 0.23 to about 0.05,
    using the extra steps to model grammar.
- *Meaning:* before searching, check that the truth scores best. Here it didn't.

**22. With per-position word tables, random starts still beat the truth.**
- *Why:* to stop the hidden frames being used to model grammar (experiment 21).
- *How:*
  - A separate table for each of the 6 word positions in a sentence, so grammar is carried by
    the tables.
  - Stay probability fixed at 0.230 = 1 - (40.9 - 1) / (52.8 - 1), from the mean frames per
    episode and sentences per episode of the two streams (no pairing).
  - 4 restarts, 25 iterations, 512 episodes.
- *Result:* tables fitted with the truth: -4.24, with 22% of frames exactly right when decoding.
  EM from random starts: -3.86 to -3.94, all at chance.
- *Meaning:* random starts still beat the truth; experiment 23 tests whether that's the model or
  the search.

**23. Started from the truth, decipherment prefers it; from random starts it gets stuck.**
- *Why:* if EM started at the truth drifts away, the model is wrong; if it stays and scores
  better than random starts, the search is the problem.
- *How:* tables fitted with the true paths as the starting point, then 16 ordinary EM iterations.
- *Result:* it stayed near the truth and improved to -3.60, better than any random start. State
  parts read at 0.76-0.88, with 25% of frames exactly right.
- *Meaning:* **the fit now prefers the truth.** The problem is that EM from a random start gets
  stuck.

**24. A starting hint about who is speaking isn't enough.**
- *Why:* a better starting point might steer EM into the right answer.
- *How:*
  - From the frames alone: the two values that switch on at 25.6% of steps (each agent's "acted"
    flag).
  - From the language alone: the two commonest sentence-initial words.
  - Each of the 2 pairings seeded at the subject position (0.9 probability), then 25 iterations.
- *Result:* chance (-3.82, -3.92). The hint washed out within a few iterations.
- *Meaning:* a hint about one part of the model isn't enough; the rest of the tables pull it back.

**25. Each stream's structure can be found from that stream alone.** (`entity.py`).
- *Why:* knowing which words are agents, objects and places, and which frame values belong to
  which thing, reduces the search to 144 possible matchings.
- *How:*
  - Frames: values are grouped into one-of-n blocks if they hand over to each other between
    frames and are never on together.
  - An agent block is one whose values can all appear at an episode's start, and whose every
    change comes with one "acted" flag (more than 99% of the time).
  - Hunger is assigned through the acted flag that comes with it.
  - Object values: "at" values are those on at episode starts; "held" values switch on from an
    "at" value along with one acted flag; the rest are "broken at a place" if the acting agent's
    place is consistent (more than 90%), otherwise "gone".
  - Places are lined up across blocks through take events.
  - Language: the agents are the two commonest sentence-initial words; verbs are words whose
    suffix is followed by a sentence end more than half the time; places take two or more
    different suffixes; objects are the rest.
  - Then: seeding tables for all 144 matchings (half background frequencies, half the matched
    words), ranking them by likelihood on 128 episodes (495 seconds), and EM from the top three.
- *Result:* the structure was exactly right on both sides: all 35 live frame values, and
  2 agents, 3 objects, 4 places and 8 verbs. But the seeded ranking put the true matching 4th,
  with all 144 within 0.004, and EM from the top three ended at -5.3 to -5.4, chance.
- *Meaning:* each stream's structure can be found on its own; seeds that only nudge free tables
  are too weak to rank by.

**26. Word tables shared across roles make the fit prefer the right matching.**
- *Why:* with free tables, a wrong matching can be patched up; tying the tables to the matching
  should stop that.
- *How:*
  - Each emission feature (a value that is on, turned on, turned off, or "nothing changed") has a
    kind, such as "agent place, turned on".
  - All features of a kind share one table over role classes: this agent, another agent, this
    object, another object, this place, another place, or one specific non-entity word.
  - The M-step pools counts into role classes, then expands them back into words through the
    matching.
  - Starting hint, with no truth: pseudo-counts of 20 on each value's own entity words.
- *Result:* with random role tables even the true matching failed. With the hint, the true
  matching read the state above chance for the first time (15 iterations on 256 episodes: agent
  places 0.64, objects 0.57-0.62). Long fits (30 iterations on 512 episodes) ranked the true
  matching first: true -3.939, two objects swapped -3.953, objects cycled -4.011.
- *Meaning:* shared tables make the fit prefer the right matching.

**27. Searching every matching, the places' mirror image wins: the model can't see where eating happens.**
- *Why:* to see whether the fit picks the true matching when nothing gives it away.
- *How:*
  - Stage 1: all 24 place orders, short fits (15 iterations, 256 episodes), each scored on 192
    selection episodes used for neither fitting nor scoring.
  - Then true against mirror: 3 restarts each, 30 iterations on 512 episodes.
- *Result:* stage 1 picked the **mirror** (-4.134 against the truth's -4.172). In the restart
  test the mirror won every time (mean -3.926 against -3.938; noise about 0.003-0.007).
- *Meaning:* the model couldn't see the one clue that breaks the mirror, which is that eating
  happens at the bush.

**28. Reading the language model's summaries instead of words, hunger is read poorly.** (`zhmm.py`).
- *Why:* the summaries hold the state at 0.97 (experiment 20), and using them is H2 as the
  proposal means it: aligning the language model's representation, not raw words.
- *How:*
  - The 64-number `lang-ps` summaries at sentence ends, standardised.
  - Emission: a Gaussian, z ~ N(W f + b, Σ), where f is the frame's values; full covariance,
    fitted by ridge regression on sampled paths.
  - Ceiling: W fitted with the true paths.
  - Then started from the word model's paths, under the true or mirror matching, for 12 EM
    iterations on summaries alone.
- *Result:* ceiling 35.8% of frames exactly right. From the true start: agent places 0.98, stone
  and apple 1.00, glass 0.65, hunger 0.69-0.75, 32% exact. But the mirror start fitted better on
  this model's scale (117.9 against 116.3) and stayed mirrored.
- *Meaning:* hunger was read poorly, and hunger is what breaks the mirror.

**29. A neural reader trained on its own guesses learns its own mistakes.**
- *Why:* a straight-line reading of the summary reads hunger poorly, while the probes read it at
  0.97, so a nonlinear reader might help.
- *How:* a hybrid of a neural network and a hidden Markov model, as in speech recognition. An MLP
  (64 → 256 → 256 → a softmax per block and a sigmoid per single value) was trained for 3 epochs
  per iteration on the sampled paths. Its outputs, divided by each value's frequency in the frame
  stream, were the evidence, softened by a power of 0.5.
- *Result:* worse. It drifted (agent places 0.63 → 0.51 over 12 iterations), with hunger staying
  at about 0.68.
- *Meaning:* a reader trained on its own guesses learns its own mistakes. Dropped.

**30. Adding the actor's place as a word role breaks the places' mirror.**
- *Why:* in "A eats at the bush", nothing linked the word "bush" to where A was standing, so
  getting the place wrong cost nothing.
- *How:* a fourth mixture component in the emission. It puts all its weight on the word for the
  place where the acting agent is in the step's new frame (read from the acted flag and that
  agent's place block), translated by the matching. It needs no verb meanings. Its weight is
  learned per position. 30 iterations on 512 episodes, 2 restarts per order.
- *Result:* the true order won by 0.072 (-3.935 and -3.933, against -4.007 and -4.006), more than
  ten times the noise. Agent places rose to 0.85.
- *Meaning:* **the mirror was broken.** A confusion along the near-symmetry came from the
  aligner being too coarse, not from the world.

### Session 3 continued: two trained models

Experiments 21-30 took their sense of how frames follow one another from a table of the frame
data, not from the trained frame model. The proposal needs two trained models, one per stream,
so from here on the frame side is the trained frame model.

**31. A frame model that predicts whole frames learns the world's structure unaided.** (technical decision 18).
- *Why:* the old model predicted each of the 45 values separately, so it couldn't express "if A
  acted, then A's place changed, not B's", and its samples could be impossible frames. An aligner
  that uses the model needs whole, valid frames.
- *How:*
  - A masked output head (after MADE): 2 hidden layers of 256 units, conditioned on the
    transformer's context. It predicts the values one after another, each given the earlier ones.
  - Options `joint` and `joint_hidden`, with 4 new tests.
  - Trained on frames alone, from scratch: 20,000 steps in 6 minutes, 968k parameters, saved to
    `models/obs-joint`.
- *Result:* held-out loss 0.047 per value (old model 0.116), still improving at the last step.
  99.6% of 8,030 sampled next frames, from 200 held-out episodes, are frames the stream actually
  shows.
- *Meaning:* the trained frame model has learned the world's structure without being told it, and
  can stand in for the table.

**32. Decipherment with a particle filter works, but learns slowly.** (`smc.py`).
- *Why:* a trained model predicts from the whole history, so the table-based method no longer
  applies.
- *How:*
  - 32 particles per episode. At each sentence a particle either stays (probability 0.23) or
    moves to one of 4 candidate frames sampled from the joint head. Particles are weighted by the
    sentence's emission and resampled within each episode.
  - Start frames: a uniform prior over the 6,720 combinations of one value per block, with the
    single values off.
  - The shared tables and actor-place role from experiments 26 and 30. The M-step uses one
    particle's path per episode.
  - 15 iterations on 512 episodes, about 90 seconds per iteration.
- *Result:* agent places 0.47 (chance about 0.33), objects close to chance, still improving
  steeply. True -4.2991, mirror -4.3047: smaller than the likely noise.
- *Meaning:* it works, but learns much more slowly than the table version, because each history
  is a sample rather than an exact sum over every frame.

**33. With both trained models, agents and hunger are decoded; objects aren't.**
- *Why:* the summaries carried most of the accuracy in experiment 28.
- *How:* the Gaussian from experiment 28, refitted every iteration from the sampled paths,
  switched on from iteration 5.
- *Result:*
  - At full strength it swamped the words, the tables drifted, and the run crashed on NaN weights
    at iteration 12.
  - Softened by a power of 0.2, with the M-step using 8 particle paths per episode and guards
    against NaN weights, 25 iterations: agent places 0.96-0.97, hunger 0.95. Objects rose to
    about 0.5 by iteration 10, then fell back to 0.28-0.36.
- *Meaning:* agents and hunger are solved with two trained models; objects aren't.

**34. Handling each episode's start exactly fixes the objects.**
- *Why:* objects rarely move, so their positions come from the opening sentences. But each
  particle chose its start frame from the first sentence only, so often no particle had the right
  objects.
- *How:* until a particle first moves, each episode keeps an exact distribution over all 6,720
  start frames, updated by every sentence (Rao-Blackwellising the start). A particle draws its
  start from it at its first move, and its earlier sentences are filled in with that start.
  25 iterations on 512 episodes.
- *Result:*

  | | Best point (iteration 10, 128 held-out episodes) | End (iteration 25, all 300 held-out) |
  |---|---|---|
  | Frames exactly right | **60.8%** | 54.2% (agents swapped: 5.1%) |
  | Agent places | 0.98-0.99 | 0.97-0.98 |
  | Hunger | 0.87-0.89 | 0.96-0.97 |
  | Glass / stone / apple | 0.87 / 0.97 / 0.97 | 0.83 / 0.87 / 0.91 |

- *Meaning:* the breakthrough, close to the 68% ceiling of experiment 15, from language alone
  with two trained models. But accuracy slid after its peak while the fit kept improving. The
  mixture weights drifted towards the "something changed" component (0.99 at the subject
  position).

**35. Stopping accuracy sliding; the fit prefers the truth over the mirror.**
- *Why:* to stop the drift seen in experiment 34, and to check that the fit prefers the true
  order with two trained models.
- *How:* mixture weights frozen after iteration 8 (`lamfreeze=8`); 20 iterations; identical
  settings for the true and mirror orders.
- *Result:* true order **58.4%** of frames exactly right (peak 64.7% at iteration 15), fit
  -8.3516. Mirror 0%, fit -8.4114. A check on 128 episodes had pointed the other way (-8.33
  against -8.47).
- *Meaning:* with two trained models, the fit prefers the truth on all 300 held-out episodes; 128
  episodes are too few to select by.

**36. Faster code alone can't make a full search affordable.**
- *Why:* each run took 35-90 minutes, too slow to search many matchings.
- *How:*
  - Start-frame emissions precomputed once per iteration as a table, not per episode in Python.
  - The joint head's context projections computed once, not on each of its 45 passes.
  - The frame model run only up to the longest history.
  - Decoding on the GPU, with frames packed into integers.
  - Two jobs run side by side as a throughput test.
- *Result:* each piece was 2-600 times faster on its own, but the whole loop only 1.6-2.2 times
  (6-9 seconds per iteration on 64 episodes, against 13). Two jobs at once stalled almost
  completely: neither finished its first iteration in 10 minutes, against 23 seconds alone.
- *Meaning:* faster code alone won't make a full search affordable, and the GPU can't be shared,
  so the search itself has to be designed to be cheap.

**37. Relabelling one fit can't score other matchings; each must be fitted separately.**
- *Why:* the shared role tables don't depend on the matching, so one fit might be able to score
  every matching cheaply.
- *How:*
  - Starting matching from counts alone: objects by word count against frame changes; places by
    word count against agent arrivals.
  - Fit (12 iterations, 512 episodes). Then for each candidate, pool the role tables from the
    fit's counts and expand them under the candidate's words.
  - Score each candidate with one filter pass on 128 selection episodes, with the same random
    seed for all; about 15 seconds each.
  - Refit under the winner and repeat. Also run from deliberately wrong starts (the mirror, and
    a scrambled matching).
- *Result:*
  - The starting matching from counts was **exactly the true matching**.
  - From it, the search kept the truth by wide margins (places -8.24 against -8.53; objects
    -8.24 against -8.88). Under it: agent places 0.98, hunger 0.97, stone 0.94, apple 0.95,
    glass 0.54; 41.5% of frames exactly right.
  - **Started from the mirror or a scrambled matching, it kept those too.**
  - Comparing the three separate fits ranked them correctly: true -8.24, mirror -8.37,
    scrambled -8.50.
- *Meaning:* a fit's paths and summary evidence lean towards the matching it was fitted under, so
  relabelling only confirms that matching. Fitting each candidate separately works. **For the
  proposal:** in this world, raw counts find the matching on their own.

**38. Words alone can't rank matchings from tables fitted under a wrong one.**
- *Why:* the summary evidence ties scores to the fitted matching, so scoring with words alone
  might remove that lean.
- *How:* as in experiment 37, but scoring without the summary evidence, so each candidate's paths
  are inferred afresh from its own words. Started from the mirror.
- *Result:* the top five scores fell within 0.007, and the true order came 5th. Stopped early.
- *Meaning:* role tables fitted under a wrong matching carry too little information for words
  alone to rank alternatives.

**39. A two-stage search decodes 75% given the right matching, but selects a wrong one.**
- *Why:* separate fits rank matchings correctly (experiment 37), but fitting all 144 would take
  about 36 hours, so cut the field cheaply first.
- *How:*
  - Shortlist: all 144 matchings ranked by how far apart the two streams' frequency profiles are
    under each one. That is the sum of squared differences in log shares: object word counts
    against object change counts, and place word counts against agent arrivals.
  - Keep the top 6.
  - Fit each finalist separately (12 iterations, 512 episodes) and pick the one with the best
    score for its own matching on the 128 selection episodes, with the same random seed for all.
- *Result:*
  - Shortlist: the true matching ranks 1st. But the top four (the truth, two half-mirrors and the
    full mirror) are tied on counts to the fourth decimal place (0.1762-0.1764), so the separate
    fits must tell them apart.
  - Finalist 0 (the true matching): own score -8.2949. Truth check: **75.3% of frames exactly
    right** (agents swapped 6.6%); agent places 0.98, hunger 0.96-0.97, glass 0.92, stone 0.96,
    apple 0.96. That is above the 68% ceiling of experiment 15.
  - All six finalists, scored on their own matching (higher is better), with a truth check:

    | Finalist | Own score | Frames exactly right |
    |---|---|---|
    | 0: true | -8.2949 | **75.3%** |
    | 1: bush ↔ cave | -8.3298 | 2.2% |
    | 2: river ↔ hill | **-8.2169** | 0.4% |
    | 3: full mirror | -8.5027 | 0% |
    | 4: stone ↔ apple | -8.4188 | 7.9% |
    | 5: stone ↔ apple and bush ↔ cave | -8.5096 | 0.8% |

  - The search selected finalist 2, which is wrong.
- *Meaning:* given the right matching, two trained models decode 75% of whole frames exactly
  right. But selecting by each fit's own score failed: a wrong matching (river ↔ hill) scored
  best while reading almost nothing correctly. Experiment 40 finds out why.

**40. With corrected scoring, the pipeline selects the true matching with no truth used.**
- *Why:* the score for the summary evidence was a Gaussian (bell-curve) density, and the code had
  left out its width penalty: the term -0.5 log det(Σ), where Σ is the fitted noise spread. That
  term is a constant within one fit, so it didn't matter while fitting. But it matters when
  different fits are compared. A wrong matching explains the summaries badly, so its fitted
  spread is wide, and without the penalty a wide spread makes everything look close. So worse
  fits scored better.
- *How:*
  - Added the term to the Gaussian's score (`Gauss.set_norm` in `smc.py`), including the
    start-frame scoring.
  - Rescored the six saved finalist fits from experiment 39, with no refitting, on the same 128
    selection episodes, with 2 random seeds each.
  - Two scores per finalist: the corrected full score (words plus summaries), and words only.
  - About 37 seconds per scoring pass.
- *Result:* (means of the 2 seeds; higher is better; seed-to-seed noise about 0.1 for the full
  score and about 0.01 for words only)

  | Finalist | Corrected full score | Words only | Frames exactly right (truth check) |
  |---|---|---|---|
  | **0: true** | **18.23** | **-5.028** | **75.3%** |
  | 4: stone ↔ apple | 17.60 | -5.048 | 7.9% |
  | 3: full mirror | 17.40 | -5.048 | 0% |
  | 2: river ↔ hill | 16.42 | -5.055 | 0.4% |
  | 1: bush ↔ cave | 16.31 | -5.110 | 2.2% |
  | 5: stone ↔ apple and bush ↔ cave | 16.05 | -5.158 | 0.8% |

- *Meaning:*
  - **With the corrected score, the whole pipeline selects the true matching without using the
    truth**, by a margin of 0.63 over the next best, about six times the seed noise. The closest
    rivals are those with fewer things wrong.
  - Words alone also rank the truth first, but only by about 0.02 over the mirror and the object
    swap, which is close to the noise. **The language model's summaries are what make the
    selection confident.**
  - Caveat: the scoring was corrected *after* it picked a wrong answer on this pair. The fix is a
    missing term in a standard formula, not a tuned choice. But the method has to be frozen and
    confirmed on a fresh pair, ideally one where raw counts don't already rank the truth first,
    before this result counts.

### Session 4: packaging and the fresh pair

**41. The method moved into the package and frozen.**
- *Why:* to confirm the result on a fresh pair, the method has to be fixed in advance and run from
  tested code, not from scratch scripts.
- *How:*
  - `grounding/align/structure.py` finds each stream's structure. It is generalised so it no
    longer assumes two agents: the "acted" flags are found as the set of single values of which
    exactly one is on after every step. On pair 0 it reads all 35 live values correctly.
  - `grounding/align/decipher.py`: the particle filter, shared tables, fitting and the two-stage
    search.
  - Scoring: `grounding.analysis.score --kind decipher`, behind the pre-registration firewall,
    using a new frame-to-state decoder in the renderer.
  - Settings frozen in `configs/pilot-decipher.yaml`; recorded as technical decision 19.
  - 7 new tests on a tiny world, CPU only; 146 tests pass in all.
- *Result:* the tests found one flaw carried over from the scratch code. Smoothing was added to
  role classes that have no words for a kind of feature, so the tables summed to slightly less
  than 1 (0.9987 in the tiny test). Fixed, so every table is an exact distribution.
- *Meaning:* the package version is the scratch method with that fix, and with its own seeded
  sample of episodes. Pair 1 is its first real test.

**42. On a second pair, objects fail: its language model stores them in a tangled form.**
- *Why:* pair 1 is a fresh corpus pair with its own two trained models (language model held-out
  loss 0.825, frame model 0.047, both about the same as pair 0's). It was the first test of the
  frozen method.
- *How:*
  - Ran the frozen pipeline (`python -m grounding.align.decipher runs/pilot/pair-001`).
  - Stopped it after the first finalist and checked it against the truth, because its numbers
    looked wrong.
  - Then four diagnostics, each using the truth, to find the cause:
    1. The packaged code on pair 0 with the true matching, at small scale (128 episodes,
       8 iterations).
    2. The same at full scale (512 episodes, 12 iterations), to separate a bug in the packaged
       code from a problem with pair 1.
    3. Pair 1's count shortlist against the true matching.
    4. *Probes* (small classifiers trained to read a fact out of the language model's summaries)
       on both pairs, in two forms: a two-layer probe ("is the fact there at all?") and a linear,
       straight-line probe ("can a straight-line read-out reach it?").
- *Result:*
  - First finalist on pair 1: agent places 0.97-0.98 and hunger 0.95-0.96, but objects only
    0.43-0.47, even on the opening sentences, where they are easiest.
  - Diagnostic 1: objects poor on pair 0 too (stone 0.45, apple 0.54), but this scale had never
    been enough on pair 0 either.
  - Diagnostic 2: **the packaged code is fine.** On pair 0 at full scale it decodes **78.7% of
    frames exactly right** (the scratch code: 75.3%), objects 0.95-0.98. Its fit tracks the
    scratch code's within 0.01 per sentence.
  - Diagnostic 3: on pair 1 the raw counts ranked the true matching first, and that was the
    finalist fitted. So the objects failed with every word correctly assigned.
  - Diagnostic 4:

    | Object location | Pair 0: two-layer / linear probe | Pair 1: two-layer / linear probe |
    |---|---|---|
    | Glass | 0.77 / 0.75 | 0.78 / **0.43** |
    | Stone | 0.97 / 0.95 | 0.97 / **0.48** |
    | Apple | 0.97 / 0.97 | 0.97 / **0.63** |

    Agent places and hunger read out at 0.97 both ways on both pairs.
- *Meaning:*
  - Both language models know where the objects are. But pair 1's stores it in a tangled form:
    where an object is can only be read in combination with other facts, such as where the
    agents are. The aligner modelled each summary as a straight-line function of the frame's
    single values, so on pair 1 it couldn't use the summaries for objects at all, and the words
    alone weren't enough. Pair 0 worked because its language model happened to store objects in
    a straight-line form. **The method had a hidden assumption that held on pair 0 by chance.**
  - The fix keeps the same kind of model but gives it richer inputs: the frame's values *and the
    product of every pair of them* (for example "the stone is at the river" × "A is at the
    river"). That is 35 live values plus 595 products. It uses only the frame's own structure,
    nothing from the truth.
  - Checked offline before any GPU run, by fitting the summary model on true frames and asking
    which position of each object the summary points to:

    | Object (layout order) | Pair 1: linear / with products | Pair 0: linear / with products |
    |---|---|---|
    | Object 0 | 0.44 / **0.71** | 0.96 / 0.97 |
    | Object 1 | 0.54 / **0.96** | 0.74 / 0.76 |
    | Object 2 | 0.65 / **0.95** | 0.97 / 0.97 |

    It repairs pair 1 and costs pair 0 nothing.
  - Code: `summary_features: pairs` in `configs/pilot-decipher.yaml`; `Gauss` in
    `grounding/align/decipher.py` builds the products and fits in chunks to bound memory; one
    new test. **Caveat:** this change was made *after* seeing pair 1 fail, so pair 1 is no longer
    a clean confirmation. A third pair, run with the method as it now stands, is needed for that.

**43. A richer summary model confirms the filter's mistakes instead of correcting them.**
- *Why:* entry 42's offline check fitted the summary model to the *true* frames. In the real
  method it is fitted to the particle filter's own guesses, which start out wrong. Separately,
  the purity question: how much do the raw words do without the language model?
- *How:*
  - The frozen pipeline on pair 1 with `summary_features: pairs`, checked against the truth after
    its first finalist (the true matching); the run was then stopped.
  - The true matching on pair 1 with the summaries switched off: trained frame model plus raw
    words only, no language model (512 episodes, 12 iterations).
- *Result:*
  - With pairwise features, objects got **worse**: glass 0.38-0.41, stone 0.24-0.27, apple
    0.39-0.41. Hunger fell from 0.95 to 0.78-0.84. Agent places held at 0.97-0.98.
  - Words only: agent places 0.42-0.46, hunger 0.62, objects 0.18-0.28, 0.1% of frames exactly
    right. That is near chance throughout.
- *Meaning:*
  - The richer summary model is flexible enough to fit the filter's wrong early guesses well, so
    it confirms them instead of correcting them. A summary model fitted to its own decoding needs
    to be simple enough that the truth fits it better than a wrong guess does. Pairwise features
    are dropped from the method.
  - **For purity, a good sign:** with the current word model the raw words achieve almost
    nothing alone. On pair 0, the language model's summaries were doing the work, agents
    included. The words are needed to start the process, and the summaries finish it.
  - That points the fix at the word side. The word model explains each word separately, each
    from any fact in the frame, so nothing ties "stone" and "river" in "the stone is at the
    river" to the same fact: a frame with the stone elsewhere and an agent at the river explains
    the words almost as well. Real sentences state one fact or one change each, so entry 44 tests
    a word model in which every word of a sentence comes from a single fact.

**44. A word model in which each sentence states one fact.**
- *Why:* entry 43 found that the word model can't tie a sentence's words to the same fact. Real
  sentences state one fact or one change each, for example "stone cave *be*" (the stone is at the
  cave), "A hill *go*" (A goes to the hill), "A stone *take*" (A takes the stone).
- *How:*
  - New option `emission: sentence` (`SentenceFit` in `grounding/align/decipher.py`). A sentence
    first picks one *focus*: one value that is on in the frame, or, after a step, one change.
    Every word of the sentence then comes from that focus's table, from the background
    (word frequencies), or from the acting agent's place.

        P(sentence) = sum over foci f of  pi(f) × product over positions j of
                      [mu0 × background_j(word) + mu1 × table_j(word | f) + mua × (word = actor's place)]

  - It uses the same shared role tables as before ("this object", "this place", and so on), so
    the matching still turns roles into words, and the same particle filter, start handling and
    summaries. EM now also learns which focus each sentence had.
  - Two new tests: the fast start-frame odds must equal the general formula for both word
    models.
  - Pair 1, true matching, straight-line summaries, 512 episodes, 12 iterations (it uses the
    truth only to score).
- *Result:*

  | Pair 1, true matching | Old word model (entry 42) | One fact per sentence |
  |---|---|---|
  | Frames exactly right | 0.5% | **43.0%** |
  | Agent places | 0.97-0.98 | 0.97-0.99 |
  | Hunger | 0.95-0.96 | 0.96-0.97 |
  | Glass / stone / apple | 0.43-0.47 | **0.67 / 0.86 / 0.78** |
  | Fit at iteration 4 (words only so far) | -5.560 | -5.041 |

- *Meaning:* tying each sentence to one fact makes the words far more informative, and objects
  are now mostly right on a pair where the summaries can't see them in a straight line. Objects
  are still short of pair 0's 0.95. Next: check that this doesn't break pair 0, and how much of
  it is words alone (entry 45).

**45. The one-fact word model holds up on the first pair; without the language model, alignment fails.**
- *Why:* to check that the change in entry 44 doesn't break pair 0, and to measure how much of
  it comes from the words alone, with no language model (a purity check: the proposal needs the
  trained language model to do the work).
- *How:* both with the true matching, 512 episodes, 12 iterations:
  - pair 0 with `emission: sentence` and straight-line summaries;
  - pair 1 with `emission: sentence` and the summaries switched off, so the only language input
    is the raw words.

  From here on, GPU jobs run through an unattended queue (`runs/queue/`) that doesn't depend on
  the Claude session, after a dropped connection lost a night of work.
- *Result:*

  | | Frames exactly right | Agent places | Hunger | Glass / stone / apple |
  |---|---|---|---|---|
  | Pair 0, old word model (entry 42) | 78.7% | 0.97-0.98 | 0.96-0.97 | 0.95 / 0.96 / 0.98 |
  | Pair 0, one fact per sentence | 74.2% | 0.97-0.98 | 0.97 | 0.91 / 0.98 / 0.94 |
  | Pair 1, one fact per sentence | 43.0% | 0.97-0.99 | 0.96-0.97 | 0.67 / 0.86 / 0.78 |
  | Pair 1, one fact per sentence, **words only** | **0.2%** | 0.49-0.50 | 0.69-0.73 | 0.35 / 0.39 / 0.37 |

  These checks decode 64 held-out episodes, so differences of a few points are within noise.
- *Meaning:*
  - On pair 0 the new word model is about as good as the old one, and on pair 1 it is far
    better, so it replaces the old one.
  - **Without the language model, alignment fails even with the better word model.** The words
    alone explain the sentences well (fit -3.77 per sentence) but can't pin down which frame they
    describe. The language model's summaries are what make alignment work. Together with
    entry 43, this is the first direct evidence that the trained language model, not raw word
    statistics, carries the alignment.

**46. Rules fixed in advance for the confirmation runs.** (written before any of their results).
- *Why:* the method changed after pair 1 was seen (entries 42-45), so the remaining choices and
  the test on a fresh pair must follow rules set before the outcomes are known, and must run
  unattended.
- *How:* a job queue (`runs/queue/jobs.txt`, run by `runs/queue/runner.ps1`) works through these
  steps one at a time:
  1. **Summary inputs** (`decide.py`): straight-line, unless the pairwise version beats it on
     pair 1 (true matching) by at least 5 points of frames exactly right. If pairwise wins, it
     must also not cost pair 0 more than 5 points (`decide2.py`), or it reverts to straight-line.
     The choice and `emission: sentence` are written into `configs/pilot-decipher.yaml`, which
     is then frozen.
  2. **Pair 2's models**: language model and frame model trained from scratch on pair 2's own
     streams, with the same settings as pairs 0 and 1.
  3. **The full frozen pipeline on pair 1, then pair 0** (`--name decipher-v2`): shortlist, fit
     six finalists, select by own score. No truth used.
  4. **The gate** (`gate.py`): pair 2 runs only if the selected matching is the true one on
     *both* pair 0 and pair 1. Both are scored against the truth.
  5. **If the gate passes, pair 2**: the full frozen pipeline, untouched, then scored once. The
     result is reported whatever it is, and the method is not changed in response to it.
  6. **Purity checks on pair 1** (true matching): counted transitions instead of the frame model,
     with and without the summaries; and both agent orders, to check they tie (entry 41's agent
     ordering).
- *Result:* see the entries that follow.
- *Meaning:* step 5 is the pilot's first clean confirmation on a pair the method never saw.

**47. Pairwise summary inputs fit their own mistakes, and are dropped.**
- *Why:* entry 43's pairwise inputs failed with the old word model, whose early guesses were
  poor. With the new word model the guesses are far better (entry 44), so pairwise inputs might
  now help pair 1's objects instead of confirming mistakes. This is step 1 of entry 46.
- *How:* pair 1, true matching, `emission: sentence`, `summary_features: pairs`, 512 episodes,
  12 iterations. Compared with entry 44's straight-line run by the rule in entry 46.
- *Result:*

  | Pair 1, one fact per sentence | Frames exactly right | Agent places | Hunger | Glass / stone / apple |
  |---|---|---|---|---|
  | Straight-line summary inputs (entry 44) | **43.0%** | 0.97-0.99 | 0.96-0.97 | 0.67 / 0.86 / 0.78 |
  | Pairwise summary inputs | 8.2% | 0.93-0.98 | 0.95-0.97 | 0.43 / 0.66 / 0.44 |

  `decide.py`: "linear 0.430, pairs 0.082, chose linear". `configs/pilot-decipher.yaml` now has
  `summary_features: linear` and `emission: sentence`, frozen.
- *Meaning:* even with good early guesses, a summary model with 630 inputs fits the guesses'
  mistakes too, and confirms them. A summary model learned from its own decoding has to be
  simple. Pairwise inputs stay in the code as an option but are not part of the method.

**48. The third pair's two models are trained.**
- *Why:* pair 2 is the fresh pair for the clean confirmation (entry 46). It needs its own two
  models, trained from scratch on its own streams.
- *How:* corpus generated with `python -m grounding.data.make configs/pilot.yaml --pairs 2`
  (50,000 episodes per stream, root seed 2), then the same training settings as pairs 0 and 1:
  the predictive-state language model (60-minute limit) and the joint-head frame model
  (20,000 steps).
- *Result:* language model held-out loss **0.826** (best step 34,000 of 41,450; pair 0: 0.826,
  pair 1: 0.825). Frame model validation loss **0.048** (pair 1: 0.047).
- *Meaning:* all three pairs' models are trained to the same standard, so differences between
  pairs come from the pairs themselves. Nothing on pair 2 has been aligned or scored yet.

**49. The frozen method works end to end on the second pair.**
- *Why:* step 3 of entry 46. Entries 44-47 gave the method the true matching. Here it must find
  the matching itself, with no truth used, through the shortlist and the selection.
- *How:* `python -m grounding.align.decipher runs/pilot/pair-001 --name decipher-v2` with the
  frozen config (`emission: sentence`, `summary_features: linear`). Shortlist of 6 by raw counts,
  each fitted (512 episodes, 12 iterations), selected by own score on 128 selection episodes
  (2 seeds). 68 minutes. Checked against the truth afterwards.
- *Result:*

  | Finalist | Matching (from the truth check) | Selection score | Gap to the truth |
  |---|---|---|---|
  | **1** | **true** | **18.42** | **0** |
  | 5 | stone ↔ apple | 18.09 | -0.33 |
  | 4 | full place mirror | 17.94 | -0.49 |
  | 2 | river ↔ hill | 17.34 | -1.09 |
  | 3 | bush ↔ cave | 16.99 | -1.43 |
  | 6 | stone ↔ apple and river ↔ hill | 16.87 | -1.55 |

  Seed-to-seed spread: 0.27 for finalist 1, 0.03-0.08 for the others. **Selected: finalist 1, the
  true matching.** Under it, on all 300 held-out episodes: **43.8% of frames exactly right** (with
  the agents swapped: the predicted symmetry), agent places 0.96-0.97, hunger 0.96-0.97, glass /
  stone / apple 0.74 / 0.78 / 0.80.
- *Meaning:*
  - The method now works on pair 1 end to end, without the truth. Half of the gate for pair 2 is
    passed.
  - The closest rivals are the ones the world predicts: the object swap, because objects are
    pair 1's weak point, and the full place mirror, the near-symmetry broken only by eating at the
    bush. The margin over the object swap (0.33) is about 1.2 times finalist 1's own seed spread,
    so on this pair the selection is right but not by a wide margin.

**50. The frozen method works end to end on the first pair, so the confirmation runs.**
- *Why:* steps 3 and 4 of entry 46: the new method must also find pair 0's matching by itself,
  and pair 2 may run only if both pairs select the true matching.
- *How:* as entry 49, on pair 0 (67 minutes). Then `gate.py` compared each pair's selected
  matching with the true one and scored both.
- *Result:*

  | Finalist (pair 0) | Selection score | Gap to the truth |
  |---|---|---|
  | **1: true** | **19.29** | **0** |
  | 4 | 18.81 | -0.48 |
  | 5 | 18.79 | -0.50 |
  | 3 | 18.17 | -1.12 |
  | 2 | 17.84 | -1.45 |
  | 6 | 17.67 | -1.62 |

  **Selected: finalist 1, the true matching.** Under it, on all 300 held-out episodes: **64.6% of
  frames exactly right** (agents swapped), agent places and hunger 0.97, glass / stone / apple
  0.88 / 0.93 / 0.92. Gate output: "pair-000: true matching YES; pair-001: true matching YES;
  GATE PASSED: pair 2 queued".
- *Meaning:*
  - The frozen method finds the true matching on both pairs without the truth, so pair 2 runs.
  - The new word model costs pair 0 about 10 points (75.3% with the old one, entry 40), in return
    for working on pair 1 at all (0.5% to 43.8%). A method that works on both pairs is worth more
    than one that fits pair 0's luck.
  - On both pairs the closest rivals sit about 0.3-0.5 below the truth. The selection is right,
    but its margin is modest.

**51. Clean confirmation: 70% exactly right on a fresh pair, with no truth used.**
- *Why:* step 5 of entry 46. Pair 2 is a fresh pair the method had never seen, run once with the
  method frozen. This is the pilot's first confirmation that doesn't depend on any choice made
  after seeing the pair.
- *How:* `python -m grounding.align.decipher runs/pilot/pair-002 --name decipher-v2` (67 minutes),
  then `python -m grounding.analysis.score runs/pilot/pair-002 --kind decipher --name decipher-v2`,
  both run automatically by the queue after the gate. No finalist was checked against the truth
  before the scoring. The matching check below was done after the scoring, for the record.
- *Result:*

  | Finalist | Raw-count rank | Objects | Places | Selection score |
  |---|---|---|---|---|
  | 1 | 1st (0.1805) | right | wrong | 17.51 |
  | **2** | **2nd (0.1808)** | **right** | **right** | **19.27** |
  | 3 | 3rd | right | wrong | 18.81 |
  | 4 | 4th | right | wrong | 17.63 |
  | 5 | 5th | wrong | wrong | 17.26 |
  | 6 | 6th | wrong | right | 18.61 |

  **Selected: finalist 2, the true matching**, 0.46 ahead of the runner-up. Scored on all 300
  held-out episodes: **70.1% of frames exactly right** allowing for the agent swap (the agents
  came out swapped); chance 0.02%. Against the swapped truth: agent places 0.98, hunger 0.97,
  glass / stone / apple 0.89 / 0.97 / 0.91.
- *Meaning:*
  - **On a fresh pair, with no paired data and no truth, the two trained models decode 70% of
    held-out sentences into exactly the right world state.** The only systematic confusion is the
    agent swap, which is the symmetry the proposal predicts (with the caveat in technical
    decision 19 that the agents' order is fixed by construction).
  - **Raw counts alone would have chosen wrong here:** they ranked a place swap first. The
    separately fitted models chose the truth. This partly answers the concern that the pilot
    world is "easy on counts": on this pair, counting wasn't enough.

**52. What each trained model contributes, and the agent-order tie.**
- *Why:* the proposal's claim is about what the two *trained models* learned, so each one has to
  earn its place. And the agent order is fixed in the search (technical decision 19), so the
  evidence should score both orders the same.
- *How:* pair 1, true matching, 512 episodes, 12 iterations, `emission: sentence`. Each trained
  model switched off in turn:
  - the frame model replaced by `frame_model: counts`: next frames drawn from the frame stream's
    own counted transitions out of the current frame, with no neural network (the nearest seen
    frame where a frame was never seen);
  - the language model's summaries switched off, leaving only the raw words.

  And the normal setup with the agents' order flipped (`agents=swap`). A first attempt at the
  counted runs crashed (integer matrix multiplication isn't supported on the GPU); fixed and
  rerun.
- *Result:*

  | Pair 1, true matching | Summaries on | Words only |
  |---|---|---|
  | **Trained frame model** | **43.0%** (fit 18.51) | 0.2% |
  | **Counted transitions** | 37.0% (fit 17.63) | 1.8% |

  Agents' order flipped: fit **18.52** against 18.51 in the normal order, 31.6% of frames exactly
  right (agent places 0.97-0.98, hunger 0.96, objects 0.63-0.76). The counted runs took about 2
  minutes each, against about 11.
- *Meaning:*
  - **The language model is essential.** Without its summaries, alignment is at chance whichever
    frame side is used.
  - **The frame model, in this world, improves the fit but adds little accuracy** over a lookup
    table of the frame data. The pilot world is small enough to enumerate: the table holds 49,219
    distinct frames, so it already knows almost every transition. The pilot can't show the frame
    model earning its place; that needs the richer worlds of the main phase, where a table can't
    cover the states.
  - **The two agent orders tie** (a difference of 0.01, within noise). The evidence genuinely
    can't tell the agents apart, as predicted. The 12-point difference in frames exactly right
    comes from run-to-run variation in the objects, not the agents. Restarts within one pair vary
    this much, which matters for the variance estimates the pilot gate asks for.
  - Next: both counted versions as full pipelines on all three pairs. Counted transitions with
    words only is the proposal's no-neural-model co-occurrence control, which the method has to
    beat (entry 53).

**53. The controls as full pipelines: the pilot's alignment gate is met.**
- *Why:* the pilot gate asks that alignment beat the co-occurrence baseline. Entry 52 compared the
  models under the true matching on one pair; this runs each control the way the method runs,
  finding its own matching with no truth, on every pair.
- *How:* `python -m grounding.align.decipher` with two control configs, then scoring:
  - `configs/pilot-decipher-baseline.yaml`: **no neural model.** Counted transitions instead of
    the frame model, and no language model (raw words only). This is the co-occurrence control:
    raw word statistics against raw frame statistics, through the same word model and search.
  - `configs/pilot-decipher-counts.yaml`: counted transitions instead of the frame model, with the
    language model's summaries.

  About 8-9 minutes per pair each. Running the controls on pair 2 doesn't change or tune the
  method.
- *Result:* (frames exactly right on 300 held-out episodes, allowing for the agent swap)

  | | Pair 0 | Pair 1 | Pair 2 | Mean |
  |---|---|---|---|---|
  | **Method:** trained frame model + language model | **64.6%** | **43.8%** | **70.1%** | **59.5%** |
  | Counted transitions + language model | 61.9% | 32.9% | 63.0% | 52.6% |
  | No neural model (counted transitions + raw words) | 0.8% | 1.0% | 0.6% | 0.8% |
  | Chance (shuffled predictions) | 0.03% | 0.02% | 0.02% | |

  Every version selected the true matching on every pair.
- *Meaning:*
  - **The pilot's alignment gate is met:** the method beats the co-occurrence control on all three
    pairs, by 43 to 69 points.
  - **Both trained models earn their place.** Without the language model, decoding collapses to
    about 1%. The trained frame model adds 2.7, 10.9 and 7.1 points over a lookup table, positive
    on every pair, even in a world small enough to enumerate. (Entry 52's single comparison under
    the true matching was too noisy to show this.)
  - The raw words are enough to pick the right matching in this world, but not to decode states.
    Matching words to entities is the easy part here; reading the state of the world from a
    sentence is what needs the trained language model.

**54. Results vary far more between pairs than between restarts.**
- *Why:* the pilot gate asks for variance estimates for the power analysis. The proposal's unit of
  analysis is an independent pair, with restarts summarised rather than counted, so the question
  is how much results vary between restarts on one pair compared with between pairs.
- *How:* a new `seed` setting for a fit's random draws (seed 0 reproduces every earlier result).
  On each pair, the selected matching (the true one on every pair) refitted with seeds 1 and 2,
  512 episodes, 12 iterations, scored on the same 300 held-out episodes as the full runs. Seed 0
  is the full run's own result (entries 49-51). Running this on pair 2 measures the method; it
  doesn't change it.
- *Result:* (frames exactly right, allowing for the agent swap)

  | Pair | Seed 0 | Seed 1 | Seed 2 | Mean | Standard deviation |
  |---|---|---|---|---|---|
  | 0 | 64.6% | 69.1% | 70.8% | 68.2% | 3.2 |
  | 1 | 43.8% | 41.6% | 39.8% | 41.7% | 2.0 |
  | 2 | 70.1% | 67.2% | 73.0% | 70.1% | 2.9 |

  Pooled spread within a pair: **2.7 points**. Spread of the pair means: **15.8 points**, around a
  mean of 60.0%. (Entry 52's flipped-agent run on pair 1, 31.6%, is the one outlier; it also
  changes the random draws.)
- *Meaning:*
  - **Almost all the variation comes from the pair** (its corpus and its two trained models), not
    from the aligner's randomness. That matches the proposal's design: the power analysis must
    count independent pairs, and extra restarts add little.
  - Pair 1 is consistently harder (about 42%), because its language model stores object
    positions in a form the straight-line summary model can't read (entry 42). How often a pair's
    language model does that is itself something the main phase should measure.
  - With this, both parts of the pilot gate are met: alignment beats the co-occurrence control
    (entry 53), and variance estimates are in hand.

### Session 5: reporting

**55. Correction: one accuracy measure for every result.**
- *Why:* while exporting the pilot results for the website, the method's numbers in entries 49-54
  didn't match the score files exactly (pair 1: 43.8% in the log, 44.0% in the file).
- *How:* compared the two measures the score files hold.
  - *Up to symmetry* (the proposal's measure, "accuracy modulo the symmetry group"): a decoded
    frame counts as right if it equals the true frame under any of the world's symmetries, here
    either order of the two agents.
  - *Under one labelling:* right only under the single agent order that agrees best overall.
  - The two differ only on frames where both agents are in the same state.
  - Entries 49-51, 53 and 54 reported the method under one labelling, but entry 53's two controls
    up to symmetry, so its table mixed the two. Everything was rescored up to symmetry, including
    restarts 1 and 2 (from their saved predictions), with
    `scripts/export_site_data.py`.
- *Result:* (frames exactly right, up to symmetry)

  | | Pair 0 | Pair 1 | Pair 2 | Mean |
  |---|---|---|---|---|
  | Method | 64.7% | 44.0% | 70.1% | 59.6% |
  | Lookup table in place of the frame model | 61.9% | 32.9% | 63.0% | 52.6% |
  | Co-occurrence control | 0.8% | 1.0% | 0.6% | 0.8% |
  | Restarts (seeds 0, 1, 2) | 64.7, 69.2, 70.9 | 44.0, 41.8, 40.1 | 70.1, 67.3, 73.1 | |

  Spread within a pair: 2.7 points; between pairs: 15.8 points, around a mean of 60.1%.
- *Meaning:*
  - **Nothing changes in substance.** The method's figures rise by at most 0.2 points, the
    controls are as reported, and the variance figures are the same to one decimal place.
  - From here on, every result is reported up to symmetry, as the proposal defines it, unless an
    entry says otherwise. The per-labelling figures stay in the score files (`by_symmetry`),
    since they show which way each pair swapped the agents.

### Session 6: grid video (phase 3, steps 1 and 2)

The proposal's main observation condition is grid video, not state vectors. The pilot used only
state vectors. Before anything is pre-registered, this session asks whether decipherment works
when the watcher sees pictures, and replaces the two hand-written steps the pilot left open: the
step that sorts frame values and words into kinds, and the search that fixed the agents' order.

**56. A renderer for grid video.**
- *Why:* the main condition needs a picture-like observation stream, and the pilot had none.
- *How:* `grounding.render.grid.frames`. Each frame is a small image, 2 rows by 15 columns of
  cells, with 8 yes/no channels per cell (240 values).
  - Each place is a patch of 2 by 3 cells; patches sit side by side with a one-cell gap. Which
    patch shows which place is shuffled per pair.
  - Each agent and object is one cell in its own channel. An agent has two channels (normal and
    hungry), the glass two (intact and broken). Which channel draws which entity is shuffled per
    pair, so the two agents look alike apart from a randomly assigned colour.
  - Where in its patch a thing is drawn is random: a free cell is chosen when it arrives and kept
    while it stays. A carried object is drawn in its carrier's cell and moves with it; dropped,
    it lands in a random free cell. An eaten apple disappears.
  - Nothing marks who acted, so a wait changes nothing on screen. (State vectors had "acted"
    flags.)
  - The grid's size is public, as an image's size is; what each channel and patch means is in
    the pair's hidden key.
  - Grid pair 0 (`runs/pilot-grid/pair-000`, config `configs/pilot-grid.yaml`) uses root seed 0.
    Its language stream is byte-for-byte pilot pair 0's, and its frames show the same episodes as
    pilot pair 0's state vectors, only drawn differently. So pair 0's trained language model was
    reused, and grid and state vectors can be compared on the same episodes.
- *Result:* tests confirm that every frame decodes back to its true state, that nothing is drawn
  outside the patches, that only a carrier and what it carries ever share a cell, and that
  agents get their channels symmetrically.
- *Meaning:* the renderer is ready; the open question is whether the alignment can read it.

**57. A frame model for grid video, and a faster exact sampler.**
- *Why:* the frame model's whole-frame head predicts a frame value by value. A grid frame has 240
  values against the state vector's 45, so the head needs more width, and sampling (which
  recomputed the whole head once per value) would be about 90 times slower than in the pilot.
- *How:*
  - Head width 1,024 hidden units instead of 256 (about four per value, as in the pilot);
    otherwise the pilot's frame-model settings (`configs/pilot-grid-models.yaml`).
  - Sampling now computes each hidden unit once, at the step where all the values it depends on
    are known. This gives exactly the same samples (checked against the old way at three sizes),
    at roughly the cost of one pass through the head instead of one pass per value.
- *Result:* trained in 7.8 minutes, 20,000 steps. Held-out loss 0.0154 per value, about 3.7 nats
  per frame (the pilot's state-vector model: about 2.1). The difference is about what the random
  choice of cell adds: each arrival picks one of about five free cells, about 1.6 nats.
- *Meaning:* the frame model learns the grid world about as well as the state-vector one; the
  extra loss is the renderer's own randomness, which no model can predict.

**58. Finding the frames' structure from the frames alone.**
- *Why:* decipherment needs to know which frame values belong to which thing and which place. In
  the pilot this was a hand-written parser that knew the state vector's meaning (acted flags,
  hunger flags, held, broken, gone). That can't read pictures, and the user asked for it to be
  replaced by a general procedure.
- *How:* `grounding.align.units`, using only the frame stream's statistics and the fact that a
  frame is a grid of cells with channels:
  - *Places* are the separate patches of the image where anything ever appears (connected groups
    of cells that are ever drawn on).
  - *Things* are groups of values of which at most one is on at a time, and between which a
    thing's presence hands over from frame to frame. They are built by merging groups, most
    hand-overs first, while no two values in a group are ever on together.
  - *Appearances* are the channels a thing is drawn in.
  - Each frame is then read as *units*: a thing at a place in an appearance; two things in one
    cell (one carries the other); a thing gone.
- *Result:* on grid pair 0, with no knowledge of what any channel means, it finds 4 places and 5
  things, each exactly one entity's channels (checked against the hidden key afterwards), and 39
  units. On held-out frames, every "at" unit count and "in one cell" unit matches the true state
  (tests).
- *Meaning:* for grid video, the frame side no longer needs a hand-written parser.

**59. Sorting words into kinds from the language alone.**
- *Why:* the pilot's word types (agent, object, place, verb) came from rules about the pilot
  grammar's word endings. The user asked for this to be replaced too.
- *How:* `grounding.align.words`. Each token is described by the distributions of the token just
  before it and just after it. Tokens are clustered by how alike those are (Jensen-Shannon
  divergence, average linkage), and the clustering is cut at its largest jump in merge distance.
  Groups of two or more tokens are the *classes*. Which classes name the frames' things and
  places is decided by size alone: every class whose size equals the number of places, and every
  union of classes whose size equals the number of things, is a candidate *reading*.
- *Result:* on all three pilot pairs the cut falls at 0.063 (next merge: 0.25), and the classes
  are the two agents, the three objects, the four places, and three groups of other words. The
  places are unambiguous; the things have 6 candidate readings, of which one is right. Ranking
  every matching under every reading by the raw frequency profiles (as the pilot's shortlist did)
  puts the right reading far ahead on grid pair 0: best mismatch 0.021, against 0.345 for the
  next reading.
- *Meaning:* the words' kinds can be found without grammar rules, and the search over readings
  costs nothing extra, because the frequency profiles already rule out the wrong readings.

**60. Decipherment on grid video runs end to end, with a low ceiling.**
- *Why:* to check the new pipeline works on grid video before spending GPU time on a search, and
  to measure the best it can do.
- *How:*
  - A smoke run (64 episodes, 4 iterations, the top 2 matchings by raw counts) to check the
    whole pipeline, including scoring through the pair's hidden drawing key.
  - Then the pilot's standard diagnostic: fit under the *true* matching (read from the hidden key,
    so this measures the ceiling, not a result), at small scale (128 episodes, 6 iterations) and at
    the pilot's full scale (512 episodes, 12 iterations). Script: `runs/queue/grid_truth.py`.
  - A gate (`runs/queue/gate_grid.py`) ran the full search only if the full-scale ceiling decoded
    at least 20% of frames exactly right.
- *Result:* (frames exactly right, up to symmetry; then each part of the state)

  | Run | Exact | Agent places | Hunger | Glass | Stone | Apple |
  |---|---|---|---|---|---|---|
  | Smoke, top 2 by counts (both wrong matchings) | 0.1% | 23-29% | 62-67% | 12-14% | 22-34% | 22-23% |
  | True matching, 128 episodes, 6 iterations | 7.2% | 60-63% | 67-72% | 85% | 51% | 70% |
  | True matching, 512 episodes, 12 iterations | 14.1% | 85-87% | 70% | 82% | 65% | 75% |
  | Pilot pair 0, state vectors, true matching (entry 42) | 78.7% | | | | | |

  The full-scale fit took 31 minutes. The gate held the search back.
- *Meaning:*
  - The pipeline works on grid video: it reads agents' places well (85-87%, against 25% by
    chance) with nothing hand-written on the frame side.
  - But whole frames are rarely right, because every part has to be right at once and hunger and
    the objects are weak. Searching for the matching would be wasted GPU time until the ceiling is
    higher.

**61. On grid video, where a thing is and how it looks were tied into one fact.**
- *Why:* to find out whether the weakness is in the frame model or in how decipherment reads
  frames.
- *How:* sampled next frames from the grid frame model on 8,030 held-out steps and compared the
  rate at which each unit switches on with the real data.
- *Result:* the rates match closely for every unit (for example, an agent becoming hungry at a
  place: 0.055 real, 0.057 sampled). 94% of sampled frames are valid; the rest draw some thing
  twice.
- *Meaning:* the frame model is fine. The problem is in the units of entry 58: "agent A at the
  bush, drawn hungry" was one unit, so it switched on both when A *became hungry* at the bush and
  when A *walked to* the bush already hungry. One word table had to explain both "Tofem is hungry"
  and "Tofem goes to the bush". (In the state vectors, hunger was a separate value with its own
  table.)
- *Change:* units now keep where a thing is apart from how it looks: `("at", thing, place)` and
  `("looks", thing, channel)`, still found from the frames alone. The full-scale ceiling run was
  repeated with this change.
- *Result of the change:* worse, not better: 7.3% of frames exactly right (against 14.1%); agent
  places 71-78%, hunger 69-71%, glass 53%, stone 68%, apple 77%.
- *Meaning of the change:* hunger didn't improve at all, so mixing "became hungry" with "walked
  while hungry" was not the main cause. Something else is holding hunger back (entry 62).

**62. The grid frame model's own rollouts drift into impossible frames.**
- *Why:* in the pilot, the language model's summaries were what made hunger readable (entry 33),
  and grid decipherment uses them in the same way, so something specific to grid video must be
  getting in the way.
- *How:* rolled the grid frame model forward for 40 steps from 2,000 real start frames, feeding
  it its own samples (as decipherment's particles do), and counted frames that break the
  structure found in entry 58: a thing drawn twice, or a thing that is always on screen missing.
- *Result:*

  | Step | Valid at this step | Valid at every step so far |
  |---|---|---|
  | 1 | 92% | 92% |
  | 5 | 83% | 70% |
  | 10 | 76% | 51% |
  | 20 | 75% | 30% |
  | 40 | 71% | 14% |

  The pilot's state-vector frame model gave 99.6% valid samples (entry 31).
- *Meaning:* by the middle of an episode most particles are following impossible histories, in
  which an agent may be drawn in two places or in both colours. Decipherment then reads hunger and
  places from nonsense. This explains why every part of the state is weaker than in the pilot.
- *Change:*
  - Decipherment can now give no weight to candidate frames that break the frame stream's own
    structure (option `valid_only`): nothing drawn outside the things, each thing drawn at most
    once, and a thing that is on screen in every training frame drawn exactly once. This uses
    only what entry 58 found from the frames; a particle with no valid candidate simply stays.
  - Whether units keep place and appearance apart is now an option (`grid_units`: split or
    joint), so the two designs can be compared fairly once impossible frames are excluded.
  - Both designs were then compared under the true matching at medium scale (256 episodes, 8
    iterations), with impossible candidates excluded.
- *Result of the comparison:* joint units 9.0% of frames exactly right (agent places 71%, hunger
  71-72%, glass 78%, stone 47%, apple 77%); split units 5.9%. Hunger stayed at about 70% in both.
- *Meaning:* excluding impossible frames helps a little, and joint units are the better design,
  but hunger is still barely read. Agents are hungry 59% of the time, so 70% is close to always
  guessing "hungry".

**63. On grid video the summaries hold hunger, but the fit never learns to read it.**
- *Why:* to find out which part of decipherment fails to read hunger on grid video.
- *How:* three diagnostics, all under the true matching (they use the hidden truth, so they are
  diagnostics only):
  1. *The summaries' evidence.* Each sentence's true state was drawn as a grid frame, the summary
     Gaussian was fitted on true units (512 episodes), and on 300 held-out episodes the true frame
     was compared with the same frame with one agent's hunger flipped, or one agent moved
     (`runs/queue/grid_oracle.py`).
  2. *The particle filter with that oracle Gaussian fixed* and the word tables at their starting
     values, decoding the held-out episodes.
  3. *The fit, traced:* the true-matching fit (joint units, impossible candidates excluded, 256
     episodes, 12 iterations), scored on 64 held-out episodes after every iteration
     (`runs/queue/grid_truth.py trace=1`).
  4. *Words alone with oracle tables:* word tables fitted on the true paths, then decoding from
     words alone (`runs/queue/grid_oracle_words.py`).
- *Result:*
  1. The summaries prefer the true frame over the hunger-flipped one 97% of the time, and over
     the moved one 98%.
  2. With the oracle Gaussian the particle filter decodes **64% of frames exactly right**, with
     agent places 96-97%, hunger 92-94%, glass 89%, stone 95%, apple 93%.
  3. The traced fit: words alone (iterations 1-4) leave agent places at 35-37%; once the
     summaries are used (iteration 5 on), places climb steadily to 0.91 by iteration 12, the
     glass to 0.87 and the apple to 0.84, but hunger stays at 0.72-0.74 throughout and the stone
     at 0.30-0.49. 15.9% of frames exactly right at iteration 12.
  4. Words alone with oracle tables: hunger 67-72%, agent places 60-67%, under either design.
- *Meaning:*
  - The language model's summaries hold hunger, and the particle filter and the grid frame model
    can follow it: with the right summary Gaussian the grid pipeline comes close to the pilot.
  - What fails is learning that Gaussian. It is fitted to the fit's own sampled paths. Words alone
    hardly anchor hunger, so the early paths get hunger at random, the Gaussian learns that
    hunger doesn't matter, and nothing ever corrects it. Places escape this trap because words
    anchor them well enough to start.
  - In the pilot the same bootstrapping did reach hunger (entry 33). The likely difference is the
    pilot's "acted" flags, which tied each step to one agent; grid video has none, so "Tofem is
    hungry" can't be pinned to the right agent's change as easily. [TBD: test.]

**64. The first full search on grid video, with no truth used.**
- *Why:* the question for this step is whether the general search, with the frame structure and
  the word kinds found from each stream alone and the agents' order left free, picks the true
  matching on grid video, even while the fit's accuracy is limited by hunger (entry 63).
- *How:* `runs/pilot-grid/configs/search.yaml`: 17,280 candidate matchings (6 readings of the word
  classes, every order of the words). The 24 closest by raw counts were each fitted briefly (128
  episodes, 4 iterations) and scored on 128 selection episodes; the best 4 were fitted in full
  (512 episodes, 12 iterations) and the one that best explains the selection episodes was kept.
  Joint units, impossible candidates excluded. Fewer finalists than the pilot's 6, to keep the
  night within its GPU budget.
- *Result:*
  - Screening took 72 minutes (about 3 minutes per matching). By raw counts the true matching was
    8th and its agent-swapped twin 7th; their counts are identical, as every agent-swapped pair's
    are. After screening they were 6th (14.997) and 10th (14.930), so neither reached the 4
    finalists. The screening winner (15.212) had all four places right but the stone and apple
    words swapped. Of the top 6 after screening, 4 had all four places right. Full table:
    `runs/pilot-grid/pair-000/align/screening-report.json`.
  - Agent-swapped twins, which should tie, differed by up to 0.17 in screening score, about as
    much as right and wrong matchings did. So a short fit is too noisy to choose on.
  - The search was stopped after screening, since its 4 finalists were all wrong matchings and
    fitting them would only have chosen among them. The GPU time went to entry 65 instead.
- *Meaning:*
  - The general search runs end to end on grid video with nothing hand-written, and it leaves the
    agents' order free (twins are always screened together).
  - Screening finds the places but not the objects, and its noise is as large as the differences
    it has to detect. With the fit's accuracy limited by hunger (entry 63), a short fit sees too
    little to separate matchings.

**65. On grid video, a wrong matching fits better than the truth.**
- *Why:* the proposal's objective diagnostic (design step 7). If the full-scale selection score
  prefers the true matching over the screening winners, the search only needs a better screen; if
  not, the objective itself can't yet tell matchings apart on grid video.
- *How:* full fits (512 episodes, 12 iterations, exactly as the search's finalists) of the true
  matching, its agent-swapped twin, and the two screening winners, each scored on the 128
  selection episodes (`runs/queue/grid_objective.py`). The truth is used only to choose which
  matchings to fit.
  - Run as one process, the second fit ran about 3.5 times slower: the GPU's memory had filled to
    7,976 of 8,192 MiB, and Windows was likely moving memory out to the computer's main memory.
    It was stopped and each remaining fit run in its own process. The second screening winner
    was dropped to stay within the night's GPU budget. The search now releases GPU memory after
    every fit (`_free_gpu` in `align/decipher.py`); that hasn't yet been tested at full scale.
- *Result:*

  | Matching | Selection score (seeds) | Frames exactly right | Agent places | Hunger | Objects |
  |---|---|---|---|---|---|
  | True (rank 8) | 16.995 (17.040, 16.950) | 14.6% | 87-90% | 70% | glass 87%, stone 52%, apple 79% |
  | Screening winner (rank 16): places right, all three object words wrong | **17.103** (17.099, 17.106) | 0.5% | 86-88% | 68-70% | 14-22% |
  | True, agents swapped (rank 7) | 17.021 (17.059, 16.982) | 15.9% | 26% (read as swapped, as expected) | 64-65% | glass 74%, stone 43%, apple 78% |

- *Meaning:*
  - **The objective prefers a wrong matching.** The screening winner beats the truth by 0.11,
    more than the seeds' spread (under 0.1), while getting every object wrong. Getting the objects
    wrong costs the fit almost nothing, because what it has learned to use is the agents' places;
    hunger and the objects carry little weight in it (entry 63).
  - In the proposal's terms (design step 7), the found mapping scores better than the truth, so
    this error counts as "the models encode the world differently", but here the difference is
    in the aligner's own learned model, not in the trained models. The method has to be fixed
    before any search on grid video can succeed. A better screen alone would not be enough.
  - **The true matching and its agent-swapped twin tie** (16.995 and 17.021, a gap of 0.026,
    well inside the seeds' spread of about 0.09), as the agent symmetry predicts. With the
    agents' order left free, the tie is measured rather than decided by numbering: the first such
    measurement on grid video.

**66. Marking which agent acted doesn't help the fit learn hunger.**
- *Why:* entry 63's leading explanation: the pilot's state vectors marked who acted on every
  step, which tied each step's changes to one agent; grid video marks nothing. You allowed
  radical changes to the renderer, so this tests the explanation directly.
- *How:* a renderer option (`grid_actor`) adds one channel, shared by both agents so the symmetry
  survives, that marks the cell of the agent that acted. Pair `runs/pilot-grid-actor/pair-000`
  (root seed 0: the same language stream and episodes). The general procedure of entry 58 finds
  the mark as a sixth thing, with units such as "the mark shares a cell with agent B", meaning B
  acted, and no word names it. Then the traced true-matching fit of entry 63 (joint units,
  impossible candidates excluded, 256 episodes, 12 iterations).
- *Result:*
  - The marked frame model trained in 7.8 minutes (held-out loss 0.0146 per value), but its own
    rollouts drift into impossible frames even faster: 84% valid at the first step, 56% at step 10,
    and only 1% of 40-step rollouts valid throughout (unmarked: 14%).
  - The traced fit: at iteration 12, 14.8% of frames exactly right (unmarked, entry 63: 15.9%),
    agent places 78-82%, **hunger 72-75%** (unmarked: 73-74%), glass 76%, stone 64%, apple 78%.
- *Meaning:*
  - **Marking who acted doesn't help the fit learn hunger.** Entry 63's leading explanation is
    ruled out: the missing actor mark is not why grid video fails where state vectors worked.
  - This fits the decision taken this morning: grid video stays unmarked, and the fix has to come
    from inside the method, from how the fit learns its summary model. The option `grid_actor`
    stays in the code as a diagnostic only.

### Session 7: learning hunger, and state vectors with nothing hand-written

**67. A statistics-only parser for state vectors.**
- *Why:* decided this morning: state vectors, the comparison condition, should have their facts
  sorted into kinds from the frame stream's statistics alone, replacing the pilot's hand-written
  parser. A shuffled list of facts has no cells or patches, so places have to come from which
  values change together.
- *How:* `find_units_vector` in `align/units.py`:
  - *Things* as for grid video (entry 58).
  - A thing's *actor value* is a value outside it that is on whenever the thing changes. Two
    things have one: they are the agents, and their actor values are the "acted" flags.
  - *Places:* when a value of another thing changes while an agent acted, and that agent is then
    always in the same one of its values (at least 98% of the time), the two values are linked.
    Places are the connected groups of linked values that include agents' values.
  - Units: a thing at a place, with an aspect separating values at the same place (0 if ever
    seen at an episode start: intact and broken glass); a value that changes only when an agent
    acts and isn't at a place is "held by" that agent; values outside every thing that change only
    when an agent acts are that agent's flags (hungry, acted); anything else is "gone".
  - Decipherment uses it with `structure: statistics`, through the same units as grid video.
- *Result:*
  - On all three pilot pairs it finds 5 named things, 4 places and 35 units: every agent and
    object position, held by each agent, the apple gone, and each agent's hunger and acted flags.
    Every real frame passes the validity check. Tests check the units against the hidden truth
    on a small world.
  - Ranking matchings by raw counts needed one fix. "A thing changes" now means its place changes
    or it disappears. Three definitions were compared on the three state-vector pairs and grid
    pair 0; this one ranks the right reading of the word classes first on all four, with the
    largest margins (0.17-0.32). Counting every frame value of a thing put a wrong reading first
    on grid video; counting flags too did so on state vectors.
  - The full decipherment of pilot pair 0 with nothing hand-written: [TBD].
- *Meaning:* [TBD]

**68. Stronger summary evidence doesn't help the fit learn hunger.**
- *Why:* during fitting, the summaries' evidence is softened (raised to the power 0.2, entry 33)
  so that it can't swamp the words. That might also leave too little pull for the fit to ever
  start reading hunger.
- *How:* the traced true-matching fit of entry 63 (grid pair 0, joint units, impossible
  candidates excluded, 256 episodes, 12 iterations) at power 0.5 and at power 1.0.
- *Result:*

  | Power | Exact (iteration 12) | Agent places | Hunger | Glass | Stone | Apple |
  |---|---|---|---|---|---|---|
  | 0.2 (entry 63) | 15.9% | 89-91% | 73-74% | 87% | 49% | 84% |
  | 0.5 | 17.0% (best 19.2% at 9) | 88-97% | 72-75% | 73% | 79% | 73% |
  | 1.0 | crashed at iteration 9 (invalid particle weights) | 80-82% at 8 | 71-75% at 8 | 72% | 48% | 71% |

- *Meaning:* more summary evidence speeds up the places and helps the stone, but hunger stays at
  about 0.72 at every power, and full power is numerically unstable, as in the pilot. The
  softening isn't what holds hunger back.

**69. Why the fit can't learn hunger ("no effect" is a trap), and a way out.**
- *Why:* entries 63 and 68 showed the fit never learns to read hunger, at any summary strength.
- *How:*
  1. *Where hunger is lost*, under oracle word tables (`runs/queue/grid_oracle_words2.py`):
     hunger accuracy by sentence kind, and whether the decoded frames lean one way.
  2. *Learning the summary model from distributions alone* (`align/latent.py`): every summary
     is treated as coming from one combination of units, weighted by how often the frame stream
     shows that combination (24,157 distinct ones on grid pair 0), and fitted by
     expectation-maximisation with no pairing. First from a random start, then on synthetic data
     with a known answer, to see whether it can learn a fact whose effect starts at zero.
  3. *Split moves:* for a fact with near-zero effect, restart its effect along each of the main
     directions of the left-over noise (both signs), refit, and keep the restart if the fit
     improves. The bias is moved back by the fact's average contribution, which it had absorbed.
- *Result:*
  1. Under oracle word tables, hunger is right 64% of the time during ordinary event sentences,
     70% at effect sentences, and 67% where it hasn't just changed. The decoded frames show agents
     hungry 61-64% of the time, against 55-56% in truth.
  2. From a random start the distribution-only fit reads nothing: hunger and places at chance
     (50-55% in a two-way test). On synthetic data, a fact whose effect starts at zero *stays* at
     zero, even when nudged: the noise term absorbs its variation, so every small step makes the
     fit worse first. "No effect" is a local optimum, not a balance point.
  3. Split moves escape it: on five synthetic data sets, the missing fact's effect is recovered
     to within 0.005-0.014 (effects 0.6-1.8). Without the bias correction the effect came back
     with its sign reversed.
  4. A bug found on the way: the particle filter's floor on weights was 1e-300, which rounds to
     zero in 32-bit numbers, so a sharp summary model could leave a particle with no option of
     nonzero weight and crash the run. This likely caused entry 68's crash at full power. The
     floor is now the smallest positive 32-bit number.
- *Meaning:*
  - This is the likely mechanism behind entry 63. The fit's summary model starts with no effect
    for hunger, and no amount of evidence moves it, because the starting point is a local optimum
    of the fit itself. Places escape only because the words give them a start.
  - Split moves are a general way out, and the summary model needs no pairing to use them: it
    maps frame units to summaries, whichever words name the things. They need hunger to be a fact
    of its own, as in split units ("A drawn in the hungry colour"). In joint units it is hidden in
    the difference between two place units.
  - Next: split moves inside the true-matching fit on grid video (`split_at`), then, if hunger
    becomes readable, the full search. [TBD]

**70. Learning the summary model from the summaries' own distribution reads one agent's hunger.**
- *Why:* to try entry 69's way out on the real problem. With split units, each agent's appearance
  (normal or hungry colour) is a fact of its own.
- *How:* `runs/queue/grid_split_refine.py`, under the true matching (diagnostic): the particle fit
  of grid pair 0 (split units, power 0.5, impossible candidates excluded, 256 episodes, 12
  iterations); then the distribution-only fit (entry 69) started from its summary model, with
  split moves on the 4 facts of smallest effect, on 20,000 summaries and the 23,735 unit
  combinations covering 99.9% of training frames. Each agent's hunger was then tested by how
  often the summaries prefer the true frame over the same frame with that agent's hunger flipped
  (the oracle model of entry 63: 97%).
- *Result:*

  | | Exact frames | Agent places | Hunger (decoded) | True beats flipped: agent A / agent B |
  |---|---|---|---|---|
  | Particle fit, run 2 | 21.4% | 97% / 97% | 74% / 68% | 72% / 66% |
  | After the distribution-only fit, run 2 (no fact counted as weak) | crashed | | | **95%** / 68% |
  | Particle fit, run 3 | 18.4% | 97% / 97% | 66% / 69% | 66% / 66% |
  | After split moves on the 4 weakest facts, run 3 | crashed | | | 60% / **97%** |

  - In run 3 the 4 weakest facts were exactly the two agents' normal and hungry appearances.
    Split moves on two of them improved the fit (110.70 to 111.16 to 112.19); the other two gave
    no gain.
  - The particle fit's summary model had learned a partly shared hunger direction: the two
    agents' hunger effects point fairly similarly (cosine 0.55), a "someone is hungry" signal.
    After the split moves the directions separate (cosine -0.24), but only one of them comes out
    right.
  - Decoding with the refined model crashed in both runs. Found with synchronous error
    reporting: entry 69's float floor had been added to every option, so a particle at the end
    of the frame model's context, or with only excluded candidates, could still move, and read
    past its history. The floor now goes on "stay" only, which is always allowed. (Two other
    places that could divide 0 by 0 were also given safe floors.)
- *Meaning:*
  - **For the first time on grid video, the fit's summary model reads an agent's hunger about as
    well as the oracle (95% and 97%), learned from the summaries and the frame stream's statistics
    alone,** with no pairing.
  - But only one agent at a time: the two agents' hunger compete for the same direction in the
    summaries. Next: alternate split moves with a few particle-fit iterations, whose paths then
    follow the hunger already learned, and repeat (queued after the state-vector run).

**71. With nothing hand-written, state vectors give the right words but the places' mirror image.**
- *Why:* entry 67's full test. Can the pilot's state-vector pair 0 be deciphered with the facts
  sorted by the statistics-only parser and the words sorted by the language stream's own classes?
- *How:* `configs/pilot-decipher-stats.yaml`: the pilot's settings, with `structure: statistics`,
  word kinds by classes, joint units, impossible candidates excluded, and the 8 readings closest
  by raw counts fitted in full (12 iterations, 512 episodes). Agent order is free.
- *Result:*
  - All 8 finalists name the 5 things with the right words. They differ only in the order of two
    pairs of place words, and in which agent is which.
  - The selection score picked a reading with both place pairs reversed: 18.99, against 18.83
    and 18.75 for the two readings with the right places. Those two decode 60% and 50% of frames
    exactly (modulo the agent swap); the chosen one 0%.
  - The same reversed reading was the runner-up in the earlier rule-based runs of this pair
    (18.81 against the right reading's 19.29). The gap here (0.16) is smaller than the spread
    between two scoring seeds of one finalist (up to 0.27).
- *Meaning:*
  - The statistics-only parser and the word classes give the decipherment everything it needs:
    the right reading is among the finalists and, once found, decodes as well as with the
    hand-written parser.
  - What fails is the final choice, between two readings that the world nearly can't tell apart.
    The rule-based fit separated them by 0.48, this one fits all readings a little differently.
    Re-scoring the saved finalists with 6 seeds, and on 384 extra unused training episodes with
    4 seeds (`runs/queue/rescore_finalists.py`), shows it isn't noise: on the extra episodes the
    reversed reading scores 19.13 (spread 0.03), the right readings 18.97 and 19.00.
  - The reversed reading is the places' **mirror image**. The pilot's places lie on a path,
    river - bush - cave - hill, and the path is symmetric end to end; only feeding at the bush
    breaks it (`sim/worlds.py`). So this is a near-symmetry of the world, and the only evidence
    against it comes through hunger, which the chosen fit reads at 62%. It is the same weakness
    as on grid video (entries 63-72), showing up as a wrong choice instead of wrong frames.
  - Next: refit both readings with split moves inside the fit, and compare (entry 75).

**72. Restarting an agent's two appearances together reads both agents' hunger.**
- *Why:* in entry 70, split moves taught the summary model one agent's hunger but not the
  other's. A single fact restarted on its own may not be enough: an agent's two appearances
  (normal and hungry colour) exclude each other, so only the difference between their effects
  means anything, and moving one alone barely changes the fit.
- *How:* `runs/queue/joint_split_cpu.py`, on the CPU: the distribution-only fit (5,000 summaries,
  the 21,217 unit combinations covering 98% of training frames) from run 3's particle summary
  model; then each pair of the 4 weakest facts restarted together along two different main
  directions of the left-over noise, every order and sign, keeping any restart that improves the
  fit. Evaluated with entry 70's flip test (true frame against the same frame with one agent's
  hunger flipped; oracle 97%).
- *Result:*
  - Start: agent A 94.6%, agent B 65.4%. The 4 weakest facts were agent B's two appearances and
    two object positions.
  - Restarting B's two appearances together improved the fit (112.25 to 112.81 to 112.82) and
    brought **both agents to 94-95%**: A 94.4%, B 95.1%. None of the other 5 pairs of weak
    facts improved the fit further.
- *Meaning:*
  - The first summary model, learned without pairing, that reads both agents' hunger about as
    well as the oracle.
  - The move needs nothing from the truth: which facts are one thing's appearances comes from
    the frame structure. It is now an option of the split moves (`together`, and
    `split_together` in decipherment), and a test checks it on synthetic data.
  - Next: the same inside the true-matching particle fit on grid video, alternating with particle
    iterations (`split-refine-5`), to see whether decoded hunger follows. [TBD]

**73. Alternating split moves with particle iterations still reads only one agent's hunger.**
- *Why:* entry 70's plan: after split moves, a few particle-fit iterations whose paths follow the
  hunger already learned, then split moves again.
- *How:* `split-refine-4`: grid pair 0, true matching (diagnostic), split units, power 0.5,
  256 episodes, 12 iterations; then two rounds of split moves (the 4 weakest facts, single moves
  only, up to 2 passes) each followed by 4 particle iterations.
- *Result:*

  | | Exact (modulo swap) | Places A / B | Hunger decoded A / B | True beats flipped A / B |
  |---|---|---|---|---|
  | Particle fit | | | | 66% / 66% |
  | Round 1, split moves | 13.7% | 93% / 95% | 58% / 96% | 57% / 78% |
  | Round 1, + 4 particle iterations | 16.2% | 96% / 96% | 62% / 97% | 60% / 97% |
  | Round 2, split moves | 19.5% | 96% / 95% | 61% / 96% | 60% / 78% |
  | Round 2, + 4 particle iterations | 13.2% | 95% / 95% | 61% / 97% | 61% / 78% |

- *Meaning:*
  - **Decoded hunger follows the summary model**: once it reads agent B's hunger, the decoded
    frames get B's hunger right 96-97% of the time, up from 68%.
  - Alternating doesn't bring in the second agent. Single moves can't place it; entry 72's joint
    move can, and is tried inside this loop next (`split-refine-5`).
  - Exact frames stay low (13-20%) while one agent's hunger and the objects' places (62-80%) are
    still weak.

**74. With split moves inside the fit, exact grid frames rise from 20% to 47%.**
- *Why:* entry 73's loop, with entry 72's joint move available.
- *How:* `split-refine-5`: as entry 73 (true matching, diagnostic), with one thing's two
  appearances also allowed to move together. The joint move is tried only on pairs whose facts
  both stay weak after the single moves.
- *Result:*

  | | Exact (modulo swap) | Places A / B | Hunger decoded A / B | True beats flipped A / B | Glass / stone / apple |
  |---|---|---|---|---|---|
  | Particle fit (12 iterations) | 20.1% | 97% / 97% | 66% / 69% | 65% / 67% | 79% / 77% / 73% |
  | Round 1, split moves | 34.9% | 96% / 96% | 95% / 95% | 95% / 97% | 69% / 74% / 80% |
  | Round 1, + 4 particle iterations | **46.7%** | 97% / 96% | 94% / 96% | 97% / 97% | 79% / 88% / 80% |
  | Round 2, + 4 particle iterations | 44.6% | 96% / 96% | 94% / 97% | 96% / 97% | 77% / 86% / 81% |

  - The oracle summary model of entry 63 gives 64% exact.
  - The joint moves found no gain here. The single moves did it: this time the weakest facts
    included one appearance of each agent, and both were placed (fit 112.04, then 113.02).
  - In entry 73 the single moves placed both appearances of one agent first, and then nothing
    could place the other agent's: fit 112.63. **The likelihood prefers the answer with both
    agents' hunger** (113.03 against 112.63), so the difference is the order the moves were
    tried in, not what the fit can tell.
- *Meaning:*
  - **With nothing paired and grid video unmarked, the fit now reads both agents' hunger as well
    as the oracle, and exact frames more than double, under the true matching.** What is left is
    the objects' places (77-88%).
  - Split moves now take `orders`: the whole procedure in several orders of the weak facts (as
    found, reversed, shuffled), keeping the likeliest fit, which needs nothing from the truth
    (`split_orders` in decipherment). A test checks more orders never fit worse.
  - Next: check that this is robust (`split-refine-6`: three orders, one round), then put split
    moves into the full grid search, where the matching isn't given. [TBD]

**75. Split moves inside the fit let the selection prefer the right places over their mirror.**
- *Why:* entry 71: on state vectors with nothing hand-written, the selection score preferred the
  places' mirror image.
- *How:* `runs/queue/stats_mirror_split.py`: the mirrored reading (finalist 2) and the right one
  (finalist 4) refitted as in entry 71, but with split moves after 8 of the 12 iterations (the 4
  weakest facts, up to 2 passes, one thing's appearances also together). Scored as entry 71's
  re-scoring: 4 seeds on the selection episodes, and 4 on 384 extra unused training episodes.
- *Result:*

  | Reading | Without split moves (extra episodes) | With split moves (extra episodes) | With split moves (selection episodes) |
  |---|---|---|---|
  | Mirrored places (finalist 2) | 19.13 | 18.81 (spread 0.02) | 18.67 |
  | Right places (finalist 4) | 18.97-19.00 | **19.21** (spread 0.01) | **19.06** |

  - The weak facts here were object and agent positions, not hunger, whose effects were already
    large. In each fit only one split move improved the summary-model fit, and by little
    (0.001-0.01).
  - The particle fit dips sharply in the iteration after the split moves (to -7.8 and -3.0), then
    recovers within an iteration, to a higher fit for the right reading and a lower one for the
    mirror.
- *Meaning:*
  - **The right reading now wins by 0.4, about 20 times the spread between seeds.** If this holds
    across all finalists, the statistics-only state-vector decipherment selects correctly.
  - But the mechanism isn't the one expected. The split moves barely changed the summary model,
    so the gain seems to come from the jolt to the fit: refitted from a summary model fitted to
    the summaries' distribution, the paths re-settle. One pair, two readings; it needs the full
    search to count (`stats-split-search`, queued after the grid search). [TBD]
  - A test of selection by the summaries' distribution alone, without words, is entry 77.

**76. Trying split moves in several orders makes learning hunger reliable.**
- *Why:* entries 73 and 74 differed only in the order the weak facts were tried, and one of them
  read only one agent's hunger.
- *How:* `split-refine-6`: as entry 74 (grid pair 0, true matching, diagnostic), with the split
  moves run in three orders of the weak facts (as found, reversed, shuffled), keeping the likeliest
  fit; one round, then 4 particle iterations.
- *Result:*
  - The weak facts were the two agents' four appearances. All three orders ended at nearly the same
    fit (113.20, 113.18, 113.20); the joint move on one agent's two appearances gave the largest
    single gain (112.46 to 113.20) in two of them.
  - Hunger flip test: 68% / 64% before, 96% / 94% after the split moves, 96% / 97% after the 4
    particle iterations. Decoded hunger 93% / 97%, agent places 97% / 96%.
  - Exact frames (modulo swap): 17.4% before, 21.3% after the split, **37.1%** after the 4 particle
    iterations (entry 74: 46.7%). Objects: glass 76%, stone 81%, apple 73% (entry 74: 79%, 88%,
    80%).
- *Meaning:*
  - **Learning both agents' hunger is now reliable**, and with no pairing: every order found it.
  - Exact frames vary by about 10 points between runs (37% and 47%), all of it in the objects'
    places, which are now the main thing between grid video and the oracle's 64%.

**77. The summaries' distribution alone can't break the places' near-symmetry.**
- *Why:* a cheaper, more direct selection score: each finalist's summary model maps frame units to
  summaries, and can be scored on the summaries' distribution against the frame stream's
  statistics, with no words. A mirrored reading should leave a mirrored summary model.
- *How:* `runs/queue/latent_select.py` on entry 71's 8 finalists (pilot pair 0, statistics-only
  parser): each finalist's summary model scored as it was left (one step of entry 69's
  distribution-only fit), and after refitting it from there (60 iterations), on 20,000 summaries
  and the 48,372 unit combinations covering 99.9% of training frames.
- *Result:*

  | Finalists | Places | Score as left | After refitting |
  |---|---|---|---|
  | 4, 5 | right | 112.93, 112.19 | 121.89, 122.40 |
  | 2, 3 | mirrored | 112.01, 112.86 | 122.36, 122.43 |
  | 0, 1, 6, 7 | one pair swapped | 108.18-108.30 | 120.79-122.37 |

- *Meaning:*
  - As left, the score rejects the readings with one pair of places swapped by about 4, but it
    can't separate the right places from the mirror. After refitting, it separates nothing: the
    refit reaches different local optima, and agent-swapped twins differ by up to 1.2.
  - The frame stream's statistics barely break the mirror either: it is a near-symmetry of the
    frames, as H3 expects. What breaks it is feeding at the bush, and the words have to carry
    that. This is why split moves inside the word fit helped (entry 75) and this doesn't.
    Distribution-only selection is dropped.

**78. GPU memory fragmentation, not the method, was slowing the searches.**
- *Why:* tonight's grid search ran some screens 2-3 times slower than others, with nothing else
  using the GPU (after the thermal slowdown was fixed by raising the laptop).
- *How:* Windows' per-process GPU memory counters; then `runs/queue/mem_probe.py`, one particle
  filter pass on grid pair 0 (64 episodes, 79 sentence steps) comparing the memory actually in use
  with the memory PyTorch reserved.
- *Result:*
  - The search held 7.7 GB of the card's 8 GB and **another 6.2 GB spilled into system memory**,
    which the GPU reaches far more slowly.
  - One filter pass used at most 1.35 GB but reserved **13.5 GB**. The filter re-runs the frame
    model on histories one frame longer at every step, and PyTorch's cache can't reuse a freed
    block for a larger request, so it keeps asking for more. With
    `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,roundup_power2_divisions:8` the same
    pass reserved 4.8 GB. The setting changes only where tensors are placed, not what is computed.
  - Running the probes beside the search made it run out of GPU memory and crash at screen 12
    (20:42). It was restarted with the setting (`grid-split-search-2`), and held 3.6-4.5 GB with
    nothing spilled; its first screen took 4 minutes.
  - The restart's first screen scored 45.01 against 44.93 in the crashed run, with the same seeds:
    some GPU operations add in a varying order, so reruns differ slightly. This is within restart
    noise.
- *Meaning:*
  - Long runs now set the allocator option, and nothing else runs on the GPU beside a search.
  - The fragmentation comes from recomputing the whole history at every step, which caching the
    frame model's attention would remove (a speed-up already proposed).

**79. A faster particle filter and split moves, with the same results.**
- *Why:* a grid search takes about 9 hours, and the work ahead needs many of them. Done on a
  separate branch (`feat/perf`, its own worktree) so tonight's running searches were untouched,
  and tested on the CPU only.
- *How:* each change checked against outputs recorded from the current code, on a tiny grid pair
  and on small batches of grid pair 0 and pilot state-vector pair 0: the frame sampler on fixed
  inputs, two particle-filter passes (without and with a summary model), and a short fit with
  split moves. The recorded outputs reproduce themselves exactly, so any difference is the change.
  1. *The frame sampler* builds its per-value plan (which hidden units to compute at each value,
     with their weights) once and keeps it while the weights are unchanged, instead of on every
     call. Building it made the CPU wait for the GPU about 480 times per call on grid video.
  2. *In the filter loop:* each frame's unit view computed once per step (it was computed up to
     four times); word-table look-ups done once per sentence and copied to its particles and
     candidates; constants and checks moved out of the loop.
  3. *Cached context* (`context_cache`, off by default): the frame model's keys and values kept
     for each particle's history, so a step puts only the new frame through the model instead of
     the whole history. Resampling copies the cache with the particles.
  4. *Split moves:* at temperature 1, the responsibilities reuse the log-sum-exp already made for
     the likelihood instead of a second pass over the summaries x views matrix; the likelihood is
     summed on the device and read once.
- *Result:*
  - Changes 1 and 2 give **bit-for-bit identical** outputs on all three cases. All 180 tests pass.
  - Change 3: the cached context matches the full recompute to within 2e-6 (values of typical
    size 0.8). Every path and likelihood in the three cases came out identical, because the
    context is used only to sample 0/1 frames, and differences this small changed no sample.
    CPU time: grid pair 0 21.2 s to 16.3 s, state vectors 9.3 s to 3.8 s.
  - Change 4: 30-35% faster on a synthetic problem (split moves 26.5 s to 18.7 s); coefficients
    agree to within 1.7e-6. A restart is kept only if it gains more than 1e-4, so a trial within
    about 1e-5 of that line could, rarely, be decided differently.
  - GPU timings can't be measured until the queue is empty; changes 1 and 3 are expected to gain
    most there, since they remove waits for the GPU and the history recompute that fragmented its
    memory (entry 78).
- *Meaning:* changes 1 and 2 can go in as they are. Changes 3 and 4 give the same results within
  rounding; 3 is behind a switch so that earlier runs can be reproduced exactly. Next: GPU timings
  and one full fit compared with and without them, before they are used for new results.

**80. The world's dynamics alone can't break the places' near-symmetry.**
- *Why:* a selection check independent of the words: under a mirrored reading, the decoded story
  should put eating at the wrong end of the path, which the frame model (trained on the frame
  stream alone) should find implausible.
- *How:* `runs/queue/dynamics_select.py` on entry 71's 8 finalists (pilot pair 0, statistics-only
  parser, no split moves): each finalist's decoded frames for the 300 held-out episodes, compressed
  to each episode's sequence of distinct frames, scored by the frame model's log-probability per
  transition (about 10,500 transitions per finalist). No truth used.
- *Result:*

  | Finalists | Places | Dynamics per transition |
  |---|---|---|
  | 2, 3 | mirrored | -2.145, -2.144 |
  | 4, 5 | right | -2.172, -2.157 |
  | 0, 1, 6, 7 | one pair swapped | -2.222 to -2.237 |

  The spread across episodes is 0.12-0.23, so the standard error per finalist is about 0.007-0.013.
- *Meaning:*
  - The dynamics reject readings with one pair of places swapped (by about 0.07), but not the
    mirror: they prefer it slightly (by about 0.02, two to three standard errors).
  - The mirror is a near-symmetry of the dynamics too, broken only by feeding at the bush, which
    runs through hunger; these fits read hunger at 62%, so the one asymmetry is lost in hunger
    errors. The words alone (entry 71), the summaries' distribution alone (entry 77) and the
    dynamics alone all fail to break it. Split moves inside the fit did (entry 75), presumably by
    getting hunger right. Dropped as a selection fix.
  - For H3 this is the expected pattern: a near-symmetry is confused by every part of the method
    that can't see what breaks it.

**81. On grid video with no truth, the selection chose the places' mirror over a truth fit that misread the apple.**
- *Why:* the first full grid search with split moves inside every fit, matching not given
  (`configs/pilot-grid-decipher-split.yaml`): 24 matchings screened, 6 finalists fitted in full.
- *How:* grid pair 0, restarted with the allocator setting of entry 78. The true matching was
  screened sixth of 24 and ranked second in screening (45.88, against 45.99 for a reading with
  two object words exchanged and the places wrong). The search was stopped after 2 of its 6
  finalists, because finalist 1 had already outscored the true matching, so the selection could
  no longer choose it; a partial summary was written so the two could be scored.
- *Result:*

  | | Finalist 1: places mirrored, two object words exchanged | Finalist 2: the true matching |
  |---|---|---|
  | Selection score (2 seeds) | **50.32** (50.29, 50.36) | 50.15 (50.17, 50.13) |
  | Exact frames | 0.1% | 17.4% |
  | Agent places | 26%, 26% | 97%, 97% |
  | Hunger, A / B | 64% / 63% | 94% / 96% |
  | Glass / stone / apple | 14% / 7% / 21% | 77% / 85% / 31% |

  The diagnostic runs under the true matching (entries 74 and 76) read the apple at 73-80% and
  decoded 37-47% of frames exactly.
- *Meaning:*
  - **Split moves work inside the real search:** the true matching's fit reads both agents'
    hunger at 94-96%, as in the diagnostics.
  - But that fit settled in a poor optimum for the apple (31%), and lost the selection to a
    reading that is the places' mirror image, the same near-symmetry as on state vectors (entry
    71), with two object words also exchanged.
  - This is the unequal-fits problem the proposal's objective diagnostic now guards against
    (proposed edit 8): the selection compared one fit of each matching, and a fit can land
    anywhere. The proposal already asks for every alignment to be restarted several times; the
    search doesn't yet. Next: refit both matchings from several seeds and compare their best
    fits. [TBD]

**82. The full state-vector search with split moves ran out of GPU memory speed and was stopped after 2 of 8 finalists, neither of them the truth.**
- *Why:* the state-vector counterpart of entry 81 (`configs/pilot-decipher-stats-split.yaml`,
  pilot pair 0): does the selection choose the truth once every finalist is fitted with split
  moves?
- *How:* 8 finalists, each fitted for 12 iterations with split moves after iteration 8. Started
  25 November 23:52 and stopped 26 November 06:24, during finalist 3.
- *Result:*
  - Split moves were far slower than in the diagnostics: the first split phase took about 1.9
    hours and the second about 35 minutes. The third was still running after 3 hours. Normal
    iterations take about 45 seconds.
  - The GPU's own 8 GB memory was full, with about 1 GB spilled into shared system memory. That
    is the likely cause, even with the allocator setting of entry 78.
  - Both finished finalists are wrong matchings. Scored on CPU from their saved frames:

    | | Finalist 1 | Finalist 2 |
    |---|---|---|
    | Selection score (2 seeds) | 17.85 | 17.79 |
    | Exact frames | 2.5% | 2.6% |
    | Agent places | 25%, 27% | 44%, 23% |
    | Hunger, A / B | 63% / 61% | 96% / 87% |
    | Glass / stone / apple | 44% / 35% / 51% | 49% / 55% / 63% |

  - All 8 finalists had almost the same raw-count mismatch (0.4987 to 0.4989), so screening
    barely separated them.
- *Meaning:*
  - No verdict yet on selection with split moves for state vectors. The search would have
    needed about 10 more hours at this pace.
  - The split phase's memory use has to come down before this search is rerun.
  - Stopping it frees the GPU for the restart test that entry 81 calls for, which comes first
    because the grid search failed.

**83. The filter speed-ups, the cached context and the sampler graph reproduce the current code exactly on the GPU; the split-move change didn't and was dropped.**
- *Why:* the rule agreed on 25 November. A speed-up goes into the method only if it gives
  exactly the same numbers on the GPU. A change that differs by rounding goes in only if a full
  fit reproduces within restart noise.
- *How:*
  - The current code was run twice on the GPU (a small grid pair, grid pair 0 and state-vector
    pair 0), and the faster code (branch `feat/perf`) once. Values the current code doesn't
    reproduce from one run to the next were left out of the comparison.
  - The frame sampler's CUDA graph was compared with the ordinary loop: the same seeds, three
    consecutive calls, two trials, at 8,192 rows.
- *Result:*
  - The faster filter matched the current code bit for bit: the frame sampler's output, the
    filter's trajectories and its likelihoods, with and without the summary model.
  - The same was true with the cached context switched on.
  - The only differences were in the summary model after a short fit with split moves (largest
    difference 0.23). They come from the one change to the split-move arithmetic, which rounds
    differently.
  - The sampler graph was identical on every call, and it left the random generator in the same
    place.
  - Times: one filter pass on grid pair 0 took 12.0 seconds, against 17.9 seconds for the
    current code. The cached context made no further difference at this size (12.2 seconds). The
    sampler graph took 64 ms per call against 79 ms. The tiny state-vector case was slightly
    slower (5.6 seconds against 4.4); that is unexplained and still to check.
- *Meaning:*
  - The filter changes, the cached context and the sampler graph go into the method: they change
    the speed, not the numbers.
  - The split-move change was reverted, because it saves little time. It would only be worth a
    full-fit check if the split step ever became the bottleneck for that reason.
  - The restart test (entry 81's next step) was already running on the current code, so the
    faster code is used from the next GPU job on.

**84. Change of direction: a pilot world where the places' mirror is strongly broken, to test the method without the near-symmetry.**
- *Why:*
  - Five attempts (entries 71, 75, 77, 80 and 81) and the start of the restart test have tried to
    make the method choose the right places over their mirror image in the pilot world.
  - The world itself barely tells the two apart: the mirror's violation rate is about 7%, and
    only feeding at the bush breaks it. H3 predicts that such a near-symmetry gets confused. So
    the method failing on it is no evidence that the method fails in general.
  - Refitting the mirror reading from a new seed made it score higher (50.50, against 50.32 in
    entry 81) while it still decoded nothing correctly. A better fit of the wrong reading wins;
    more restarts can't fix that.
  - So the next test separates the two questions. Does the method work when the world clearly
    breaks the mirror? If it does, the pilot world's mirror becomes a measured near-symmetry, the
    kind H3 is about, rather than an obstacle.
- *How:*
  - A new world, `lured` (`sim/worlds.py`): the pilot world, with both agents three times as
    likely to choose moving to the river or the cave. The mirror would put that preference at
    the hill and the bush, so every move breaks it.
  - The mirror's violation rate rises from 0.074 to 0.249 (visit-weighted, episodes of 20-60
    steps). The agent swap stays exact; the symmetry group is unchanged (a new test checks both).
  - Two new conditions, `pilot-lured` (state vectors) and `pilot-grid-lured` (grid video). Pair 0
    of each shares one language stream, which was checked by hash.
  - Each pair has a new language model and new frame models. The searches are unchanged: the
    same configs as entries 81 and 82, on the faster code of entry 83, with the split step's
    memory released first.
  - Mirror restart from seed 1, for the record: fit 50.72, selection 50.50, exact frames 0%,
    agent places 25%, hunger 63%, glass, stone and apple 15%, 5% and 22%.
  - True reading restarted from seed 1: fit 50.20, selection 50.14, exact frames 15%, agent places
    97%, hunger 94% and 96%, glass, stone and apple 78%, 84% and 29%. That is almost exactly entry
    81's first fit (50.15, apple 31%). So the apple misreading is systematic, not an unlucky
    optimum. The restart test was stopped there; restarts don't close a gap of 0.36.
- *Result:* [TBD]
- *Result:* see entry 85.

**85. In the lured world the grid search rejects every wrong reading of the places, but confuses object words that the world barely tells apart; the confusions follow the violation rate.**
- *Why:* entry 84's test. With the places' mirror strongly broken, does the grid search with no
  pairing choose the true reading?
- *How:*
  - Grid pair 0 of `pilot-grid-lured`, searched with the config of entry 81, unchanged, on the
    faster code. 24 matchings were screened and 6 finalists fitted in full; it took 4 hours,
    against about 8 for entry 81.
  - Monitoring only (uses hidden/): `runs/queue/true_rank.py` named each finalist, and each was
    scored from its saved frames.
  - Afterwards all 6 finalists were re-scored (`runs/queue/rescore_words.py`), with and without
    the summary model, on the selection episodes and on 384 unused training episodes.
  - Each confused relabelling's violation rate was computed in both worlds (visit-weighted,
    episodes of 20-60 steps).
- *Result:*

  | Finalist | Reading | Selection | On 384 more | Words only | Exact frames |
  |---|---|---|---|---|---|
  | 1, **selected** | stone and apple words exchanged | **53.56** | 53.45 | -3.896 | 7% (32% allowing the exchange) |
  | 2 | glass and apple words exchanged, agents swapped | 53.48 | 53.32 | -3.898 | 6% (20%) |
  | 5 | glass and apple words exchanged | 53.22 | 53.09 | -3.900 | 5% (16%) |
  | 4 | **the true reading**, agents swapped | 53.13 | 53.33 | -3.918 | **43%** |
  | 6 | places shifted two steps, agents swapped | 52.93 | 52.91 | -3.950 | 0% |
  | 3 | places shifted two steps | 52.70 | 52.83 | -3.928 | 0% |

  | Relabelling | Violation rate, pilot | Violation rate, lured | Outcome in this search |
  |---|---|---|---|
  | stone <-> apple | 0.030 | 0.027 | beat the truth, selected |
  | glass <-> apple | 0.053 | 0.046 | beat the truth |
  | places' mirror | 0.074 | 0.249 | not among the finalists (it was chosen in entry 81's pilot world) |
  | places shifted two steps | 0.325 | 0.345 | scored lowest |

  - The true reading decodes 43% of frames exactly (agent swap allowed), the best yet on grid
    video. Places are 97%, hunger 94-96% and the objects 66-77%.
  - Finalists 2 and 5 are the same reading apart from the exact agent swap, yet their scores
    differ by 0.26. That is a direct measure of fit-to-fit noise. The truth's gap to the selected
    reading is 0.43.
- *Meaning:*
  - **The lure worked.** Every reading with the places wrong scored lowest, by 0.4-0.9. In the
    pilot world the mirror had won.
  - What is left are relabellings of the objects. Their violation rates are 0.03-0.05, below
    even the pilot's mirror (0.07). Glass breaking and an apple being eaten both make an object
    disappear, and what tells them apart runs through hunger, as with the mirror.
  - Words alone don't separate them either (-3.896 against -3.918). So this isn't the summary
    model: on these readings the truth's fit simply explains the words no better.
  - Across the two worlds the pattern matches H3's graded prediction. Relabellings the world
    breaks at a rate of 0.07 or less are confused; those at 0.25 or more are not. The threshold
    lies somewhere between [TBD].
  - For the method this is a partial success. With no pairing and no truth, grid video was
    decoded to the right world up to one barely broken exchange of object words: 32% of frames
    are exact once that exchange is allowed for.
  - One pair can't say whether the truth lost by noise or by bias. It is under two fit-noise
    gaps behind. Across pairs, the proposal counts such confusions like a swap rate.

**86. With nothing hand-written and no hints, the state-vector search selects the true reading in the lured world, and the wrong readings lose in order of how strongly the world breaks them.**
- *Why:* entry 82's search, which stalled in the pilot world, rerun in the lured world (entry 84).
  In the pilot world state vectors got every object word right and failed only on the places'
  mirror (entry 71).
- *How:*
  - `pilot-lured` pair 0, with `configs/pilot-decipher-stats-split.yaml`, unchanged, on the
    faster code with the split step's memory released (entries 82-83). There were 8 finalists;
    it took 4 hours 7 minutes.
  - Each finalist took about 30 minutes, against 1-2 hours in entry 82. GPU memory stayed at
    about 6 GB, with no spill.
  - Monitoring only (uses hidden/): each finalist was scored from its saved frames as it
    finished, and named with the hidden key.
- *Result:*

  | Finalist | Reading | Violation rate (lured) | Selection | Exact frames |
  |---|---|---|---|---|
  | 1, **selected** | **the true reading**, agents swapped | 0 | **20.37** | **68.1%** |
  | 2 | the true reading | 0 | 20.05 | 64.0% |
  | 7 | the places' mirror (river-hill, bush-cave) | 0.249 | 19.75 | 0% |
  | 8 | the same, agents swapped | 0.249 | 19.70 | 0% |
  | 6 | river and hill exchanged, agents swapped | 0.448 | 19.32 | 2.8% |
  | 5 | river and hill exchanged | 0.448 | 19.20 | 2.5% |
  | 3 | bush and cave exchanged | 0.510 | 19.08 | 4.2% |
  | 4 | bush and cave exchanged, agents swapped | 0.510 | 18.88 | 2.8% |

  - Official score (`analysis/score-decipher-stats-split.json`): 68.1% of held-out frames exactly
    right up to the agent swap, against 0.06% for the same frames shuffled.
  - The selected reading has the agents swapped: 6.0% exact as numbered, 68.0% under the swap.
  - Places are read at 95-97% by the unswapped twin, hunger at 97%, and the objects at
    90-92% (72-82% for the selected fit).
  - The two true readings differ only by the exact agent swap and score 0.32 apart. That gap is
    fit-to-fit noise; on grid video it was 0.26 (entry 85).
- *Meaning:*
  - **The first full success with nothing hand-written.** Both streams' structure comes from their
    own statistics, the matching is searched with no pairing, and the selection uses no truth;
    the result is 68% of frames exactly right. The pilot's hand-written method averaged 59.6%
    across three pairs (entry 55), and 64.7% on its pair 0.
  - **The losses are graded.** Each wrong reading's gap to the truth grows with its violation rate:
    0.6 for the mirror (0.25), about 1.1 for river-hill (0.45) and about 1.4 for bush-cave
    (0.51). In the pilot world the mirror's rate was 0.07, and it won (entry 71). Together with
    entry 85, readings the world breaks at 0.07 or less are chosen or tie, and the selection
    margin grows with the violation rate above that. This is H3's graded prediction, seen inside
    one pair's selection (a tuning pair, so not evidence yet).
  - State vectors resolve the objects that grid video confused (entry 85). The lists of facts give
    the fit cleaner evidence about who holds what.
  - **Next: confirmation on fresh pairs.** The method is frozen as it ran here (below), and runs
    once on `pilot-lured` pairs 1 and 2, then on grid pair 1.
- *Frozen method for the confirmation:*
  - Research repo at commit 8ff3381, plus the lured world (`sim/worlds.py`, `runs/conditions.py`).
  - Search code: the `feat/perf` worktree at commit 2a83c36 plus its uncommitted changes, saved
    as `runs/pilot-lured/method-frozen.patch` (sha256 starting 128f9ba35a05b21c).
  - Config: `configs/pilot-decipher-stats-split.yaml` for state vectors, and
    `configs/pilot-grid-decipher-split.yaml` for grid video. No setting changes.

**87. Confirmation on fresh pairs: the frozen method selects the true reading on lured pair 1 (state vectors, 62.9% of frames exactly right).**
- *Why:* entry 86's result came from a tuning pair. This is the test, with the method frozen as
  entry 86 records, run once on pairs never used before. Nothing was looked at against the truth
  until each search had made its choice.
- *How:* `pilot-lured` pairs 1 and 2 got new language and frame models (held-out losses match
  pair 0's), then one search each with `configs/pilot-decipher-stats-split.yaml`, then the
  official score. Grid pair 1 follows (entry 88).
- *Result, pair 1* (search 3 hours 56 minutes, 8 finalists):
  - **Selected: the true reading. 62.9% of held-out frames exactly right**, against 0.03% for
    the same frames shuffled. Its agent order matches the key: 62.8% as numbered.
  - Places 96-97%, hunger 97%, glass, stone and apple 89-91%.
  - Finalist scores (selection, and exact frames scored afterwards): truth 19.34 (62.9%); its
    agent-swapped twin 19.27 (51.0%); then 18.72 and 18.62 (0%); 18.04, 18.01, 17.78 and
    17.64 (0.6-3.2%). The truth's twin is the only reading within 0.6.
- *Frozen code:* the patch of entry 86 was later committed unchanged as `feat/perf` commit
  81c97db (its diff hashes to the same 128f9ba35a05b21c), so these runs are reproducible from
  that commit.
- *Result, pair 2* (search 4 hours, 8 finalists):
  - **Selected: the true reading. 64.0% of held-out frames exactly right**, against 0.03% for the
    same frames shuffled. Its agent order matches the key: 63.8% as numbered.
  - Places 97-98%, hunger 97%, glass, stone and apple 88-93%.
  - Finalist scores: truth 19.52 (64.0%); its agent-swapped twin 19.45 (62.7%); then 19.23 and
    19.17 (0%), and 18.42, 18.17, 18.14 and 17.98 (0.9-6.2%). The nearest wrong readings trail
    by 0.29-0.35, a narrower margin than on pairs 0 and 1 (0.6).
- *Meaning:*
  - **Confirmed on both fresh pairs**, each run once with the method frozen, nothing
    hand-written and no truth used: 62.9% and 64.0% of frames exactly right. Pair 0, a tuning
    pair, gave 68.1%. The mean over the three pairs is 65.0%; the hand-written pilot method
    averaged 59.6% on its three pairs (entry 55), and the co-occurrence control 0.8%.
  - Each time, the top two finalists were the truth and its agent-swapped twin. They score
    within 0.07-0.32 of each other, as an exact symmetry should.
  - The nearest wrong readings are 0.29-0.6 behind. Not named here, since naming them uses the
    key; they decode no frames exactly, like the mirror in entry 86.
  - Next: grid video on the fresh pairs 1 and 2 (entry 88).

**88. On a fresh grid pair the frozen search selects the true reading, but its fit reads hunger poorly: 13% of frames exactly right.**
- *Why:* entry 87's confirmation, now on grid video. The method frozen as in entry 86, with
  `configs/pilot-grid-decipher-split.yaml`, run once on `pilot-grid-lured` pairs 1 and 2. The
  truth was looked at only after each search had chosen.
- *Result, pair 1* (search 4 hours 20 minutes, 24 matchings screened, 6 finalists):
  - **Selected: the true matching**, every thing and place word as in the hidden key. It was
    screened sixth of 24 and won the selection with 51.42.
  - But only **13.2% of held-out frames are exactly right** (0.01% shuffled). Places are 95-98%,
    glass 85%, stone and apple 53-58%, and hunger only 71-73%. On pair 0 the true matching's fit
    read hunger at 94-96% (entry 85).
  - The runner-up, at 51.11, has the places' mirror and the glass and apple words exchanged
    (0% exact). Then come the agent swap with glass and apple exchanged (50.32), and three
    readings with wrong places (49.31-50.17).
- *Result, pair 2:* [TBD]
- *Meaning so far:*
  - The grid search can choose the truth with no hints. Here the margin was 0.31, about one
    fit-to-fit noise gap (entry 85).
  - Choosing isn't decoding well. This fit's split moves didn't establish hunger, and with
    hunger weak the objects' rules (the apple is eaten when hungry) are read weakly too. On pair
    0 the opposite happened: hunger was read well and the choice went wrong.
  - Both come back to fit-to-fit variation on grid video, and the proposal already asks for
    restarts. Next: restarts per finalist, keeping the best by its own fit value (no truth), so
    each matching is judged on a good fit. [TBD]

**89. The grid world model proposes the next moment as well as the state-vector model, so grid video's gap is in the aligner, not the watcher.**
- *Why:* on grid video the decipherment decodes 13-47% of frames exactly under the true matching,
  against 63-68% on state vectors (entries 85-88). Before improving the learned world model,
  check whether it is the weak part.
- *How:* `runs/queue/world_model_check.py` (uses hidden/ to evaluate only). On 200 held-out
  episodes of lured pair 0, each trained world model sees the true frames up to each step and
  samples 32 next frames. Each sample is read back into a world state and compared with the true
  next state (8,044 steps).
- *Result:*

  | | State vectors | Grid video |
  |---|---|---|
  | Samples exactly right | 18.0% | 14.3% |
  | Most common sample exactly right | 23.6% | 23.6% |
  | Agent places | 59-60% | 69% |
  | Hunger | 95% | 95% |
  | Glass, stone, apple | 92-96% | 93-96% |

  The exact rates are low for both because the agents' moves are random. A world model can only
  give their odds.
- *Meaning:*
  - Given the true history, the two learned world models propose equally good next moments. So
    a larger or longer-trained grid world model would not close the gap, and nor would a learned
    abstraction of the video (option 4, discussed with Shane).
  - The loss is in the decipherment's fit on grid video. Its settings differ from the
    state-vector search in three ways, all chosen while grid searches took 8 hours: half the
    training episodes (256 against 512), summary power 0.5 against 0.2, and split units against
    joint.
  - Next: the true matching's fit, as in the search, with 512 training episodes, on grid pairs 1
    and 0 (entry 90). [TBD]
- *Also stopped:* the grid confirmation on pair 2 (entry 88), during screening, before any finalist
  was fitted. Its method is about to change, and the pair stays unused for confirming the new one.
