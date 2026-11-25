# Literature and novelty check (phase 2)

Phase 2 of the [proposal](proposal.md)'s work plan. **Gate:** the design is confirmed as novel,
or adjusted.

**Status:** first pass done (33 sources), and the gate is met: the design was adjusted, and the
proposal revised to match. Some full texts and a few searches are still to do; see "Still to read". The tracker shows how much of each source has been read.

## What this check is for

The proposal claims that no earlier work combines three things:

1. a controlled world with ground-truth scoring;
2. alignment of a form-only language model with a separately trained perceiving model, with no
   paired data;
3. predictions, registered in advance, of which errors must remain, derived from the world's
   symmetries.

The check looks for work that already does any two of these, or all three, and for work that
changes how the project should be designed or argued. It also covers the method the pilot
adopted (decipherment), which the original proposal didn't cite.

**How sources are judged.** Each source gets a *threat to novelty*:

- **High:** already does what we claim is new.
- **Medium:** does part of it, so we have to state the difference carefully.
- **Low:** related, and worth citing.
- **None:** background only.

Each source also gets a list of *design consequences*: anything that should change the plan.

**Rules.** Only sources that were found and checked are listed. Claims about a paper come from its
abstract or text, not from memory. A source that hasn't been read beyond its abstract is marked
that way.

## Tracker

| # | Source | Area | Read | Threat | Notes |
|---|---|---|---|---|---|
| 1 | [Syrota, Lopez i Serrano, Franck & Hauberg (2026), Identifiable alignment of unpaired representations](https://arxiv.org/abs/2611.11460), NeurIPS 2026 (per arXiv comments) | Identifiability of unpaired alignment | Read in full, including appendices | **Medium-high** | Proves unpaired alignment is identifiable **if and only if** the data distribution breaks every symmetry of the data manifold, up to its automorphisms. This is the theoretical core of H3, stated formally. The appendices confirm the gaps we rely on: one modality, static data, no graded result for near-symmetries, no partial overlap of concepts. See notes. |
| 2 | [Goldwasser, Gruber, Kalai & Paradise (2023), A Theory of Unsupervised Translation Motivated by Understanding Animal Communication](https://arxiv.org/abs/2211.11081), NeurIPS 2023 | Theory of unsupervised translation | Introduction and appendices C-F verbatim; sections 2-4 from summaries | Low | Already cited. The translator is a maximum-likelihood fit under a prior over the target language. In the knowledge-graph model, error is at most O(log n/(α²d) + (1/α)√(r log n/m)). The first term is a floor that **more data doesn't remove**, caused by ambiguities such as "multiple triangles in the source and target graphs". Richer relations (larger d) leave fewer ambiguities: "more complex languages can be translated … with less common ground". The target is assumed to contain the source, and the **partially overlapping case is left open**. Experiments are synthetic (10-node graphs). |
| 3 | [Maniparambil et al. (2024), Do Vision and Language Encoders Represent the World Similarly?](https://arxiv.org/abs/2401.05224), CVPR 2024 | Blind vision-language matching | Abstract | Low-medium | Unaligned vision and language encoders can be matched with no training, framed as seeded graph matching, via quadratic assignment. Earlier than Schnaus et al. (2025), and **not cited in the proposal**. |
| 4 | [Levy, Paradise, Carmeli, Meir, Goldwasser & Belinkov (2025), Unsupervised Translation of Emergent Communication](https://arxiv.org/abs/2502.07552), AAAI 2025 | Unsupervised translation, emergent language | Abstract | Low | Unsupervised machine translation of languages that agents invent in referential games. Semantic diversity makes them easier to translate. Relevant to the H4 sweep, and to the website's idea 2. |
| 5 | [Baevski, Hsu, Conneau & Auli (2021), Unsupervised Speech Recognition (wav2vec-U)](https://arxiv.org/abs/2105.11084), NeurIPS 2021 | Unpaired alignment across modalities | Abstract | Low | Speech recognition with no transcribed audio: a small generator maps frozen self-supervised speech features to phonemes, trained adversarially against unpaired text. The strongest real-world proof that two unpaired streams can be aligned. |
| 6 | [Bruce et al. (2024), Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391), ICML 2024 | World models from unlabelled video | Abstract | None-low | The Google DeepMind paper the user remembered. It learns a playable world model, including a small discrete set of *latent actions*, from unlabelled videos of 2D games. Notes below. |
| 7 | [Valevski, Leviathan, Arar & Fruchter (2024), Diffusion Models Are Real-Time Game Engines (GameNGen)](https://arxiv.org/abs/2408.14837), ICLR 2025 | Neural game engines | Abstract | None | A diffusion model simulates DOOM from past frames and actions, trained on an agent's recorded play. Background for a richer observation stream. |
| 8 | [Locatello et al. (2019), Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359), ICML 2019 | Identifiability | Abstract | None | Unsupervised disentanglement is impossible without inductive biases on both model and data. Background for why "up to symmetry" is the right target. |
| 9 | [Alvarez-Melis & Jaakkola (2018), Gromov-Wasserstein Alignment of Word Embedding Spaces](https://aclanthology.org/D18-1214/), EMNLP 2018 | Unsupervised word translation | Abstract | None | Aligns embedding spaces by matching how pairwise similarities line up. The pilot tried Gromov-Wasserstein matching on single frames, and it failed (decision 16). |
| 10 | [Schnaus, Dagès, Cremers, Wang & Isola (2026), Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data](https://arxiv.org/abs/2610.09411), preprint (v3, 3 Nov 2026) | Blind vision-language matching | Abstract | Medium | Follows up Schnaus et al. (2025). GeoWarP fits one orthogonal map between independently trained unimodal models with **zero pairs** (Wasserstein Procrustes with a coarse geometric start). Standard geometric alignment metrics predict *when* this works. The abstract says nothing about symmetries or which concepts get confused. It's the state of the art for the unpaired cross-modal claim on real data, so the proposal should cite it. |
| 11 | [Gröger, Wen & Brbić (2026), Revisiting the Platonic Representation Hypothesis: An Aristotelian View](https://arxiv.org/abs/2602.14486), ICML 2026 | Platonic hypothesis | Abstract | None | Similarity metrics are inflated by model scale. After permutation-based calibration, global convergence largely disappears, but local neighbourhood relations still agree across modalities. |
| 12 | [You, Jang, Mo & Jung (2026), What Converges in the Platonic Representation Hypothesis? Structure over Geometry](https://arxiv.org/abs/2609.27252), preprint | Platonic hypothesis | Abstract | None | Relational structure (which samples are related) converges locally and globally; metric geometry (distances) converges much less. Together with 11, this supports the pilot's switch away from geometric matching. |
| 13 | [Vafa, Chen, Rambachan, Kleinberg & Mullainathan (2024), Evaluating the World Model Implicit in a Generative Model](https://arxiv.org/abs/2406.03689), NeurIPS 2024 | World models in sequence models (H1) | Abstract | None | Models that pass next-token and probe diagnostics can still have incoherent world models. The proposed metrics are based on the Myhill-Nerode theorem: two states are different only if some continuation tells them apart. **Design consequence for H1** (see notes). |
| 14 | [Nanda, Lee & Wattenberg (2023), Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941), BlackboxNLP 2023 | World models in sequence models (H1) | Abstract | None | OthelloGPT's board *is* linearly readable once it's expressed relative to the current player ("mine"/"yours") rather than black/white. A linear probe can fail just because the frame of reference is wrong, which matches what the pilot saw on pair 1 (entry 42). |
| 15 | [Luo, Cao & Barzilay (2019), Neural Decipherment via Minimum-Cost Flow: from Ugaritic to Linear B](https://aclanthology.org/P19-1303/), ACL 2019 | Decipherment | Abstract | None | Neural decipherment of lost languages: about 5.5 points better than Snyder et al. (2010) on Ugaritic, and the first automatic results on Linear B (67.3% of cognates). The neural successor to the decipherment work the proposal now cites. |
| 16 | [Narayanan & Shmatikov (2009), De-anonymizing Social Networks](https://arxiv.org/abs/0903.3276), IEEE S&P 2009 | Graph alignment | Abstract and secondary sources | None | Re-identifies users by aligning two social graphs from structure alone, though it needs an initial seed step. A well-known non-ML example of the same problem: structure fixes identity up to the graph's symmetries. |
| 17 | [Feng, Ma, Liu & Luo (2019), Unsupervised Image Captioning](https://arxiv.org/abs/1811.10787), CVPR 2019 | Unpaired image-text | Abstract | None | Captioning from unpaired images and sentences, but with a pretrained object detector, which brings labels in. A useful contrast for the proposal's "no pretrained components" rule. |
| 18 | [Sharma et al. (2024), Contextual and combinatorial structure in sperm whale vocalisations](https://www.nature.com/articles/s41467-024-47221-8), Nature Communications | Animal communication | Press coverage only | None | Project CETI's "phonetic alphabet" of sperm-whale codas: combinable features (rhythm, tempo, rubato, ornamentation). What they mean is still unknown. Motivation for H4. |
| 19 | [Xu & Kanan (2026), More Data Cannot Break a Symmetry: Identifiability by Design](https://arxiv.org/abs/2608.27651), preprint (27 Aug 2026) | Symmetry-limited alignment, experiments | Full text (HTML), figures not viewed | **Medium-high** | Gromov-Wasserstein alignment over permutations, between two systems' responses to the *same* stimuli. Theorem: the correspondence is identifiable "at best up to an orbit of Aut(D1), irrespective of sample size or optimiser". They credit this invariance to Demetci et al. (2024). Their contribution is a design-time check plus an intervention: replacing a symmetric colour ring with a set chosen to break symmetry cuts catastrophic failures from 75.3% to 2.2% across 93 vision-model representations. **The closest experimental precedent for H3.** See notes for the differences. |
| 20 | [Mollo & Millière (2023, rev. 2025), The Vector Grounding Problem](https://arxiv.org/abs/2304.01481), Philosophy and the Mind Sciences (accepted) | Philosophy of grounding | Abstract | None | Of several senses of "grounding", only *referential* grounding matters, and text-only models may achieve it. Useful for stating what H2 tests in philosophical terms. |
| 21 | [Wu et al. (2025), The Mechanistic Emergence of Symbol Grounding in Language Models](https://arxiv.org/abs/2510.13796), ICML 2026 | Grounding in language models | Abstract | Low | In a controlled setting built from child-caregiver dialogue (CHILDES), models with no grounding objective still link words to the environment, and this happens in middle layers. Unlike us, their models see the environment; ours never do. |
| 22 | [Merrill, Goldberg, Schwartz & Smith (2021), Provable Limitations of Acquiring Meaning from Ungrounded Form](https://aclanthology.org/2021.tacl-1.62), TACL | Theory of meaning from form | Abstract | None | Proves that meaning can be emulated from "assertions" when expressions are semantically transparent, but can be uncomputable when meaning depends on context. A theoretical limit on the language side; ours is a limit set by the world's symmetries. |
| 23 | [Schmidt & Jiang (2024), Learning to Act without Actions (LAPO)](https://arxiv.org/abs/2312.10812), ICLR 2024 | Latent actions from video | Abstract | None | Recovers latent actions, policies and world models from video alone. On Procgen the latent action spaces reflect the true ones. With Genie, the closest work on the frame side. |
| 24 | [Hyvärinen, Khemakhem & Monti (2024), Identifiability of latent-variable and structural-equation models: from linear to nonlinear](https://arxiv.org/abs/2302.02672), Annals of the Institute of Statistical Mathematics | Identifiability | Abstract | None | Review: nonlinear latent-variable models are badly unidentifiable without extra structure, but become identifiable given time series or auxiliary variables. A formal reason why the pilot needed the world's dynamics (decision 16). |
| 25 | [Schnaus, Araslanov & Cremers (2025), It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](https://arxiv.org/abs/2503.24129), CVPR 2025 | Blind vision-language matching | Main text (HTML); appendices C-G not read | Medium | Already the proposal's closest comparison. One embedding per class per modality, matched as a quadratic assignment problem (Gromov-Wasserstein distortion) with a new solver. At 10 classes most model pairs beat 10% chance, and the best reach 80-100%. Accuracy falls as the number of classes grows. Some classes "cannot be matched reliably" because the two modalities represent them differently. Symmetry gets one sentence: matching "may be ambiguous due to multiple (equivalent) local minima". No confusion analysis, no synthetic world, all models pretrained. |
| 26 | [Demetci, Tran, Redko & Singh (2024), Breaking isometric ties and introducing priors in Gromov-Wasserstein distances](https://proceedings.mlr.press/v238/demetci24a.html), AISTATS 2024 | Symmetry-limited alignment, theory | Abstract | Low | Gromov-Wasserstein can't tell apart solutions related by an isometry ("isometric ties"). Adds a feature-level term to break them. Xu & Kanan (19) credit the invariance to this paper. |
| 27 | [Pashakhanloo & Zavatone-Veth (2026), Stimulus symmetries can confound representational similarity analyses](https://arxiv.org/abs/2605.21324), preprint | Symmetry and representation comparison | Abstract | Low | Symmetries in the stimuli make many neural codes do the same job, yet give different similarity matrices. A warning for H1's probes and for any comparison of representations. |
| 28 | [Jha, Zhang, Shmatikov & Morris (2025), Harnessing the Universal Geometry of Embeddings (vec2vec)](https://arxiv.org/abs/2505.12540), NeurIPS 2025 | Unpaired embedding translation | Abstract (already cited) | Low | Translates text embeddings between models with no pairs, via adversarial and cycle losses (cosine up to 0.96 with the true target vectors). The authors argue it supports a strong Platonic hypothesis. It failed on the pilot's frames-to-frames control (decision 16). Text-to-text only. |
| 29 | [Liang, Jordan & Klein (2009), Learning Semantic Correspondences with Less Supervision](https://aclanthology.org/P09-1011/), ACL 2009 | Grounded language learning with EM | Abstract | Low (method only) | Segments text into utterances and maps each one to a record of a world state, fitted by EM, on RoboCup, weather and NFL data. **A direct precedent for the pilot's word model, where each sentence states one fact** (entry 44). Unlike us, text and world state come paired at the level of the scenario. |
| 30 | [Ravindran & Barto (2003), SMDP Homomorphisms: An Algebraic Approach to Abstraction in Semi-Markov Decision Processes](https://ijcai.org/proceedings/2003/144), IJCAI 2003, pp. 1011-1017; see also [Ravindran's thesis (UMass Amherst, 2004)](https://all.cs.umass.edu/pubs/2004/ravindran_thesis04.pdf) | Symmetries of transition systems | Abstract, plus summaries of the thesis | None | Homomorphisms map one dynamic system to another while keeping its dynamics. The automorphisms of a Markov decision process form its symmetry group. *Approximate* homomorphisms merge states that are nearly equivalent. This is the established formal home for the proposal's "relabellings that commute with the dynamics" and for its near-symmetries. |
| 31 | [Social-JEPA: Emergent Geometric Isomorphism in Independently Trained World Models](https://arxiv.org/abs/2603.02263), preprint (2026) | Aligning world models | Search summary only | Low, to check | World models trained separately on the *same* environment fall into the same linear equivalence class, so one linear map aligns them. **Not yet clear whether that map is fitted without pairs.** The only work found that aligns two world models. Read in full. |
| 32 | NeurIPS pre-registration workshops: [2020](https://neurips.cc/virtual/2020/workshop/16158) and [2021](https://neurips.cc/virtual/2021/workshop/21885) | Pre-registration in ML | Workshop pages | None | Papers accepted on their experimental protocol, before any results, following a trial in computer vision (Henriques et al., 2019). A precedent for the proposal's pre-registration. |
| 33 | [van Miltenburg et al. (2021), Preregistering NLP Research](https://arxiv.org/abs/2103.06944) | Pre-registration in ML | Search summary only | None | Argues for pre-registration and registered reports in NLP, where they're rarely discussed. |

## Notes by area

### 1. Symmetry limits unpaired alignment: now established (19, 1, 26, 27)

The idea at the centre of H3 is that unpaired alignment can only work up to the world's
symmetries. Four recent papers say this, two of them formally:

- **Syrota et al. (2026)** prove that aligning two models of the same data without pairs is
  identifiable *if and only if* the data distribution breaks every symmetry of the data manifold.
  The leftover ambiguity is exactly the data's automorphisms. Their real-data failures are mirror
  images: on CelebA, "the maps send faces to their mirror images". They also show that symmetry
  breaking holds *generically*, so exact symmetries are rare in natural data.
- **Xu & Kanan (2026)** state and test the same principle for comparing representations: the
  correspondence is identifiable "at best up to an orbit of Aut(D1)". They also show that more
  restarts don't help (a 64× restart budget leaves a symmetric ring unmoved), and that designing
  stimuli to break the symmetry cuts catastrophic failures from 75% to 2%.
- **Demetci et al. (2024)** and **Pashakhanloo & Zavatone-Veth (2026)** describe the same kind of
  ties for Gromov-Wasserstein matching and for similarity analyses.

**What this means for us.** The proposal must no longer present "meaning up to symmetry" as an
idea we bring to the field. It's now a theorem in a closely related setting. That strengthens the
project rather than weakening it: H3 becomes a test of whether an established principle carries
over to a setting it hasn't been tested in. Each of these papers differs from ours in the same
ways:

| | Syrota et al. (2026) | Xu & Kanan (2026) | Project 01 |
|---|---|---|---|
| What is aligned | Two models of the **same data** | Two systems' responses to the **same stimuli** | A language model and a perceiving model trained on **different, independent streams** |
| Do the inputs correspond? | Same data manifold | Same stimuli | No shared items, episodes or timing |
| Source of the symmetry | Geometry of a static data manifold | Geometry of a static stimulus set | Dynamics of a world: relabellings that commute with stepping |
| Language involved? | No | No | Yes: a form-only language model trained from scratch |
| Predictions registered in advance? | No | No (a design-time check) | Yes |
| Swap rates across independent samples | No | No (solver restarts only) | Yes, tested for equivalence to 50% |
| Near-symmetries scored and ranked | No | Partly, without calibration | Yes, violation rate against confusion rate |
| Model-attributable errors beyond the symmetries | Not separated | Not separated | Yes, via the objective diagnostic and the noise floor |

One point matters for H3 directly. Syrota et al.'s theorem assumes both models are smooth,
invertible maps of the *same* data manifold. Language and perception break that assumption,
since they share a world but not their data. So whether errors in our setting follow the
symmetries *exactly* is a real empirical question, not a corollary. If they don't, that's the
second row of the proposal's outcomes table, and the theorem's assumptions tell us where to look.

**What the appendices of Syrota et al. add.** Checked against the full paper (arXiv v1):

- **The setting is one modality, by design.** Both models must be smooth charts of one shared
  data manifold. The two samples can be disjoint (lung-cell models trained on different cells),
  but they come from the same data-generating process. The paper calls this "the case to settle
  first" and leaves other cases as "an open problem". Two streams in different modalities, sharing
  only a world, fall outside the theorem.
- **No dynamics.** A symmetry is a distance-preserving map of the data manifold that keeps its
  probability distribution unchanged. Nothing involves time, sequences or transitions.
- **Near-symmetries are qualitative only.** Their one case is the CelebA left-right flip, which
  "nearly preserves" the distribution of faces (Appendix E.5). 7 of 35 draws failed with their
  method, and 10 of 35 with plain Gromov-Wasserstein matching, and every failure was a mirror
  alignment. They don't compare that rate with any expected value or relate it to how nearly the
  flip holds. Their error bounds (Proposition B.15) depend on a "margin" between points' local
  distance profiles, not on how nearly a symmetry holds, and they're not evaluated because the
  constants are unknown. So a graded prediction (violation rate against confusion rate) is still
  open.
- **A warning for our symmetry-breaking tests.** Appendix C shows that breaking every symmetry
  isn't always enough in practice. A world can have no exact symmetry yet still contain distinct
  points that look identical to the matching method, so recovery can still fail. In our terms, a
  symmetry broken only slightly may leave a near-symmetry, and its confusion, in place. The
  pre-registration should state how strongly each symmetry is broken, measured by violation rate,
  rather than only *that* it is broken.
- **Their "overlap" sweep is about shared samples, not shared concepts.** Figure 6 varies the share
  of images both samples contain (0 to 100%). The theory assumes full support on both sides, so
  partial overlap of what the two sides can describe, our H4, isn't covered.
- **No language, no cross-modal experiments.** Experiments are on a sphere, image VAEs (MNIST,
  FashionMNIST, CelebA) and single-cell models.

### 2. Blind cross-modal matching on real data (25, 10, 3, 28)

Maniparambil et al. (2024), Schnaus et al. (2025), Schnaus et al. (2026, "Rosetta") and Jha et
al. (2025) show that independently trained vision and language encoders, or two text encoders,
can be matched with few or no pairs. All of them use large pretrained models on real data, so
nobody knows the true referents, the world's symmetries or the overlap between modalities. That
gap is exactly the proposal's motivation, and it still stands. Schnaus et al. (2025) mention
symmetry in one sentence and give no confusion analysis. Rosetta (2026) predicts *when*
alignment works from geometric metrics, not *which* errors remain. The proposal should cite
Maniparambil et al. (2024) and Schnaus et al. (2026). It currently cites neither.

### 3. The Platonic hypothesis is contested in 2026 (11, 12)

Gröger et al. (ICML 2026) show that the usual convergence measures are inflated by model scale.
After calibration, only local neighbourhood structure is shared across modalities. You et al.
(2026) argue that relational structure (which items relate to which) converges, while metric
geometry (distances) converges much less. Both fit what the pilot found: matching geometry
failed, and matching structure through dynamics worked. The proposal's Background section
describes the hypothesis without this debate. It should mention it, and point out that scoring
against ground truth sidesteps the problem of calibrating similarity measures.

### 4. Testing whether a model holds a world (13, 14, 27; bears on H1)

Vafa et al. (2024) show that models can pass next-token and probe tests while holding an
incoherent world model. Their tests ask whether states that should differ can be told apart by
some continuation, after the Myhill-Nerode theorem. Nanda et al. (2023) show that a linear probe
can fail just because the frame of reference is wrong: OthelloGPT's board is linear once it's
expressed as "mine" and "yours". The pilot hit both problems: probes read pair 0's summaries
well, and pair 1's language model stored objects in a form a straight line couldn't read
(entry 42). H1 currently relies on linear probes plus interventions. It would be stronger with a
test in the style of Vafa et al., which our simulator makes cheap, since every state is known.

### 5. Unpaired translation theory and overlap (2, 4, 18; bears on H4)

Goldwasser et al. (2023) predict that richer languages are easier to translate given common
ground. Their error bound has a floor that more data doesn't remove, coming from structural
ambiguities, which is the same idea as H3 seen from the theory side. They assume the target
contains the source, and they leave **partial overlap open**. The H4 overlap sweep measures
exactly that, so the proposal can claim it as a direct test of an open case. Levy et al. (2025)
translate languages that agents invent, without parallel data, and find that semantic diversity
helps. That's evidence for the "richer is easier" side of H4.

### 6. Decipherment and grounded language learning (15, 29; the method)

The method's lineage now has a neural branch (Luo et al. 2019) and a grounded branch (Liang et al.
2009, whose EM model maps each utterance to one record of a world state, like the pilot's word
model). Neither aligns two *learned* models, and neither works without scenario-level pairing.
The proposal should cite Liang et al. (2009) next to the decipherment papers.

### 7. World models from unlabelled video (6, 7, 23; the user's lead)

The Google DeepMind paper the user remembered is **Genie** (Bruce et al., 2024). It learns a
playable world model from unlabelled videos of 2D games, including a small discrete vocabulary
of *latent actions* discovered with no action labels. LAPO (Schmidt & Jiang, 2024) does the same
on Procgen and finds the latent actions mirror the true ones. GameNGen (Valevski et al., 2024)
learns DOOM as a neural game engine, but from recorded actions.

They matter to us in three ways:

1. They're the state of the art for the *watcher* side: models that learn a world's dynamics
   from observation alone. That's a pointer for the grid-video frame model in phase 3.
2. Latent actions are only defined up to relabelling: a model can't know which of its codes is
   "jump". That's the same identifiability limit, on the frame side alone. A small extension of
   our setup could test it.
3. Their worlds are real games with unknown symmetries. Ours are built so the symmetries are
   known. None of them aligns a world model with a language model.

### 8. Philosophy of grounding (20, 21, 22)

Mollo & Millière (2023, rev. 2025) separate kinds of "grounding" and argue that text-only models
may achieve the one that matters, referential grounding. Merrill et al. (2021) give a formal
limit on learning meaning from form alone, on the language side. Wu et al. (2025) show grounding
emerging in models that do see the environment. These help state precisely what H2 tests:
whether referential grounding can be *recovered* by matching to a perceiver. They don't overlap
with the method.

## Novelty assessment

**Provisional verdict: the design is still novel, but the framing needs adjusting.** Based on the
33 sources found so far:

- **No source does all three things** the proposal claims together: a controlled world with
  known answers; unpaired alignment of a form-only language model with a separately trained
  perceiver; and symmetry-derived predictions of the leftover errors, fixed in advance.
- **The principle behind H3 is no longer ours to claim.** Syrota et al. (2026) prove it for
  unimodal, static alignment, and Xu & Kanan (2026) test it experimentally on static stimuli,
  including deliberate symmetry breaking. The proposal's "Meaning up to symmetry" section and its
  "Relation to prior work" table must cite them and present H3 as testing the principle in a new
  setting: across modalities, through dynamics, with registered predictions, swap rates across
  independent pairs, and graded near-symmetries.
- **What stays new**, as far as this search found:
  1. Two streams that share a world but no data. Every alignment paper found aligns
     representations of the same items.
  2. Symmetries defined by dynamics (automorphisms of a transition system), not by the static
     geometry of a set of points.
  3. Registered predictions tested across independent pairs: equivalence of swap rates to 50%, a
     rank correlation for near-symmetries, and symmetry-breaking interventions on a world.
  4. A direct test of the partial-overlap case that Goldwasser et al. (2023) leave open (H4).
  5. Ground-truth scoring of a question that the blind-matching papers can only study on real
     data with unknown referents.

This check isn't complete yet; see "Still to read". A source found later could change the
verdict.

## Proposed design adjustments

All nine are now in the proposal, whose second version was rewritten to include them; its section "How this proposal came to be" summarises them.

1. **Cite the new identifiability results and reframe H3** (Syrota et al. 2026; Xu & Kanan 2026;
   Demetci et al. 2024). H3 becomes a test, in a cross-modal and dynamic setting, of a principle
   now proven for static unimodal alignment. Make the point that the theorem's assumptions fail
   across modalities, so the result isn't a foregone conclusion.
2. **Add the 2026 blind-matching work** (Schnaus et al. 2026; Maniparambil et al. 2024) to
   "Relation to prior work".
3. **Mention the 2026 debate over the Platonic hypothesis** (Gröger et al.; You et al.) in the
   Background, and note that ground-truth scoring avoids the calibration problem.
4. **Strengthen H1** with a world-model test in the style of Vafa et al. (2024): do states that
   should differ lead to different continuations? This complements probes, which can over-report
   (Vafa et al.) or under-report (Nanda et al.; pilot entry 42).
5. **Frame H4 as the partial-overlap case** that Goldwasser et al. (2023) leave open, and cite
   Levy et al. (2025).
6. **Pre-register a symmetry-cost diagnostic for each world**, after Xu & Kanan's "catastrophic
   cost": how much a candidate relabelling costs under the dynamics, computed before any model is
   trained. It's close to our violation rate. It would give a second, independent prediction of
   which near-symmetries alignment will confuse.
7. **Cite the method's lineage** (Liang et al. 2009; Luo et al. 2019), **the formal home of the
   symmetry definition** (Ravindran & Barto 2003), and **precedents for pre-registration in ML**
   (the NeurIPS workshops of 2020 and 2021).
8. **Phase 3, the observation stream:** look at Genie and LAPO when designing the grid-video
   frame model. A possible extension is a latent-action condition that tests the same symmetry
   limit on the frame side alone.
9. **Register how strongly each symmetry is broken**, not only that it is broken. Following
   Syrota et al.'s Appendix C, a symmetry broken only slightly can leave a near-symmetry and its
   confusion in place. Each symmetry-breaking intervention should state its violation rate in
   advance, and the H3 prediction should depend on it.

## Still to read

- **Full texts** still needed for: Schnaus et al. (2026,
  Rosetta); Maniparambil et al. (2024); Vafa et al. (2024); Genie (2024), especially its
  latent-action model.
- **Read Social-JEPA (31) in full**, to see whether its alignment uses pairs.
- **Already in the proposal, not yet re-read here:** Huh et al. (2024), Li et al. (2023), Li, Nye
  & Andreas (2021), Jin & Rinard (2024), Patel & Pavlick (2022), Abdou et al. (2021), Bender &
  Koller (2020), Piantadosi & Hill (2022), Conneau et al. (2018), Lample et al. (2018), Artetxe et
  al. (2018), Søgaard et al. (2018), Newman (1928).
- **Searches not yet run:**
  - equivalence testing of swap rates (method references for the analysis plan);
  - more on emergent communication and its translation.
