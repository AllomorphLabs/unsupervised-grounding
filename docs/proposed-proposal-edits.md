# Proposed proposal edits: grid-video pilot (phase 3, steps 1 and 2)

*Each item says what the evidence is (experiment log entries 56-80) and what would change. Items
1, 2, 4, 7 and 8 were applied to [proposal.md](proposal.md) on 25 November; 5 and 6 stay open.*

## 1. Say how grid video is drawn, and what that does to symmetry

**Applied (25 November).**

**Evidence:** entries 56-57. The renderer draws each place as a patch of cells, each thing in a
random free cell of its patch, a carried object in its carrier's cell, and nothing that marks who
acted. Which colour draws which agent is assigned at random per pair.

**Edit:** in *Experimental design*, step 3, add one sentence: places are patches of the image,
things are drawn in random cells within them, and which appearance belongs to which entity is
assigned at random per pair, so interchangeable agents differ only by an arbitrary colour. In
*Renderer checks*, note that a random appearance per pair is how grid video keeps the agent
symmetry.

## 2. Replace "Structure supplied to the aligner" with what is now done

**Applied (25 November)**, for both formats: state vectors now have their own statistics-only
parser (entry 67), so no [TBD] is left. The assumptions the procedures build in are stated
(technical decision 24).

**Evidence:** entries 58-59. For grid video, the frames' structure (places, things, appearances,
carrying) and the words' kinds are found from each stream alone, with no hand-written parser and
no grammar rules. The general procedures assume only that a frame is a grid of cells with
channels, and that tokens come in sequences.

**Edit:** rewrite the threat to say the grouping is now a general procedure for grid video, which
is the main condition, and that the structure found is checked against the hidden key only after
the fact. Keep the [TBD] only for state vectors (item 3).

## 3. Decide what to do about the state-vector comparison condition

**Evidence:** decision 21. A shuffled vector of facts has no cells or patches, so the general
procedure can't find places in it; state vectors still use the pilot's hand-written parser.

**Decided (25 November):** (b), a statistics-only parser for state vectors. The options as they
were put:
- (a) Keep state vectors, and accept the hand-written parser there as a declared limitation of the
  comparison condition only.
- (b) Keep state vectors, and build a statistics-only parser for them before pre-registration
  (more work; places would have to come from which values change together).
- (c) Replace the comparison condition with a "symbolic grid": the same grid geometry, one cell
  per thing per place and no randomness in cell choice. The same general procedure then applies
  to both conditions, and the gap between them measures what picture-like randomness costs.

(The recommendation was (c); (b) was chosen, which keeps the proposal's comparison condition as
written.)

## 4. Add impossible candidate frames to the threats to validity

**Applied (25 November).**

**Evidence:** entry 62. The grid frame model's own rollouts drift into impossible frames: only 14%
of 40-step rollouts stay valid throughout, against 99.6% valid single samples for the pilot's
state-vector model.

**Edit:** under *Threats to validity*, add: "A frame model can propose impossible frames. The
aligner gives no weight to candidates that break the structure found in the frame stream itself
(a thing drawn twice), and the share of valid samples is reported for every frame model."

## 5. Flag the open problem: learning to read hunger without an actor mark

**Evidence:** entry 63. The summaries hold hunger (97% in a two-way test), and with an oracle
summary model the grid pipeline decodes 64% of frames exactly right under the true matching. But
the fit learns its summary model from its own paths, and words alone don't anchor hunger, so it
never learns to read it (hunger stays at about 72%).

**Further evidence:** entry 65: on the search's own measure, a matching with all three object
words wrong beats the truth (17.10 against 17.00), so the search can't succeed until this is
fixed. Entry 66: marking who acted in the pictures doesn't help, so the missing mark isn't the
cause.

**Decided (25 November):** grid video stays unmarked; the fix comes from inside the method.

**Progress (entries 69-76):** split moves teach the summary model both agents' hunger with no
pairing, under the true matching; the full search is being tested. Still no design edit.

**Edit:** none to the design yet. This is a method problem to fix before the method is frozen on
the cipher control (phase 4). It should be listed among the open questions until it is solved.

## 6. Compute

**Evidence:** entries 60 and 64. A full-scale grid fit takes about 31 minutes against about 13
for state vectors, and the search now screens 24 matchings before fitting the finalists.

**Edit:** none yet. Revisit the compute estimate in *Work plan and resources* once the method is
settled, since the per-pair cost of alignment has roughly doubled.

## 7. Count near-symmetry confusions across pairs, not only within them

**Evidence:** entries 71 and 75. On pilot state vectors with nothing hand-written, the selected
matching was the places' mirror image: the path is symmetric end to end except for feeding at the
bush. That is a near-symmetry confused at the level of the whole pair, which the proposal's H3
analysis only covered within pairs.

**Applied (25 November):** *Meaning up to symmetry* says a near-symmetry can be confused within a
pair or across pairs, and that both are counted, the second like a swap rate; the H3
near-symmetry test measures confusion at both levels.

## 8. The objective diagnostic must compare equal fits

**Evidence:** entries 71 and 75. Without split moves the mirrored matching fitted better than the
right one (19.13 against 18.97-19.00); with them, the right one fitted better (19.21 against
18.81). Which of two fits scores higher can reflect where each fit settled, not the mappings.

**Applied (25 November):** step 7 (*Diagnose*) says both mappings are fitted by the same
procedure with the same restarts.

## 9. The cipher control, and what the method is tuned on

**Evidence:** technical decision 27 (decided with Shane, 27 November). The proposal's cipher control
was written for representation alignment. Under decipherment, a language is read against the world
model's frames, and a ciphered language stream has none.

**Applied (27 November):** the cipher row, *Freezing the method*, the overfitting threat and the
work plan's phase 4 now say the method is tuned on pilot-derived worlds; the cipher row describes
its decipherment. CLAUDE.md updated to match.

**Proposed:**
- *Controls* table, cipher row: keep "word-for-word substitution of a second corpus in the same
  language over the same world". Add that it is deciphered by classic substitution decipherment
  (Knight et al., 2006) against its own language model trained from scratch, and that it is the
  same-kind-of-data case the 2026 proof covers. It is the reference point against which the
  cross-modal result is compared.
- *Freezing the method* says the method "is tuned on the cipher control". That no longer holds: the
  frame-based decipherment was tuned on the pilot world and pilot-derived worlds, as technical
  decision 4 allows, and the cipher uses a different, simpler decipherment. Proposed wording: "The
  alignment method and its hyperparameters are tuned on pilot-derived worlds, then checked on a
  separate calibration world, used for nothing else. They are then frozen before any main
  condition runs." The cipher is tuned only for its own decipherment.
- *Threats*, alignment-method overfitting: the same change.

## 10. Main worlds are new worlds, not pilot variants

**Evidence:** technical decision 28 (decided with Shane, 27 November). The lured world (experiment
log, entry 84) is the pilot world with one added rule, and was used for tuning and confirmation.

**Applied (27 November):** step 1 (*Build the world*) says so.

**Proposed:** where the proposal describes the main worlds, say that they are new worlds (other
places, objects or rules), not variants of the pilot world, and that results in pilot-derived
worlds are reported as method development, not as evidence for the hypotheses.
