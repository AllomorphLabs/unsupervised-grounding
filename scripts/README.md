# Scripts

Experiment tooling that is not part of the `grounding` package.

- `queue/`: the unattended GPU job queue. `runner.ps1` runs the jobs listed in `runs/queue/jobs.txt` one at a time (each writes `runs/queue/<name>.log` and `<name>.done`); `watch.sh` reports job ends, errors, stalls, memory spill and an idle queue. Job lists, logs and results stay in `runs/`, which is git-ignored.
- `diagnostics/`: one-off checks and diagnostics from the experiment log (`docs/alignment-attempts.md`), named in the entries that used them. Most read `hidden/` to score a fit, so they run on tuning pairs only and are never part of the method. The main ones:
  - `refit_restarts.py`: refits chosen finalists from given seeds; saves checkpoints after every iteration and resumes from one with `FIT_RESUME` (entry 98).
  - `hunger_halves.py`, `halves_score.py`: decode or score a fit with its word tables or summary model swapped for ones fitted to the true frames (entries 101-102).
  - `check_exact_objects.py`: checks that `exact_objects` decoding reproduces the scripts that first tested it (`object_smooth2.py`, `smoothlib.py`, `smoothed_em.py`, entry 106), which `grounding.align.objects` replaces.

Run them from the repo root, for example `uv run python scripts/diagnostics/refit_restarts.py ...`.
