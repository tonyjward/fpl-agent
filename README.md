# FPL

Fantasy Premier League data pipeline: archives the live API before its
live-only fields (availability, news) disappear for good, derives a
queryable SQLite layer from that archive, and predicts P(starts) from a
validated lookup-table model. Full design spec and rationale live in
`docs/build_spec_minutes_model.md` and `docs/README.md`.

## Setup

    uv sync

## Running the pipeline

Run in this order. `derived.py` needs to run **twice** in a full cycle —
once before predicting, to establish current state, and once after, to pick
up the prediction snapshot `starts_model.py` just wrote — because
`scoring.py` only ever reads `derived.db`, never `predictions/*.json`
directly.

1. **Archive today's data.**

       uv run python src/archiver.py

   Fetches `bootstrap-static` + `fixtures`, and sweeps every gameweek that's
   now `data_checked` but not yet archived (self-healing backfill — this is
   where a just-finished gameweek's real results get picked up).

2. **Rebuild the derived layer.**

       uv run python src/derived.py

   Replays `raw/` (and whatever's already under `predictions/`) into
   `derived.db`: `teams`, `players`, `player_gameweek_stats`,
   `player_availability_snapshots`, `predictions`.

3. **Predict the upcoming gameweek.**

       uv run python src/starts_model.py

   Auto-detects the target round (max archived round + 1) and season, writes
   a timestamped snapshot to `predictions/{season}/`.

4. **Rebuild the derived layer again**, so the new prediction is actually
   queryable — easy step to forget, since nothing errors if you skip it,
   `scoring.py` will just silently score a stale prediction:

       uv run python src/derived.py

5. **Wait** for the gameweek to be played and `data_checked`, then repeat
   steps 1–2 to pull in the real outcome.

6. **Score it.**

       uv run python src/scoring.py --target-round N

   Compares the archived prediction against actual outcomes, per stratum
   (Core / Rotation / Marginal / Deep) and against three baselines
   (persistence, season-rate, constant 0.9). Never trust a single pool-wide
   number — Rotation is the stratum that matters
   (`docs/build_spec_minutes_model.md` §8b).

All four scripts also take `--season`, and `starts_model.py` /
`scoring.py` take `--target-round` / `--prior-season` if you need to
override the auto-detected defaults (e.g. to retrospectively predict or
score an already-played gameweek).

## Tests

    uv run pytest

## Layout

| Path | Purpose |
|---|---|
| `src/api.py` | Thin FPL API client |
| `src/archiver.py` | Raw archive: write-once, gzip, manifest |
| `src/derived.py` | SQLite layer rebuilt by replaying `raw/` |
| `src/starts_model.py` | P(starts) lookup-table model |
| `src/scoring.py` | Scores archived predictions against archived outcomes |
| `raw/`, `derived.db`, `predictions/` | Generated locally, not tracked in git — always rebuildable from source |
| `notebooks/` | Exploratory analysis (not tracked in git) |
| `docs/` | Design spec and rationale |
