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

   **Also archive news and injuries** (optional, but must run before step 2
   to be picked up by it):

       uv run python src/news_archiver.py --season 2026-27

   Fetches premierleague.com's news and injury content (see `src/pl_content.py`)
   and archives it the same way -- see docs/README.md's "Evidence layer".

2. **Rebuild the derived layer.**

       uv run python src/derived.py

   Replays `raw/` (and whatever's already under `predictions/`/
   `extractions/`) into `derived.db`: `teams`, `players`,
   `player_gameweek_stats`, `player_availability_snapshots`,
   `news_articles`, `injury_reports`, `news_claims`, `predictions`.
   `injury_reports`/`news_claims` are checked against `players.team_code`
   at rebuild time and flag any disagreement in a `contradiction` column
   -- see `derived.match_injury_player`.

   **Extract news claims** (needs `ANTHROPIC_API_KEY` or `ant auth login`;
   run after this step, since it reads `news_articles` from `derived.db`,
   and before the next rebuild, since that's what picks its output up):

       uv run python src/news_extraction.py --season 2026-27

   Classifies each not-yet-processed article into a fixed taxonomy (never
   a probability -- see `news_extraction.py`'s module docstring) and writes
   one write-once JSON file per article to `extractions/{season}/`.

3. **Rebuild the derived layer again**, to pick up `extractions/`:

       uv run python src/derived.py

4. **Predict the upcoming gameweek.**

       uv run python src/starts_model.py

   Auto-detects the target round (max archived round + 1) and season, and
   writes a timestamped snapshot per `model_version` ("raw_lookup",
   "refined_availability", "refined_availability_news") to
   `predictions/{season}/`.

5. **Rebuild the derived layer again**, so the new predictions are actually
   queryable — easy step to forget, since nothing errors if you skip it,
   `scoring.py` will just silently score a stale prediction:

       uv run python src/derived.py

6. **Wait** for the gameweek to be played and `data_checked`, then repeat
   step 1 to pull in the real outcome.

7. **Score it.**

       uv run python src/scoring.py --target-round N

   Compares each archived `model_version`'s predictions against actual
   outcomes, per stratum (Core / Rotation / Marginal / Deep) and against
   three baselines (persistence, season-rate, constant 0.9). Never trust a
   single pool-wide number — Rotation is the stratum that matters
   (`docs/build_spec_minutes_model.md` §8b).

Every script also takes `--season`, and `starts_model.py` /
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
| `src/pl_content.py` | Client for premierleague.com's public content API (news, injuries) |
| `src/news_archiver.py` | Raw archive for `pl_content.py`, same write-once/manifest pattern |
| `src/news_extraction.py` | LLM classification of news articles into a fixed claim taxonomy |
| `src/derived.py` | SQLite layer rebuilt by replaying `raw/`, `predictions/`, `extractions/` |
| `src/starts_model.py` | P(starts) lookup-table model, plus the news-evidence layer on top |
| `src/scoring.py` | Scores archived predictions against archived outcomes |
| `raw/`, `derived.db`, `predictions/`, `extractions/` | Generated locally, not tracked in git — always rebuildable from source |
| `notebooks/` | Exploratory analysis (not tracked in git) |
| `docs/` | Design spec and rationale |
