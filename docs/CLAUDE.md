# FPL pipeline — working context

## Environment

Python 3.7 (conda). **No walrus operator, no `X | Y` type unions, no f-string
`=` specifier.** Code written for 3.8+ will fail on this machine.

## What this is

Airflow ingests and models; LangGraph (phase 2) explains and converses. An LLM
is used in exactly two places — reading scraped prose, and the conversation.
Everything else is deterministic.

**Current implementation is plain Python, not yet Airflow-orchestrated.** The
archiver, derived layer, P(starts) model and scoring harness run as standalone
CLI scripts under `src/`, invoked manually or by cron/scheduler — see the
top-level `README.md` (repo root, not this `docs/` one) for exact commands and
run order. The derived layer is SQLite (`derived.db`), not the Postgres this
doc's Architecture section describes — a scale-appropriate substitution for
now, not a rejection of the design; revisit if/when Airflow actually gets
built.

Full detail in @README.md and @build_spec_minutes_model.md.

**Layer model** (referenced in code and validator output, defined in README):
Layer A = deterministic facts, owned by code, **errors block**. Layer B =
probabilistic judgment, owned by model and LLM, errors are scored not blocked.
Layer C = decisions under uncertainty, owned by the human.

## Non-obvious facts — do not rediscover these

- The API's `status` field means **not injured**. It has never meant "will
  start". Rotation and transfer-driven absences are invisible to it.
- `chance_of_playing_next_round` is **live state with no history**. No endpoint
  returns it for a past gameweek and no archive carries it. This is why the raw
  archiver is build step 1.
- `history` contains **unplayed** fixtures. Filter on `team_h_score is not
  None` before aggregating performance fields — but `value`, `selected` and
  `transfers_*` in those same rows are valid. The filter is field-specific.
- `element-summary.fixtures` **shrinks** as gameweeks complete. Snapshots
  cannot be reconstructed from a later pull.
- `defensive_contribution` per-gameweek starts in **2025/26**, not 2024/25,
  despite `history_past` reporting a 2024/25 total.
- Player **ids are reassigned each season**. Join on `code`.
- `bootstrap-static` is season totals only. For a time series use
  `event/{gw}/live` — one call covers the whole pool for one gameweek.
- Auto-substitutions process **after the whole gameweek**, not per fixture.
- A squad may legally hold **four players from one club** after a real-world
  transfer. Only the next transfer is constrained.
- Search team news by **club** ("<club> predicted lineup"), never by player
  name. Searching player names misses a fit rival displacing them — that is
  how Senesi was missed in GW2.

## Modelling rules

- **The model is a lookup table**, not a fitted model. `prev` × `roll4`,
  observed frequency per cell. Beta-Binomial was tested and adds nothing.
- **Anything new must beat 0.1798 Brier on the Rotation stratum.** Not
  persistence, not a pool-wide number.
- **Never report a single pool-wide Brier.** Deep players are 41% of rows and
  trivially predictable; they flatter everything.
- **Never treat a flag percentage as a probability.** A 50 flag does not mean
  P(starts) = 0.50. Use observed frequency, always.
- **Walk-forward only.** Temporal splits, and strata labelled from prior
  gameweeks only. Random splits leak heavily.
- **Score before refit.** Refitting first means evaluating on absorbed data.

## Architecture boundaries

- Agent has `SELECT` only on domain tables. Enforce with a DB role.
- Agent calls the optimiser; it does not pick the team.
- Airflow owns execution state, the archive owns domain state. Never put
  pipeline progress in an Airflow Variable or XCom.
- Archive raw responses, never parsed extracts. The derived layer must be
  rebuildable by replaying the archive.
- Scrapers **fail loudly**. An empty page must raise, never read as "nobody
  starts".

## Scripts

Built and tested (`src/`, run via `uv run python src/<name>.py` — see the
top-level `README.md` for the full run order):

- `api.py` — thin FPL API client
- `archiver.py` — raw archive: write-once, gzip, manifest; self-healing
  backfill sweeps for any `data_checked` gameweek not yet captured
- `derived.py` — SQLite layer (`derived.db`), rebuilt from scratch every run
  by replaying `raw/` and `predictions/`
- `starts_model.py` — the P(starts) lookup table, extended cross-season (see
  `notebooks/fpl_starts_analysis.ipynb` section 10) so early-season gameweeks
  aren't blind
- `scoring.py` — the calibration harness (§8): scores archived predictions
  against archived outcomes, stratified, against three baselines

Not yet built: expected minutes, the evidence layer (scraping), the
optimiser, the LangGraph agent.

`notebooks/fpl_starts_analysis.ipynb` reproduces every measured claim in
@README.md's "What has actually been measured" section, plus (section 10)
the season-boundary backtest that justified the cross-season extension in
`starts_model.py`.

The "Component requirements" section of @README.md covers details that will
be easy to get wrong when the squad validator/optimiser eventually get built,
particularly **selling price** (half of any profit, rounded down to the
nearest 0.1) — the single most costly thing to reimplement incorrectly.

**Superseded — ignore if present.** `squad_report.py` was a manual
pre-deadline report; its displacement heuristic (a fit club-mate priced within
85% with under half the minutes) was **removed from the design** and must not
be reintroduced — scraped team news replaces it. `fpl_runbook.md` describes a
manual workflow with a displacement scan that was removed. `spec_v3.odt` is the
original process spec; everything still current from it is in @README.md, and
its §5.2 market layer, §7.4 optimiser assumptions and §9.3 T-72h
pre-commitment have all been superseded.

## Current task

Build steps 1–7 (archiver, derived layer, backfill, incremental updater,
calibration harness, P(starts)) are done — see "Scripts" above. Step 5
(dedicated feature/filtering module for API gotchas 1–5) was mostly
obviated: this project's own archive only ever contains finished,
`data_checked` gameweeks, so the community archive's unplayed-fixture trap
(gotcha 1) doesn't apply to it.

Remaining, per the build order in @README.md: step 8 (expected minutes),
step 9 (evidence layer / scraping), step 10 (optimiser), step 11 (LangGraph
agent). Which one is next hasn't been decided in-session — ask rather than
assume.
