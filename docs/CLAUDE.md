# FPL pipeline — working context

## Environment

Python 3.7 (conda). **No walrus operator, no `X | Y` type unions, no f-string
`=` specifier.** Code written for 3.8+ will fail on this machine.

## What this is

Airflow ingests and models; LangGraph (phase 2) explains and converses. An LLM
is used in exactly two places — reading scraped prose, and the conversation.
Everything else is deterministic.

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

- `fpl_starts_analysis.ipynb` — reproduces every measured claim

No implementation exists yet. Build it from @README.md — the "Component
requirements" section covers the details that are easy to get wrong,
particularly **selling price** (half of any profit, rounded down to the nearest
0.1) which is the single most costly thing to reimplement incorrectly.

**Superseded — ignore if present.** `squad_report.py` was a manual
pre-deadline report; its displacement heuristic (a fit club-mate priced within
85% with under half the minutes) was **removed from the design** and must not
be reintroduced — scraped team news replaces it. `fpl_runbook.md` describes a
manual workflow with a displacement scan that was removed. `spec_v3.odt` is the
original process spec; everything still current from it is in @README.md, and
its §5.2 market layer, §7.4 optimiser assumptions and §9.3 T-72h
pre-commitment have all been superseded.

## Current task

Building the Airflow component. Start with build step 1 in @README.md — the raw
archiver. It has a real deadline: unarchived availability data is unrecoverable.
