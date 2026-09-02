# FPL decision pipeline

A measured, auditable process for Fantasy Premier League team selection.

Two systems. **Airflow** runs ingestion, modelling and scoring on a schedule.
**LangGraph** runs a conversational agent that explains what the pipeline
produced and lets the operator feed in information the pipeline cannot see.

The design principle throughout: an LLM is used for the two things it is
uniquely good at — reading prose that fits no schema, and holding a two-way
exchange with a human. Everything else is code, solvers and validators.

---

## Status

| Component | Runs on | State |
|---|---|---|
| Raw archiver | Airflow | Specified — **build first, see Deadline below** |
| P(starts) model | Airflow | Specified and validated on real data |
| Expected minutes | Airflow | Specified, not built |
| News scraping | Airflow | Specified — see Component requirements |
| News extraction (agent) | Plain LLM call in Airflow | Not built |
| Squad optimiser | Called as a tool | Use `open-fpl-solver`, not wired up |
| Squad validator | Airflow | Specified — see Component requirements |
| Conversational agent | LangGraph | Phase 2, not started |

### Deadline

The raw archiver is the only component where delay causes **permanent,
unrecoverable loss**. `status`, `chance_of_playing_next_round` and `news` are
live state: the API reports them for today only, no endpoint returns them for a
past gameweek, and the community archive does not carry them. Verified against
`vaastav/Fantasy-Premier-League` 2025-26.

Every gameweek without an archiver is a gameweek of the only forward-looking
data in the API, gone. It is one HTTP call, a gzip and a manifest line.

---

## Files

| File | Purpose |
|---|---|
| `build_spec_minutes_model.md` | **The main spec.** P(starts) and expected minutes, in full |
| `fpl_starts_analysis.ipynb` | Reproduces every measured claim below |

---

## What has actually been measured

All from 2025-26, walk-forward, 27 folds, leak-free strata. Reproducible in the
notebook.

### The model is a lookup table

Two features, both walk-forward: `prev` (started last gameweek?) and `roll4`
(share of the last four started, binned). Cross them, take the observed
frequency in each cell. Seven cells. No regression, no fitting.

| Model | Pool | Core | **Rotation** | Marginal | Deep |
|---|---|---|---|---|---|
| Constant (base rate) | 0.2009 | 0.3768 | 0.2340 | 0.1461 | 0.0883 |
| Persistence | 0.1050 | 0.1831 | 0.2150 | 0.1378 | 0.0106 |
| Calibrated (`prev` only) | 0.0977 | 0.1718 | 0.1908 | 0.1220 | 0.0136 |
| **Lookup (`prev` × `roll4`)** | **0.0885** | **0.1555** | **0.1798** | **0.1084** | 0.0100 |

Brier score, lower better. Beats persistence in **96% of individual gameweeks**.

**Anything built later must beat 0.1798 on Rotation** — not persistence, not a
pool-wide number.

### Pool-wide scores are misleading

Deep players (never start) are 41% of rows and trivially predictable at 0.0106.
They dominate any pool average.

Simulating a 25% error reduction on Rotation alone moves the pool score by only
**12.8%**. A model could get materially better at the only job that matters and
the headline metric would barely register it.

**Rotation-stratum Brier is the headline metric. Never report a single pool
number.**

### Almost all error is transitions

11.2% of predictions are wrong, and they account for **97.8%** of total Brier.
The other 88.8% contribute 2.2%. The prediction problem is entirely *when a
player's start state flips*, not what it currently is.

| Previous gameweek | P(starts this week) |
|---|---|
| Started | 0.799 |
| Benched | 0.077 |

Persistence predicts 0.95 / 0.05. It is not wrong about direction, only
overconfident — and Brier charges for that.

### Persistence is weak where it matters

Against a constant baseline, persistence improves **51% on Core, 88% on Deep,
and only 8% on Rotation**. Nearly all its apparent skill comes from easy
players. On the ones that drive transfer decisions, knowing what they did last
week is worth very little.

### Beta-Binomial adds nothing

Tested. Shrinking each player's own start rate toward their lookup-cell rate,
performance improves monotonically as prior strength rises and converges on the
pure lookup. **Optimal weight on player-specific history is approximately
zero** — `prev` and `roll4` already carry that signal.

---

## The layer model

The organising idea, inherited from an earlier process spec. Referenced
throughout, and the squad validator reports against it directly.

| Layer | Nature | Owner | On error |
|---|---|---|---|
| **A** | Deterministic facts — rules, squad legality, scoring constants, data integrity | Code | **Blocks.** Exit non-zero, do not submit |
| **B** | Probabilistic judgment from evidence — P(starts), team news, minutes | Model + LLM | Scored for calibration, never blocks |
| **C** | Decisions under uncertainty — captain, hits, chips, risk posture | Human | Signed off, logged |

The distinction is what each error deserves. A Layer A error is a bug and must
stop the pipeline. A Layer B error is a forecast that was wrong, which is
expected and measured rather than prevented. Conflating them produces either a
validator that blocks on uncertainty or a model that silently ships illegal
squads.

### Four operating rules that survived

**Provenance.** Trace every claim to origin; never count headlines. A manager's
own words outrank an aggregator — verified in GW2, when an aggregator listed
Elliot Anderson unavailable while Maresca said he was fully fit. Check article
dates against current club membership: a search result headlining Haaland as
doubtful turned out to reference Sheffield United and Bentancur, several
seasons stale.

**Disconfirmation.** For each holding, actively search for the evidence that
would contradict it, rather than for confirmation. Searching a player's name
for bad news is the check that failed in GW2 — a fit rival returning is good
news about someone you don't own, so it can never appear that way.

**Decision log.** Every gameweek, before the deadline: the decision, the
projections it rested on, the sources, and what would have changed it. This is
what makes "auditable" true rather than aspirational, and it is the raw
material the conversational agent explains from.

**Bench order.** Sort by descending P(appear) × projected points. No
kickoff-time adjustment — auto-substitutions process after the whole gameweek,
not per fixture.

---

## Architecture

```
                    ┌─────────────────────────────────┐
   AIRFLOW          │ archive → model → score → DB    │
   (batch)          │ scrape  → extract (LLM) ────────┤
                    └────────────────┬────────────────┘
                                     │  read-only
                    ┌────────────────▼────────────────┐
   LANGGRAPH        │ conversational agent            │
   (interactive)    │ tools: projections, evidence,   │
                    │        calibration, optimiser   │
                    └─────────────────────────────────┘
```

### Boundaries that must hold

**The agent has `SELECT` only on domain tables.** Enforce with a database role,
not convention. If the agent can write projections, you lose the ability to
distinguish what the model predicted from what the conversation talked it into,
and calibration becomes meaningless.

**Two stores.** `fpl` (Postgres, Airflow writes / agent reads) and LangGraph's
checkpoint tables (conversation state). Neither writes to the other's.

**The agent does not pick the team.** It calls the optimiser and explains the
result. An LLM asked to choose 11 from 15 under formation and club constraints
will occasionally violate one, quietly and plausibly. A MIP will not.

**Airflow owns execution state, the archive owns domain state.** Never put
"have I scraped GW3" in an Airflow Variable or XCom. Airflow should be able to
lose its metadata DB entirely and the pipeline still know where it is.

**Every numeric claim the agent makes comes from a tool result, and it says
which.** If no tool returned it, the agent says it does not know. A fluent
explanation of a wrong number is worse than a bare wrong number.

### Where the LLM is used

Exactly two places:

1. **News extraction** — reading scraped markdown, emitting P(starts) JSON. One
   stateless call per club, ~20 a week, inside an Airflow task. No graph, no
   memory, no tools. Not LangGraph.
2. **The conversational agent** — LangGraph, phase 2.

Everything else is deterministic: fetching, feature construction, the lookup
table, transfer detection, optimisation, validation, scoring.

---

## API gotchas, all verified against live responses

| # | Gotcha |
|---|---|
| 1 | `history` contains **unplayed** fixtures with zeroed performance fields. Filter on `team_h_score is not None` or every rate is biased low, silently. |
| 2 | The same rows have **valid** `value`, `selected` and `transfers_*` before kickoff. The filter must be field-specific, not row-level. |
| 3 | `element-summary.fixtures` **shrinks** — completed fixtures are removed. A snapshot cannot be reconstructed later. |
| 4 | `starts` does not exist before 2022/23. `defensive_contribution` per-gameweek does not exist before **2025/26**, despite `history_past` reporting a 2024/25 total. |
| 5 | `starts` + `minutes` together disambiguate: `1, 60` is started-and-hooked; `0, 60` is came-on-early. |
| 6 | Only Premier League fixtures appear. Cup and European matches are invisible, so congestion is understated for exactly the clubs where it matters. |
| 7 | Player **ids are reassigned each season**. Join multi-season data on `code`. |
| 8 | The API holds the **current season only**. Past seasons are unavailable at any price. |
| 9 | `bootstrap-static` gives season totals, not a time series. Use `event/{gw}/live` — one call per gameweek for the whole pool. |
| 10 | Gate post-gameweek jobs on `data_checked`, not `finished`. Bonus is provisional in between. |

---

## Decisions reversed during design

Recorded because the reasoning matters more than the conclusions, and because
several were confidently wrong before being checked.

**Auto-substitutions.** Claimed a Monday-fixture player could not be
auto-subbed by an earlier game. Wrong — subs process after the whole gameweek
completes. A late fixture costs you *information timing*, which belongs in
P(starts), not bench ordering.

**Three-per-club limit.** An early validator treated four from one club as a
blocking failure. A real-world transfer can legally leave you with four; all of
them score, and only the *next transfer* is constrained. That check was
blocking legal squads. Must warn on observed squads, block on constructed ones.

**Beta-Binomial.** Specified, then tested, then removed. See above.

**Displacement heuristic.** A fit club-mate priced within 85% with under half
the minutes. Invented thresholds, never tested, and a proxy for something the
evidence layer states outright. Removed in favour of scraped team news.

**Shortlist-only overrides.** Specified on the grounds that hand-checking 620
players is infeasible. True of manual reading, irrelevant once lineups are
scraped — a predicted XI covers a club at a time, so 20 scrapes cover the pool.

**T-72h projections.** Removed. The lookup's features are fixed once the
previous gameweek completes, so an early projection returns an identical number
while being worse informed on availability. Replaced by a daily dry run whose
output is discarded. This drops the earlier pre-commitment control — a
provisional lock 72 hours out that required a specific new fact to overturn.
The substitute is requiring any late change to cite its evidence, which is a
weaker guard against reactive churn; if that becomes a problem, cap changes per
gameweek rather than reinstating the early projection.

**Evaluation leakage.** Calibration was fitted on GW6–22 and scored across
GW6–38. The result survived a clean redo, but only because the model had two
parameters. A subtler leak — strata labelled from full-season rates — was
contaminating the stratum of interest. Both fixed; see spec §8.0.

---

## A finding the model does not capture

Both GW2 failures clustered by **price band**, not by chance. Wilson 5.7%
owned, Senesi 7.6%, Anderson 6.1% — all £6.0–6.5m.

That band is *defined* by minutes uncertainty. Genuinely nailed-on players get
owned, so cheap plus low-owned usually means the market has already priced in
doubt about starts. Rotation is also correlated **within** a club, so holding
three players at one club amplifies minutes risk as well as fixture risk.

This is a squad-construction problem, not a research one, and no amount of
better P(starts) estimation fixes it. It belongs in the optimiser as a
minutes-security floor for XI slots, pushing cheap differentials to the bench
where a blank costs an auto-sub rather than a zero.

Not yet implemented, and not yet measured.

---

## Component requirements

Details that are easy to get wrong and must survive reimplementation.

### Squad validator

Layer A. Exit **0** pass, **1** rule failure (do not submit), **2** data
untrustworthy.

- **Affordability uses selling price, never `now_cost`.** FPL returns half of
  any profit, rounded down to the nearest 0.1: if `now_cost > purchase_price`,
  the sell value is `purchase_price + (now_cost - purchase_price) // 2`,
  in tenths. Confirmed live: `transfers_sell_on_fee: 0.5`,
  `element_sell_at_purchase_price: false`. Getting this wrong overstates the
  budget for every player whose price has risen, and it does so quietly.
- **No purchase ledger, no budget check.** The public picks endpoint does not
  carry purchase prices — only `/my-team/{id}/`, which needs a session cookie.
  Falling back to `now_cost` is valid *only* for a brand-new squad and must
  warn loudly.
- **Assert the scoring constants from `game_config.scoring` and
  `game_config.rules` on every run**, and fail closed on drift. This converts
  the annual manual rules re-read into a machine check.
- **Fail closed on a stale snapshot.** If the snapshot's `is_next` deadline has
  already passed, refuse to validate. Running against last week's prices is
  worse than not running.
- **Club limit is observed-versus-constructed.** Four from one club is legal in
  a real squad (see Decisions reversed) — warn, and note the next transfer is
  earmarked. In a *constructed* squad from the optimiser it is a hard failure,
  because that move is illegal to make.
- Availability flags are **warnings**, never blockers. Layer B decides.

### News scraper

- **Discover URLs by matching both club names.** Article URLs change weekly, so
  scrape each outlet's index page and keep links whose text names *both* clubs
  in a fixture. Requiring both is what rejects "Leeds transfer latest" and
  "Brentford injury update" while keeping "Leeds vs Brentford: predicted
  lineups". The fixture list gives you the club names for free.
- **Fetch markdown and archive it. Do not extract inline.** Extraction is a
  separate process reading the archive, so an improved prompt can be re-run
  over past gameweeks.
- **Empty or near-empty responses must raise.** A blank page is otherwise
  indistinguishable from "nobody starts". A club with no usable scrape falls
  back to the lookup value and is flagged by name.
- **Idempotent within a window.** Skip refetching the same outlet and fixture
  within ~6 hours, so an hourly scheduler is safe while a genuine second pass
  at T-3h still runs.

---

## Known gaps

- **Cross-season transfer untested.** Every number here is 2025-26. Rerun the
  notebook with `SEASON = "2024-25"` to check the lookup rates hold.
- **Cold start untested.** Fit and test share players, so nothing here says
  anything about new signings. Needs a player-level holdout.
- **Availability contamination.** The Rotation stratum divides starts by
  gameweeks *registered*, not *available*, so roughly a quarter of it is
  injured regulars. Unfixable retrospectively; fixed by archiving from now on.
- **No market data.** Market-implied clean-sheet probabilities (Betfair) would
  be the largest single term in a points projection and are not wired up.
  `expected_goals_conceded` in `element-summary` is a free proxy worth testing
  first.
- **Purchase ledger.** Without it, budget is checked on `now_cost` and is wrong
  for any risen player.

---

## Build order

1. **Raw archiver.** Daily `bootstrap-static` and `fixtures`, gzipped, manifest
   line. ~22 MB a season. The only step with a deadline.
2. **Derived layer.** Postgres, built *from the archive*, never from a live
   fetch. Must be reproducible by replaying the archive.
3. **Backfill.** One `event/{gw}/live` call per completed gameweek.
4. **Incremental updater**, with the season-total cross-check as a fail-closed
   assertion.
5. **Features**, with tests for gotchas 1–5.
6. **Calibration harness** and baselines, stratified and walk-forward.
7. **P(starts)** across the full pool.
8. **Expected minutes.**
9. **Evidence layer**: scrape, extract, merge, score by tier.
10. **Optimiser** wired as a tool.
11. **LangGraph agent.**

Steps 1–6 before any modelling. The scoreboard first is what makes 7 onward
measurable rather than plausible.

---

## Honest limits

None of this changes outcome variance. A gameweek is mostly noise and a season
is not many gameweeks. What the process gives you is a Layer A error rate of
zero, a measured Rotation Brier, and every decision traceable to the data that
produced it.

The defensible claim is "an auditable, error-correcting, calibration-scored
selection process" — not that it wins. The market layer, the one component that
would bring projections level with good public services, is still missing.
