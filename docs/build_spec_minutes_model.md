# Build spec: P(starts) and expected minutes

Instructions for Claude Code. Build these two models as a standalone module.

Target: Python 3.7 (conda). No walrus operator, no `X | Y` type unions, no
f-string `=` specifier.

---

## 1. Objective

For each player `i` and gameweek `g`, produce:

```
P_start[i,g]     probability the player is in the starting XI
P_appear[i,g]    probability of any minutes at all
P_60[i,g]        probability of 60+ minutes
E_minutes[i,g]   expected minutes
```

`P_60` is not decoration. FPL pays 1 point under 60 minutes and 2 at 60 or
more, and clean sheet points require 60. **Do not model expected minutes as a
point estimate and then threshold it.** E[f(X)] is not f(E[X]): a player who
plays 90 half the time and 0 half the time has E[minutes] of 45 and a P_60 of
0.5, not 0. Model the distribution.

---

## 1a. Scope — all players, not just the squad

The model runs over the **entire player pool** (~620), not the 15 in the
current squad. Two reasons:

- **Validation.** 15 players over 2 gameweeks is 30 observations, far too few
  to measure calibration. The full pool gives ~1,200 per gameweek, enough for
  a reliability curve by decile within a few weeks.
- **Transfers.** You cannot evaluate a transfer target whose minutes you have
  not modelled.

This introduces one serious trap, described in §8b. Read it before writing
the scorer.

---

## 2. Data sources — verified against the live API

### 2.1 `/api/bootstrap-static/`

~620 players in `elements`. Fields to use:

| Field | Use |
|---|---|
| `status` | `a` available, `d` doubtful, `i` injured, `s` suspended, `u` unavailable |
| `chance_of_playing_next_round` | null, 0, 25, 50, 75, 100 |
| `news`, `news_added` | flag text and timestamp |
| `minutes`, `starts` | season totals |
| `element_type`, `team`, `now_cost` | position, club, price |
| `selected_by_percent` | ownership |

Also `events` (deadlines, `finished`, `is_next`) and `teams`.

### 2.2 `/api/element-summary/{player_id}/`

**This is the primary source and is not currently being pulled.** Returns three
arrays.

`history` — one row per gameweek, this season. The important fields:

```
round, minutes, starts, total_points, opponent_team, was_home,
kickoff_time, team_h_score, team_a_score, bps,
defensive_contribution, clearances_blocks_interceptions, recoveries, tackles,
expected_goals, expected_assists, expected_goal_involvements,
expected_goals_conceded, value, selected,
transfers_in, transfers_out, transfers_balance
```

`fixtures` — upcoming fixtures: `event`, `kickoff_time`, `is_home`,
`difficulty`, `team_h`, `team_a`, `provisional_start_time`.

`history_past` — one row per prior season: `season_name`, `minutes`, `starts`,
`total_points`, `start_cost`, `end_cost`, plus aggregates.

### 2.3 Past seasons — NOT available from this API

The FPL API holds **the current season only**. `event/{gw}/live` has no season
parameter; `event/1/live` means gameweek 1 of the season in progress. When the
next season starts, everything pulled this season is gone.

**Implication: archive aggressively.** Combined with §3.7, this means anything
not written to local storage this season is unrecoverable. Snapshot every
gameweek to durable storage as a matter of course, not as an optimisation.

**The availability fields are the highest-value thing you can archive, and the
only source of truth for them is a snapshot taken at the time.**

`status`, `chance_of_playing_this_round`, `chance_of_playing_next_round`,
`news` and `news_added` are **live state, not history**. `bootstrap-static`
reports whether a player is injured *today*. There is no endpoint that returns
their status in a past gameweek, and the community archive does not carry one
either — verified against `vaastav/Fantasy-Premier-League` 2025-26, where the
per-gameweek files contain none of these fields and `players_raw.csv` holds a
single end-of-season snapshot.

This is the reason the availability contamination in §8b cannot be fixed
retrospectively. The Rotation stratum divides `starts` by gameweeks
*registered* rather than gameweeks *available*, so a first-choice starter who
missed a spell injured is misclassified as a rotation risk. Measured on
2025-26, roughly a quarter of the Rotation stratum is this contamination.

**Required, from the next gameweek onward:** persist a row per player per
gameweek at the deadline, containing at minimum

```
player_id, code, gameweek, deadline_time,
status, chance_of_playing_this_round, chance_of_playing_next_round,
news, news_added, now_cost, selected_by_percent
```

Two reasons this matters beyond fixing the stratum labels:

- It gives **real availability ground truth**, which is what lets the §4.1a
  flag table be measured rather than assumed. Without it, the mapping from a
  50 flag to a probability of starting is guesswork.
- Unlike any retrospective proxy, it is known **at prediction time**, so it can
  be a model feature and not merely an evaluation label. Retrospective
  reconstructions (for example, inferring an injury from a run of zero-minute
  gameweeks bounded by starts on either side) require knowing that the player
  returned, which is not available when the prediction is made and cannot be
  used as a feature.

Every gameweek this is not archived is a gameweek of availability data that
cannot be recovered later at any price.

### 2.3a Archive format — raw responses, not parsed extracts

**Store the complete unmodified API response. Treat the parsed layer as
disposable and always rebuildable from it.**

The §2.3 field list above is a best guess at what matters. It will be wrong.
This document has already had to add `expected_goals_conceded` as a free
clean-sheet proxy, `value` and `selected` as valid in unplayed rows, and the
`starts` + `minutes` pair for detecting truncated starts — none of which were
in the first draft. A parsed archive freezes today's guess; a raw archive does
not.

Raw archiving also survives schema changes. FPL adds fields between seasons
(`starts` in 2022/23, `defensive_contribution` in 2025/26). A raw capture picks
them up with no code change; a parsed extract silently drops them.

**Cost is not a consideration.** A `bootstrap-static`-shaped payload is about
2.1 MB, gzipping roughly 10.7× to 0.2 MB. Three snapshots a day across a
38-gameweek season is **~22 MB**. Ten seasons is ~225 MB.

#### Two layers

| Layer | Contents | Mutable | Rule |
|---|---|---|---|
| **Raw** | Unmodified gzipped JSON, one file per fetch | **Write-once, never edited or deleted** | The source of truth |
| **Derived** | Parquet or SQLite for querying | Rebuilt freely | Must be reconstructible from raw alone |

If a parsing bug is found, the derived layer is rebuilt. With a parsed-only
archive the bug is permanent, and its effect on every past estimate is
unknowable.

#### Naming

```
raw/{season}/{endpoint}/gw{NN}/{YYYYMMDD}T{HHMM}Z.json.gz
```

for example

```
raw/2026-27/bootstrap-static/gw03/20260904T1200Z.json.gz
raw/2026-27/fixtures/gw03/20260904T1200Z.json.gz
raw/2026-27/event-live/gw02/20260901T0800Z.json.gz
```

`gw{NN}` is **the gameweek that was `is_next` at fetch time** — the gameweek
the snapshot informs. Document this, because it is genuinely ambiguous: a
snapshot taken after GW3 finishes lands in `gw04`. Record both `is_next` and
`is_current` in the manifest so the convention never has to be inferred.

#### Manifest

One append-only line per fetch, in JSONL:

```json
{"path": "raw/2026-27/bootstrap-static/gw03/20260904T1200Z.json.gz",
 "endpoint": "bootstrap-static", "fetched_at": "2026-09-04T12:00:03Z",
 "next_gw": 3, "current_gw": 2, "next_deadline": "2026-09-05T17:30:00Z",
 "http_status": 200, "bytes_raw": 2118442, "sha256": "…"}
```

The hash lets you detect corruption and skip storing byte-identical
consecutive snapshots. The timestamp matters more than anything else in the
file: these endpoints are **live state**, so a snapshot without a reliable
capture time is close to worthless.

#### What to archive, and how often

| Endpoint | Frequency | Why |
|---|---|---|
| `bootstrap-static` | **daily**, plus T-3h | Availability fields change through the week |
| `fixtures` | daily | Mutates as gameweeks complete (§3.7) |
| `event/{gw}/live` | once, on `data_checked` | Final gameweek results (§8a) |
| `element-summary` | not archived | Redundant given the above; 620 calls |

**Daily, not weekly.** `chance_of_playing_next_round` moves during the week —
a player may go 25 → 50 → 75 as they recover. A single deadline snapshot
records 75 and loses the trajectory, and the trajectory is plausibly more
predictive than the level: improving and deteriorating both read as 50 on the
Wednesday.

This costs nothing extra operationally, because the daily pipeline dry run
(§4.3a) already makes the call. Have it write its response to the archive
before discarding the projections.


**`history_past` is aggregates only, and survivorship-biased.** It gives
per-season totals with no gameweek breakdown, and exists only for players
*currently in the game*. Players who lost their place and dropped out of the
Premier League are absent entirely. Start-rate priors built from
`history_past` therefore see only survivors and are biased **upward** — the
direction that produces overconfidence about minutes. If used, widen the
prior variance to compensate and record the bias in the model card.

**For genuine multi-season per-gameweek data**, use the community archive
`vaastav/Fantasy-Premier-League`. Verified coverage: 2016-17 through 2026-27.
The per-gameweek CSVs under `data/{season}/gws/gw{N}.csv` carry a schema
matching `element-summary.history`, including `defensive_contribution`,
`expected_goals_conceded` and `starts`.

**Join on `code`, never on `id`.** Player IDs are reassigned each season.
Confirmed in the live payload: the player at `id: 4` carries
`element_code: 226597` in `history_past`. `code` is stable across seasons;
`id` is not. Joining multi-season data on `id` will silently attribute one
player's history to a different player.

Note the §3.3 caveat compounds here: `starts` is absent before 2022/23 and
`defensive_contribution` before 2024/25, so the usable window for training a
minutes model is 2022/23 onward, and for anything DefCon-related, 2024/25
onward.

### 2.4 `/api/fixtures/`

All fixtures with `kickoff_time`. Needed for gaps between a club's matches.
Fetch bare and filter client-side — the `?event=N` query string is unreliable
and returns 404 in some clients.

### 2.5 `/api/event/{gw}/live/` — the endpoint that makes full-pool scope cheap

Returns an `elements` array containing **every player's stats for one
gameweek** in a single request. This is the inverse shape of
`element-summary`, which gives one player across all gameweeks.

**Three endpoints, three shapes.** The model needs the full player-by-gameweek
matrix. `bootstrap-static` gives only the current-season row margins:

| Endpoint | Shape | Calls |
|---|---|---|
| `bootstrap-static` | all players × current snapshot | 1 |
| `element-summary/{id}` | one player × all gameweeks | 1 per player |
| `event/{gw}/live` | all players × one gameweek | 1 per gameweek |

`bootstrap-static` reports `minutes: 90, starts: 1` for the season. It cannot
tell you whether the player started GW1 and was benched in GW2 or the reverse,
and that distinction is the entire rotation signal.

**Fetch budget — much smaller than it looks:**

| Job | Calls | When |
|---|---|---|
| Current-season backfill | **1 per completed gameweek** (2 today) | once |
| Weekly update | **1** | per gameweek |
| Prior-season history (cold-start priors) | 620 `element-summary` | once, **optional** |

The 620-call job is needed only for `history_past`, which §4.2b shows is not
required for the main model. It remains potentially useful for cold-start
players with no current-season history (§8b). Ship without it.

**Alternative, not recommended as primary.** Snapshotting `bootstrap-static`
weekly and differencing consecutive season totals also recovers per-gameweek
values, at one call per week. But a single missed week silently merges two
gameweeks, and any retrospective correction by FPL breaks the arithmetic with
no error. Use `event/live` for values and keep bootstrap snapshots for the
cross-check below.

**Cross-check on each update:** `bootstrap-static` carries season-total
`minutes` and `starts` per player. After appending a gameweek, assert your
accumulated totals match. Any mismatch means a missed or double-counted
gameweek and must fail the run.

---

## 3. Gotchas found by inspecting the payload

**These are not hypothetical. Each was observed in the live response.**

### 3.1 `history` contains unplayed fixtures — and they are PARTIALLY populated

Confirmed by fetching the same endpoint before and after GW2 completed.

**Before the deadline**, player 4's `round: 2` row read:

```
minutes: 0, starts: 0, total_points: 0, bps: 0, team_h_score: null,
value: 80, selected: 2878770, transfers_in: 58765, transfers_out: 124108
```

**After the gameweek finished**, the same row read:

```
minutes: 90, starts: 1, total_points: 8, bps: 22, team_h_score: 0,
value: 80, selected: 2878770, transfers_in: 58765, transfers_out: 124108
```

Two things follow, and the second is easy to miss.

**The row exists before the match is played.** Computing
`sum(starts) / len(history)` counts the upcoming gameweek as a benching and
biases every rate downward, silently. **Filter on `team_h_score is not None`
before aggregating any performance field.**

**The row is not uniformly empty.** `value`, `selected`, `transfers_in`,
`transfers_out` and `transfers_balance` were already final before kickoff and
did not change afterwards.

**Caveat on timing — not yet established.** Both observations above were taken
*after* the GW2 deadline. It is therefore known that these fields are stable
between deadline and kickoff, but **not** whether they accumulate during the
transfer window and freeze at the deadline. FPL surfaces live transfer counts
during the week, so a mid-week pull may return a partial `transfers_in` that
is indistinguishable from a final one.

Until tested, treat these fields as valid only **after the gameweek deadline
has passed**, not merely before kickoff. Those are different times for every
fixture after the first in a gameweek.

**Test to run before relying on this:** pull `element-summary` for one player
twice before a deadline, 24 hours apart, and compare `transfers_in`. If it
grows, the fields accumulate live and the pipeline must gate on the deadline
timestamp from `events[].deadline_time`.

So the filter must be field-specific, not row-level:

| Field group | Valid from | Safe pre-kickoff |
|---|---|---|
| `minutes`, `starts`, `total_points`, `bps`, all match stats | match completion | **No** |
| `value`, `selected`, `transfers_*` | deadline (unverified — see above) | **Only after deadline** |

Discarding the whole row throws away usable ownership and price signal. Write
unit tests for both halves.

### 3.2 `starts` does not exist before 2022/23

In `history_past`, the 2020/21 and 2021/22 rows show `starts: 0` alongside
1996 and 3063 minutes. The field was introduced later; zero means absent, not
observed. Only use `history_past` rows from 2022/23 onward for start-rate
priors, or derive a proxy from `minutes / 90`.

### 3.3 `defensive_contribution` — per-gameweek data begins 2025/26

**Corrected against the archive.** The API's `history_past` reports a 2024/25
DefCon total (159 for player code 226597), which suggests 2024/25 data exists.
It does not, at gameweek granularity. Verified in
`vaastav/Fantasy-Premier-League`:

| Season | `defensive_contribution` in per-GW files | in `players_raw` |
|---|---|---|
| 2024-25 | **No** | **No** |
| 2025-26 | Yes | Yes |
| 2026-27 | Yes | Yes |

The season total in `history_past` appears to be a retrospective backfill of
the raw stat; the per-gameweek series was never collected. **Usable window for
any per-gameweek DefCon modelling is 2025/26 onward** — one complete season
plus the current one. Treat any DefCon model as data-poor and say so in the
model card.

The scoring rework also changed the CBI-to-BPS conversion, so pre-2025/26
bonus data is contaminated for this purpose regardless.

### 3.4 `starts` and `minutes` together disambiguate

`starts: 1, minutes: 60` means started and hooked. `starts: 0, minutes: 60`
means came on early. Identical minutes, opposite meanings. Always use the pair.

### 3.5 Only Premier League fixtures appear

Cup and European matches are invisible to this API. A club playing Wednesday
in Europe shows a seven-day gap in `fixtures`. Congestion features built from
FPL fixtures alone will understate rotation risk for European clubs.

**Possible mitigation, unverified.** The GitHub dataset
`olbauday/FPL-Core-Insights` claims 2026/27 coverage of cup, friendly and
European fixtures aligned to official FPL player IDs. If that alignment holds
it closes this gap without a paid feed. Evaluate it before building congestion
features; if it does not hold, document the limitation rather than papering
over it.

### 3.6 Ownership and price history are free

`history` carries `value` and `selected` per gameweek. That gives price history
without a separate feed, and `transfers_balance` as a crowd-sentiment feature.

### 3.7 The `fixtures` array shrinks as gameweeks complete

Before GW2, player 4's `fixtures` array began at `event: 2`. After GW2
finished, it began at `event: 3` — the completed fixture was removed.

`element-summary.fixtures` is **future-only** and is not a stable key space.
Never join `history` to `fixtures` from the same payload to reconstruct
context; the played fixture will have vanished. Use `/api/fixtures/` for
anything historical, joining on the `fixture` id in `history`.

This also means a snapshot taken at time T cannot be reconstructed later from
a fresh pull. Archive each snapshot to disk if you want a reproducible
backtest.

### 3.8 `defensive_contribution` is pre-computed, and position-dependent

Player 4 (a defender) recorded `clearances_blocks_interceptions: 8`,
`tackles: 2`, `recoveries: 5`, and `defensive_contribution: 10`.

For defenders, DefCon is CBI plus tackles — recoveries are **excluded**. The
threshold is 10, which he hit exactly, and the 2 points show up in his total
(2 appearance + 4 clean sheet + 2 DefCon = 8).

For midfielders and forwards, recoveries **are** included and the threshold is
12. Do not compute this yourself from the components; use the supplied
`defensive_contribution` field and apply the position-appropriate threshold.

---

## 4. Model 1 — P(starts)

### 4.1 Structure — a gate, not a product

An earlier version specified:

```
P_start = P(available) * P(selected | available)
```

with `P(available)` read straight off `chance_of_playing_next_round` — a flag
of 50 meaning `P(available) = 0.50`. **That is wrong in two ways** and is
replaced by the routing below.

**Why the multiplicative form fails.** The availability fields measure
*availability*, not *selection*, and the two come apart completely at the
unflagged end. Haaland and a fourth-choice midfielder both carry
`status: "a"` and `chance: null`. The field cannot separate them because it is
not trying to — it answers "is this player injured or suspended?", not "will
the manager pick him?" Around 60% of the pool never starts and almost none of
them carry a flag, so `P(available)` is 1.0 for the great majority of players
and contributes nothing.

**Why the numbers must not be taken at face value.** FPL's 50 is a judgment
about *fitness*. A player who is a coin flip to be fit is also a player a
manager eases back from the bench, so `P(starts | chance = 50)` is very likely
well below 0.50. This is the same error as persistence's invented 0.95: use the
observed frequency, never the plausible-looking number (§4.2).

**Routing.** Each player takes exactly one path:

| Condition | Estimator |
|---|---|
| `status` in {`i`, `s`, `u`} | `P_start = 0` |
| `status == "d"`, or `chance_of_playing_next_round` present and < 100 | **flag table** (§4.1a) |
| otherwise | **lookup table** (§4.2), unchanged |

This plays to each source's strength. The flag table catches cases the start
history gets confidently wrong; the lookup handles the ~95% of players carrying
no flag, which the flag fields cannot discriminate between at all.

### 4.1a The flag table

**Worked example, GW3 2026/27.** Joe Rodon started GW1 and GW2, so `prev = 1`
and `roll4 = 1.0`, putting him in the lookup's strongest cell at **0.848**. He
went off after 34 minutes against Brentford with a hamstring injury; FPL flagged
him at **50**. The lookup is not merely uninformative here — it is confidently
wrong by 0.35 in the wrong direction, because his start history is that of a
completely nailed-on starter, which is exactly what he was until the 34th
minute.

Estimate `P(starts | flag)` **from observed frequency**, exactly as §4.2 does.
Cross the flag level with `prev` only — never `roll4` — because the flagged
population is small and the cells would not fill:

| `status` / `chance` | `prev` | P(starts) |
|---|---|---|
| `d` / 75 | 1 | measure |
| `d` / 75 | 0 | measure |
| `d` / 50 | 1 | measure |
| `d` / 50 | 0 | measure |
| `d` / 25 | any | measure |
| `a` / 0, or `d` / 0 | any | measure (expect ≈ 0) |

Fall back to the pooled "any flag" rate for cells under 50 observations, and to
the §4.2 lookup if even that is unavailable. **Never hard-code the percentages
as probabilities.**

**Test `status` and `chance` separately.** `status` is likely the more
informative of the two: `i`, `s` and `u` are unambiguous, whereas the graded
percentages are rare and noisy. In the 2025-26 season-end snapshot only **11 of
841** players sat at 25, 50 or 75, against 352 at 100 and 267 at 0. Also test
the presence of `news` as its own binary signal, since a player can carry
informative news text with a null `chance`.

**None of this is testable today.** There is no historical record of these
fields (§2.3), so every claim in this section is a hypothesis. Roughly ten
gameweeks of archiving gives enough flagged player-gameweeks to settle whether
a 50 flag means 0.50, 0.25 or something else. Until then, ship the routing with
provisional values, mark them `provisional: true` in the output, and replace
them with measured rates as soon as the data exists.

### 4.1b Truncated starts — derivable today

Rodon's GW2 row reads `starts = 1, minutes = 34`. A start cut well short of the
hour is a leading indicator available from data already held, needing no
archiving.

Add `short_start` — started but under 60 minutes — as a candidate feature and
test it as a fourth lookup dimension. It would not have flagged Rodon's injury
in advance, but "started and came off early" plausibly predicts both a
subsequent benching and a truncated next appearance. **Untested; keep only if
it beats 0.1798 on Rotation** (§4.2a).

### 4.2 The model: a conditional lookup table

Two features, both walk-forward:

- `prev` — did the player start the previous gameweek? (0/1)
- `roll4` — share of the previous 4 gameweeks started, binned into
  {0/4, 1/4, 2-3/4, 4/4}

Cross them to get seven populated cells (there is no "started last week and
0 of last 4"). For each cell, the prediction is the **observed frequency of
starting in that cell**, estimated on all prior gameweeks. Nothing else.

Measured on 2025-26, fitted GW6-22:

| `prev` | `roll4` | n | P(starts) |
|---|---|---|---|
| Yes | 1/4 | 389 | 0.591 |
| Yes | 2-3/4 | 539 | 0.688 |
| Yes | 4/4 | 3,694 | 0.848 |
| No | 0/4 | 9,453 | 0.041 |
| No | 1/4 | 770 | 0.232 |
| No | 2-3/4 | 554 | 0.269 |
| No | 4/4 | 371 | 0.447 |

Cells with fewer than 50 observations fall back to the `prev`-only rate
(0.806 / 0.079). Refit every gameweek on all prior data.

The bottom row is the case that matters: a player who did not start last week
but started all four before that sits near 0.45, not 0.05. That is the shape of
a regular who was dropped once — the GW2 Senesi case.

**Why the observed frequency is the right number.** For a group with true rate
`q`, expected Brier is `q(p-1)² + (1-q)p²`, minimised at `p = q`. Any deviation
from the observed frequency costs points. This is not a heuristic.

### 4.2a Measured performance

Walk-forward, 27 folds, leak-free strata, 2025-26. Improvement over a constant
base-rate prediction:

| Model | POOL | Core | **Rotation** | Marginal | Deep |
|---|---|---|---|---|---|
| Constant (base rate) | 0.2009 | 0.3768 | **0.2340** | 0.1461 | 0.0883 |
| Persistence | 0.1050 | 0.1831 | **0.2150** | 0.1378 | 0.0106 |
| Calibrated (`prev` only) | 0.0977 | 0.1718 | **0.1908** | 0.1220 | 0.0136 |
| **Lookup (`prev` × `roll4`)** | **0.0885** | **0.1555** | **0.1798** | **0.1084** | 0.0100 |

The lookup beats persistence in **96% of individual gameweeks**.

Note persistence beats a constant by only **8.1% on Rotation**, against 51%
on Core and 88% on Deep. Nearly all of its apparent skill comes from the easy
strata. On the players that actually drive transfer decisions, knowing what
they did last week is worth very little — which is precisely why the pool-wide
number must never be the headline (§8b).

### 4.2b Why not Beta-Binomial

An earlier version of this spec specified a Beta-Binomial with a per-player
rate shrunk toward a prior. **It was tested and adds nothing.**

Shrinking each player's own cumulative start rate toward their lookup-cell
rate, with prior strength `k`, walk-forward:

| Prior strength | POOL | Rotation |
|---|---|---|
| Player rate only (k=0) | 0.1102 | 0.2304 |
| k=3 | 0.1030 | 0.2139 |
| k=6 | 0.0985 | 0.2034 |
| k=12 | 0.0932 | 0.1911 |
| k=25 | 0.0885 | 0.1803 |
| Lookup only (k→∞) | 0.0887 | **0.1798** |

Performance improves monotonically as `k` rises, converging on the pure
lookup. **The optimal weight on player-specific history is approximately
zero.** `prev` and `roll4` already carry the player-specific signal; a
cumulative per-player rate adds no information beyond them.

The lookup table is itself a shrinkage estimator — every player in a cell gets
the pooled cell rate — so the early-season small-sample problem the
Beta-Binomial was meant to solve is already handled, non-parametrically.

**This does not prove no parametric model can win.** It proves this one does
not, and it sets the bar: any richer model must beat **0.1798 on Rotation**,
not persistence's 0.2150. Report the comparison on identical walk-forward
folds per §8.0.

### 4.3 Adjustments to the lookup output

Only two, both derived from data the API supplies directly:

- **Congestion.** Gap to the club's previous fixture under four days, from
  `/api/fixtures/`. Position-dependent: goalkeepers rotate far less than
  outfield players. **Untested** — add as a third lookup dimension (a binary
  flag crossed with `prev` and `roll4`), not a multiplier, and keep it only if
  it beats 0.1798 on Rotation. Note §3.5: cup and European fixtures are
  invisible to this API, so congestion is understated for exactly the clubs
  where it bites hardest.
- **Recent booking risk.** Not a selection signal; relevant to expected
  minutes only (§5).

**No displacement adjustment.** An earlier version inferred a rival's threat
from price and minutes — a fit club-mate in the same position, priced at 85%
or more, playing under half the minutes. That is a proxy for something the
evidence layer states outright. Van de Ven's return had to be *inferred* from
his price and zero minutes; a predicted lineup simply names him. The proxy's
thresholds were invented, it was never tested, and it would have added an
unmeasured multiplier to a measured model. Use §4.4 instead.

If an early-week signal is wanted later, add it as a lookup dimension and
measure it. Do not add it as a multiplier.

### 4.3a When the pipeline runs

**All decision analysis runs as late as practical. There is no T-72h
projection.**

The lookup output is *invariant* across the week: `prev` and `roll4` are both
fixed once the previous gameweek is verified, so the model returns an identical
number at T-72h and T-3h. Only two inputs move during the week — the API
availability flags and scraped lineups — and both improve monotonically toward
the deadline. An early projection is therefore strictly dominated by a late one
and should not be produced, because a stale number sitting in a file invites
someone to act on it.

| Stage | Runs | Purpose |
|---|---|---|
| Post-gameweek, on `data_checked` | scoring, refit, archive | §8a |
| Daily | pipeline **dry run**, output discarded | operational only |
| T-24h | scrape pass 1, projections | first usable output |
| T-3h | scrape pass 2, final projections, log | the decision |

**The daily dry run is not analysis.** It exists so that a broken scraper,
expired token or schema change surfaces with days of slack rather than at T-3h
with the deadline approaching. Run the full pipeline, assert it completes,
discard the projections. Alert on failure, not on output.

**Note for the operator.** An earlier version of this process used a T-72h
provisional lock as a behavioural control against reactive churn — decide
early, then require a specific new fact to change the decision. Removing the
T-72h stage removes that control. The information argument for dropping it is
sound, but the pre-commitment function needs replacing rather than deleting.
The substitute is to require that any change made at T-3h cite the evidence
justifying it (§4.4 records source and timestamp on every claim), which gets
the same discipline without a stale projection.

### 4.4 Evidence layer — scraped team news

This is the dominant signal and it overrides the lookup output. It applies to
the **whole pool**, not a shortlist: a predicted XI covers a club at a time, so
20 scrapes cover all 620 players.

**Sources**, in descending authority per the provenance rules:

1. Manager's own words in a press conference
2. Predicted XIs from named outlets
3. Aggregators and prediction sites

**Design requirements:**

- **Two independent outlets minimum per club.** Disagreement is the finding,
  not a nuisance to resolve. In GW2 2026/27, sources split on Senesi; the
  honest number was near 0.5, not a confident call either way.
- **Every outlet is a forecaster with a measured reliability.** Do not assume
  "outlet predicts start" means 0.95. Log the prediction, score it against the
  confirmed XI, and use the **observed** hit rate — the same logic as §4.2.
  After a few gameweeks each outlet has its own calibrated number, and they
  will differ.
- **Presence and absence are not symmetric.** A predicted XI names 11 players.
  It asserts those 11 start; it does not assert the other 20 will not. Estimate
  P(starts | named) and P(starts | not named, club covered) separately, and
  measure both.
- **Staleness checking is mandatory.** In GW2 a search returned an article
  headlining the captain as doubtful; it referenced clubs and players from
  several seasons earlier. Reject any article whose named players contradict
  current club membership from `bootstrap-static`.
- **Record source and timestamp on every claim.** Unsourced values are
  rejected by the validator (§7).

**Timing.** Predicted XIs typically appear 24–48 hours before kickoff, and
press conferences on the Friday. For a Friday-evening deadline some material
lands after lockout. Scrape at T-24h and again at T-3h, and keep both so the
value of the later pass can be measured.

**Confirmed XIs** publish about an hour before kickoff, after the deadline.
Useless for the decision, essential for scoring — this is the ground truth
that calibrates every outlet and the lookup table itself.

**Output**, merged over the lookup prediction:

```json
{"gameweek": 3, "generated_at": "2026-09-05T18:00:00Z", "evidence": [
  {"player_id": 498, "p_start": 0.35, "tier": "scraped",
   "sources": [{"outlet": "sportsmole", "ts": "2026-09-05T11:00:00Z", "predicted": "bench"},
               {"outlet": "si", "ts": "2026-09-05T09:30:00Z", "predicted": "bench"}],
   "note": "Van de Ven fit; both outlets predict Senesi benched"}
]}
```

Every prediction carries `tier` — `lookup` or `scraped` — and the scorer
reports Brier separately for each on the same players. **If the scraped layer
does not beat the lookup on the players it covers, it is not earning its
place**, and you would want to know that rather than assume it.

**Practical warnings.** Scraping is brittle: layouts change, and terms of
service vary by site — check them. Build the scraper to fail loudly rather than
returning an empty XI that silently reads as "nobody starts". A club with no
usable scrape must fall back to the lookup value and be flagged, never
defaulted to zero.

---

## 5. Model 2 — expected minutes

### 5.1 Three-state distribution

Do not emit a single number. Emit a distribution:

```
P_0      no minutes
P_1_59   appeared, under 60
P_60     60 or more
```

with `P_0 + P_1_59 + P_60 = 1`.

### 5.2 Construction

Condition on whether the player starts:

```
P_60    = P_start * P(60+ | started) + (1 - P_start) * P_sub_on * P(60+ | sub)
P_1_59  = P_start * P(<60 | started) + (1 - P_start) * P_sub_on * P(<60 | sub)
P_0     = 1 - P_60 - P_1_59
```

Estimate the conditional terms from the player's own `history`, shrunk toward
position-level averages:

- `P(60+ | started)` — high for most, materially lower for players routinely
  substituted around the hour. Compute from rows where `starts == 1`.
- `P_sub_on` — from rows where `starts == 0`: did they appear at all?
- `P(60+ | sub)` — low. Compute from rows where `starts == 0 and minutes > 0`.

`E_minutes` follows as a weighted average, using the observed mean within each
band rather than a band midpoint.

### 5.3 Substitutes matter more than they look

A player with `P_start` of 0.3 but a reliable 25 minutes off the bench is worth
more than one with the same `P_start` who never appears. `P_appear` is the
number that separates them, and it is the one that matters for bench ordering.

---

## 6. Output contract

Write `projections/minutes_gw{N}.json`:

```json
{
  "gameweek": 3,
  "generated_at": "2026-09-01T10:00:00Z",
  "bootstrap_snapshot": "sha256:...",
  "players": [
    {"player_id": 498, "web_name": "Senesi", "team": 19,
     "p_start": 0.35, "p_appear": 0.55,
     "estimator": "flag_table", "flag": {"status": "d", "chance": 50},
     "provisional": true,
     "p_60": 0.38, "e_minutes": 38.2,
     "prior_mean": 0.72, "observed_rate": 1.0, "n_observed": 1,
     "adjustments": {"congestion": 1.0},
     "tier": "scraped",
     "override_applied": true,
     "sources": ["sportsmole 2026-08-29", "si 2026-08-29"]}
  ]
}
```

`prior_mean`, `observed_rate` and `n_observed` must be present so any number
can be traced back to what produced it.

`estimator` records which path of §4.1 produced the value — `lookup`,
`flag_table` or `unavailable` — so the scorer can report Brier per estimator
and show whether the flag table is beating the lookup on flagged players.
`provisional: true` marks values from hard-coded flag rates that have not yet
been replaced by measured ones (§4.1a).

---

## 7. Validation

Code validates the model's own output and fails closed:

- All probabilities in [0, 1]
- `p_60 <= p_appear <= 1` and `p_start <= p_appear`
- Three-state probabilities sum to 1 within 1e-6
- `e_minutes` in [0, 90]
- Every player in the current squad has an entry
- Any override without sources is a hard error

---

## 8. Calibration harness — build this first

### 8.0 Validation protocol — non-negotiable

An earlier version of this analysis fitted calibration on GW6–22 and then
scored across GW6–38, so two thirds of the evaluation was on data the model
had already seen. The result happened to survive a clean redo, but only
because the model had two parameters and nothing to overfit. Any model with
real capacity would have flattered itself and given no warning.

**Rules:**

1. **Splits must be temporal, never random.** A random split lets the model
   see gameweek 30 while predicting gameweek 12. Rotation patterns are
   autocorrelated, so a random split leaks heavily and inflates every score.

2. **Strata labels must be derived from the fit period only.** Labelling a
   player "Rotation" using their full-season start rate encodes what happened
   during the test period. This is a subtle leak and it contaminates exactly
   the stratum that matters most.

3. **Walk-forward is the primary scheme.** For each gameweek, refit on all
   strictly prior gameweeks and predict that one. This mirrors production
   exactly and yields ~27 folds per season instead of one, giving variance
   estimates. A single holdout cannot tell you whether a 15% improvement is
   real or one lucky fortnight.

4. **Report mean, standard deviation and win rate across folds**, not just a
   mean. Measured for the reference implementation on 2025-26:

| Model | Mean | Std | Beats persistence in |
|---|---|---|---|
| Persistence 0.95/0.05 | 0.1050 | 0.0181 | — |
| Calibrated transition rates | 0.0977 | 0.0148 | — |
| Calibrated + rolling 4-GW rate | 0.0885 | 0.0126 | **96% of gameweeks** |

5. **Known residual limitation.** The same players appear in both fit and test
   periods. That is realistic — in production you predict players you have
   history for — but these numbers say **nothing** about cold-start
   performance on new signings. Evaluating that needs a separate player-level
   holdout: fit excluding a random 20% of players, then score only on them.
   Report separately; never merge into the headline figure.

### 8.1 Scoring loop

1. Before each deadline, snapshot `projections/minutes_gw{N}.json`.
2. After the gameweek, pull `history` and read actual `starts` and `minutes`.
3. Compute **Brier score** for `p_start`, and MAE for `e_minutes`.
4. Produce a **reliability curve**: bucket predictions into deciles and plot
   predicted against observed frequency. This is what reveals systematic
   overconfidence.

Compare against three baselines. The model must beat all three to justify its
existence:

- **Persistence** — predict the player starts iff they started last gameweek.
  Clip to [0.05, 0.95] rather than hard 0/1, since an unhedged wrong call
  scores a full 1.0 on Brier and overstates the penalty. This is the standard
  naive benchmark in forecasting ("tomorrow will be like today"). It requires
  no model and, as §8b shows, is hard to beat. **It is the real bar.**
- **Season rate** — predict the cumulative `starts / appearances` to date,
  computed walk-forward so it never sees the gameweek being predicted.
- **Always 0.9** — the naive optimistic guess, included to show what
  overconfidence costs.

If the model cannot beat persistence, say so plainly in the output rather than
shipping it.

---

## 8a. Production scheduling — when it is safe to pull

The pipeline runs after gameweeks complete. "Complete" needs defining
precisely, because there are three distinct states and only one is safe.

| State | Meaning | Safe to score against |
|---|---|---|
| All fixtures kicked off | matches in progress | **No** |
| `events[].finished` | matches over | **No — bonus is provisional** |
| `events[].data_checked` | FPL has verified the data | **Yes** |

Bonus points and some stats are provisional between final whistle and
verification. A job that fires on `finished` will capture provisional bonus
and then disagree with the record permanently, because nothing re-reads it.

**Gate the post-gameweek job on `data_checked`, not `finished`.** Verify the
field name against your bootstrap payload before relying on it; if it is
absent, poll until `bps` and `bonus` stop changing across two consecutive
pulls, and log which method was used.

Recommended schedule:

1. **Poll** `bootstrap-static` hourly after the last fixture. Do nothing until
   the target gameweek shows `data_checked`.
2. **Archive** the full snapshot to `snapshots/gw{N}_final.json`. Per §3.7 it
   cannot be reconstructed later.
3. **Score** logged predictions against actual `starts` and `minutes` (§8).
4. **Refit** base rates with the new gameweek included.
5. **Only then** generate projections for the next gameweek.

Steps 3 and 4 must run in that order. Refitting before scoring means the model
is evaluated on data it has already absorbed, and the calibration numbers
become meaningless.

Also archive the **pre-deadline** snapshot separately. Scoring needs the
prediction as it stood before kickoff, and §3.7 means you cannot recover it
afterwards.

---

## 8b. Full-pool validation — the class imbalance trap

**This section is backed by measurement, not argument.** Numbers below come
from the full 2025-26 season (`vaastav/Fantasy-Premier-League`, 841 players,
38 gameweeks, 29,757 player-gameweek rows).

### Observed stratum sizes

**First cut (as originally specified), 50% threshold:**

| Stratum | Definition | Players | Share of eval rows | Observed start rate |
|---|---|---|---|---|
| Core | started ≥50% of appearances | 223 | 27.7% | 0.746 |
| Fringe | started ≥1 but <50% | 251 | 31.0% | 0.239 |
| Deep | never started | 367 | 41.4% | 0.000 |

Pool-wide base rate of starting: **0.281**.

### Refinement — "Fringe" is three populations, not one

Decomposing the 251 Fringe players by start pattern (longest consecutive run
of starts, and number of transitions between starting and not):

| Profile | Count | Median starts | Median longest run | Median switches |
|---|---|---|---|---|
| Scattered — genuine rotation | 139 | 11 | 4 | 9 |
| Ambiguous | 88 | 2 | 1.5 | 2 |
| Clustered — likely injured regular | 24 | 11 | 7.5 | 3 |

Two problems with the single 50% cut:

- The **88 ambiguous** players average 2 starts in 38 gameweeks. They are Deep
  players who started once or twice, not rotation risks, and they dilute the
  stratum.
- The **24 clustered** players show long unbroken start runs. That is a
  first-choice starter who got injured. Predicting them is an **availability**
  problem, routed to the §4.1a flag table rather than the lookup.

**Use four strata instead:**

| Stratum | Start rate | Approx. players | Role in evaluation |
|---|---|---|---|
| Core | ≥ 0.50 | 223 | reported, easy |
| **Rotation** | 0.15 – 0.50 | ~139 | **headline metric** |
| Marginal | > 0, < 0.15 | ~88 | reported, near-trivial |
| Deep | 0 | 367 | reported, trivial |

**Caveats on these labels.** The 0.15 and 0.50 thresholds are chosen, not
derived. The labels are computed retrospectively from the full season, so they
are valid for grouping evaluation output but **must never be used as model
features** — they contain look-ahead. The denominator is gameweeks
*registered*, not gameweeks *available*, which is what allows injured regulars
to leak in. Measured on 2025-26, roughly a quarter of the Rotation stratum is
this contamination. It **cannot be fixed retrospectively** — the availability
fields are live state with no historical record (§2.3). The run-length split
above is a partial mitigation for reporting only; the real fix is to start
archiving `status` and `chance_of_playing_*` weekly, per §2.3, and reclassify
from real data once enough gameweeks have accumulated.

### Measured baseline Brier scores

Walk-forward, GW6–38, 26,063 player-gameweeks. Lower is better; 0.25 is a
coin flip.

| Baseline | Pool | Core | Fringe | Deep |
|---|---|---|---|---|
| Constant 0.281 | 0.2009 | 0.4052 | **0.1812** | 0.0790 |
| Constant 0.900 | 0.5873 | 0.2141 | 0.6233 | 0.8100 |
| Persistence (started last GW) | **0.1031** | 0.1776 | **0.1708** | 0.0025 |
| Cumulative prior start rate | 0.1077 | 0.1863 | 0.1809 | 0.0004 |

### What these numbers establish

**1. Persistence is a strong baseline and the real bar.** Pool Brier 0.1031.
Any model that cannot beat "did they start last week" is not worth shipping.

**2. Deep is 41% of rows and trivially predictable** (0.0025). It dominates
the pool average and makes every model look good.

**3. On Fringe, sophistication barely helps.** Persistence scores 0.1708
against a dumb constant's 0.1812 — a 6% improvement. **All the difficulty and
all the value sit in the stratum where current methods are weakest.** Senesi
was Fringe.

**4. Pool-wide Brier cannot detect improvement where it matters.** Simulating
a 25% error reduction on Fringe alone moves the pool score from 0.1031 to
0.0899 — a 12.8% change that is easily lost in noise. A model could get
materially better at the only job that matters and the headline metric would
barely register it.

**Therefore: never report a single pool-wide Brier number. Rotation-stratum
Brier is the headline metric.** Core, Marginal and Deep are reported for
completeness only.

Note the Fringe figures above are the pre-refinement 251-player grouping.
Recompute against the four-stratum split before using them as targets; the
Rotation-only numbers will be worse, because the easy Marginal players are no
longer averaged in. That is the point.

**5. Recency beats long-run averages.** Persistence (0.1031) outscores the
cumulative start rate (0.1077) pool-wide and on every stratum. This validates
the short recency window in §4.2 — `roll4` looks back only four gameweeks,
and lengthening it should be expected to hurt rather than help.

### Cold start

Players with no history at their current club need explicit handling, not a
silent fallback:

- **New signings.** `history` is empty or short; `history_past` may exist under
  a different club. Use `history_past` for the start-rate prior but flag
  `cold_start: true` in the output and widen the uncertainty.
- **Promoted-club players.** `history_past` may be from outside the Premier
  League and is absent from this API entirely.
- **Youth players.** No history anywhere. Fall back to the price-and-position
  prior, which will correctly say "probably does not start".

Every output row carries `cold_start` and `n_observed` so these are visible
rather than silently averaged in.

### Manual overrides do not scale to 620 — and should not

The evidence layer (§4.4) applies to the **whole pool**, not a shortlist. A
scraped predicted XI covers a club at a time, so 20 scrapes cover all 620
players. An earlier version of this spec restricted overrides to a shortlist on
the grounds that hand-checking 620 players is infeasible — true of manual
reading, irrelevant once the lineups are scraped.

Two tiers still exist, and the scorer must record which produced each
prediction:

- **`lookup`** — the §4.2 table. Applies to every player, always.
- **`scraped`** — §4.4 evidence, applied where a usable scrape exists for that
  club. Falls back to `lookup` and raises a flag where it does not.

**Report Brier separately by tier, on the players the scraped tier covers.** If
scraping does not beat the lookup on those players, it is not earning its
place. That comparison is the only way to find out, and it is also how each
outlet's individual reliability gets measured (§4.4).

---

## 9. Out of scope

Do not attempt to model these. They are not in the API and guessing at them
adds noise dressed as signal:

- Manager intent or press conference content
- Predicted lineups (no free API; enters via the override file only)
- Cup and European fixtures (invisible; see 3.5)
- Transfer-driven omissions — a player left out because a move is imminent
  keeps `status: "a"` and no `news`. Override file only.

---

## 10. Acceptance tests

1. `history` rows with `team_h_score is None` are excluded from all rates.
   Fixture: a player with one played and one unplayed row must show an
   appearance count of 1.
2. `history_past` rows before 2022/23 do not contribute to start priors.
3. A player with `starts: 1, minutes: 60` and one with `starts: 0, minutes: 60`
   produce different `p_start`.
4. A player with `status: "i"` produces `p_start == 0.0` regardless of history,
   and `estimator == "unavailable"`.
5. A player with one observed gameweek and a 1.0 start rate produces
   `p_start < 0.9` — the lookup cell rate, not the player's own rate, is being
   used (§4.2b).
6. An override without a `sources` field raises rather than being applied.
7. The three-state probabilities sum to 1 for every player in the pool.
8. A player with `chance_of_playing_next_round == 50` is routed to the flag
   table and does **not** receive the §4.2 lookup value. Fixture: a player with
   `prev=1, roll4=1.0` and a 50 flag must not return 0.848 (§4.1a, the Rodon
   case).
9. No flag percentage appears as a probability anywhere in the output. Asserting
   `p_start == 0.50` for a 50 flag is a failure, not a pass.
10. Every row carries `estimator`, and any row with a hard-coded flag rate
    carries `provisional: true`.

---

## 11. Build order

1. **Raw archiver — do this first, this week.** Fetch `bootstrap-static` and
   `fixtures` daily, gzip the unmodified response to
   `raw/{season}/{endpoint}/gw{NN}/{ts}.json.gz`, append a manifest line
   (§2.3a). Roughly 30 lines of code and ~22 MB a season. **The only item on
   this list where delay causes permanent, unrecoverable loss** — everything
   else can be built next month against data that will still be there.
2. **Derived layer.** Parquet or SQLite keyed on `(player_id, round)`, built
   *from the raw archive*, never written directly from a live fetch. It must be
   reproducible by replaying the archive from scratch, so a parsing bug found
   later can be fixed retroactively.
3. **Current-season backfill.** One `event/{gw}/live/` call per completed
   gameweek. Two calls as of GW2, not 620.
4. **Incremental updater.** One `event/{gw}/live/` call per new gameweek,
   appended to storage, with the `bootstrap-static` season-total cross-check
   from §2.5 as a fail-closed assertion.
5. **Filtering and feature extraction**, with tests for §3.1–3.4.
6. **Calibration harness** and the baselines, **stratified per §8b** and
   validated per §8.0. Report Brier per `estimator` as well as per stratum, so
   the flag table's contribution is visible separately from the lookup's.
7. **P(starts)** across the full pool.
8. **Expected minutes** across the full pool.
9. **Evidence layer** (§4.4): scrape predicted XIs per club, merge over the
   lookup output, record `tier` per prediction, and score the tiers separately.
10. **Optional:** 620 `element-summary` calls for `history_past`, to replace
    the price-and-position priors with prior-season start rates. Defer until
    the pipeline works end to end.

Steps 1 to 6 before any modelling. Building the scoreboard first is what makes
steps 7 and 8 measurable rather than plausible. Step 1 is the only one with a
deadline attached — every gameweek it is skipped is availability data lost for
good.

A useful checkpoint after step 5: run the three baselines alone, stratified.
That gives the numbers the model has to beat, and it will show immediately
that pool-wide Brier is a misleading metric — which is the lesson §8b exists
to prevent learning the hard way.
