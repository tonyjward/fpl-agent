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
- `game_config.scoring` in every archived `bootstrap-static` payload is the
  **live, authoritative source for point values per position** — never
  hardcode them. Verified live: GKP goals are worth **10**, not 6; GKP gets
  **0** defensive-contribution points (the mechanic doesn't apply to them at
  all, not just zero-weighted).
- `defensive_contribution` in `event/{gw}/live` is already the **precomputed
  raw CBIT/CBIRT sum**, not points — confirmed against real archived data
  (DEF: CBI + tackles; MID/FWD: CBI + tackles + recoveries; GKP: not
  computed, always 0). No need to reconstruct it from components. The
  thresholds themselves (10 DEF, 12 MID/FWD, capped at 2 pts) are **not** in
  `game_config` — a hardcoded external constant, confirmed against
  premierleague.com, that would need manual re-verification if FPL changes
  the rule.
- **Scraped web content cannot be trusted against this project's own
  archive without checking.** Tested directly (2026-09-03): multiple
  separately-fetched articles claimed a player was starting for a club he'd
  already been loan-transferred out of a full day earlier, per our own
  `player_availability_snapshots` — not a stale-page issue, the articles
  were dated *after* the transfer. `WebFetch` also introduced its own
  errors (garbled player names) since it summarizes through a small model
  rather than returning raw content.
  **2026-09-04 update:** `premierleague.com`'s news/injury pages are indeed
  JS-rendered SPAs and unreadable by `WebFetch` or plain HTTP, but
  `pl_content.py` bypasses that entirely by calling the site's own JSON
  content API directly (`api.premierleague.com/content/premierleague/...`,
  found by reading the site's JS bundle — undocumented, no auth needed).
  A minority of club sites (linked from syndicated articles) additionally
  block a plain request behind a JS challenge (seen: mancity.com, via
  Cloudflare) — `pl_content.fetch_raw_html` falls back to a real headless
  Chrome for those (your own code driving `playwright` + the machine's
  installed Chrome, not any Claude-side browsing tool; degrades to no-op if
  neither is present). The §4.4-equivalent contradiction-check is now
  built and validated live for injury data specifically — see
  `derived.match_injury_player` / `match_team_code`, and docs/README.md's
  "Known gaps" for exactly what's still open (free-text news extraction).

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
- `pl_content.py` — client for premierleague.com's own (undocumented) JSON
  content API: news articles (with a headless-Chrome fallback for the
  minority of syndicated sources that block a plain request) and the
  per-club injury table
- `news_archiver.py` — raw archive for `pl_content.py`, same
  write-once/gzip/manifest pattern as `archiver.py`, filed in the same
  per-season manifest under endpoints `pl-news`/`pl-injuries`
- `web_news_archiver.py` — raw archive for a wider-web source: Brave Search
  results built from the target gameweek's fixtures (`predicted lineup` /
  `team news` / both teams' `press conference` queries). Endpoint
  `web-news`, same write-once/gzip/manifest pattern. No allowlist or
  denylist on which results get fetched — every returned result gets
  archived and fetched; per-source trust is meant to come from
  `scoring.py`'s Rotation-stratum accuracy once measured, not a guess made
  up front (see the module docstring and docs/README.md's "A finding the
  model does not capture"). `fixtures_for_round`'s `fixture_ids`/
  `only_unfinished` narrow a run to specific fixtures — e.g. a gameweek
  already partly played, just what's left. No longer directly scrapes club
  sites (dropped 2026-09-06: mostly client-rendered noise) — `is_club_domain`
  instead flags a search result that already lands on one of the 20 clubs'
  own domains, so `derived.py` can grade it `club_official` regardless
- `news_extraction.py` — the one LLM call in this pipeline (build step 9's
  "extract"): classifies each article into a fixed taxonomy
  (`confirmed_starting`/`confirmed_out`/`rotation_risk`/
  `returning_from_injury`) with a verbatim quote, deliberately never a
  probability. Output is a write-once JSON artifact under `extractions/`,
  same shape as `predictions/` — not part of the raw archive, since an LLM
  call is neither free nor byte-reproducible to replay
- `derived.py` — SQLite layer (`derived.db`), rebuilt from scratch every run
  by replaying `raw/`, `predictions/`, and `extractions/`. Includes
  `news_articles`, `injury_reports` (resolved against `players`/`teams`
  with a contradiction flag — `match_injury_player`/`match_team_code`),
  and `news_claims` (same resolution, plus a `target_round` resolved from
  the closest archived availability snapshot)
- `starts_model.py` — the P(starts) lookup table, extended cross-season (see
  `notebooks/fpl_starts_analysis.ipynb` section 10) so early-season gameweeks
  aren't blind. `predict_gameweek_refined`: §4.1's availability routing
  layered on top (hard gate to 0 for status in {i,s,u}; an
  observed-frequency flag table, §4.1a, for doubtful/graded-chance
  players) — a gate, never `P(available) * P(selected)`, per §4.1's own
  explicit warning against that. `predict_gameweek_refined_news`: news
  evidence layered on top of *that* — `confirmed_out` is a hard gate, the
  other three categories use `NEWS_PRIORS` blended toward this season's own
  observed rate via `shrink_toward_prior` (continuous shrinkage, `k=10`
  pseudo-observations — deliberately not a hard `min_cell` cliff, which
  would mean running on priors alone all season given how rarely an
  article makes an explicit claim). Every prediction carries a
  `model_version` ("raw_lookup" / "refined_availability" /
  "refined_availability_news") so all three are separately queryable and
  comparable.
- `scoring.py` — the calibration harness (§8): scores archived predictions
  against archived outcomes, stratified, against three baselines.
  `compare_models` scores multiple `model_version`s for the same gameweek
  side by side — this is how "did refining help" actually gets answered,
  not assumed.

Not yet built: expected minutes, the evidence layer (scraping — attempted
2026-09-03, see the new "Non-obvious facts" entry above on why it wasn't
shipped), the optimiser, the LangGraph agent. There is also no "Model 3:
expected points" anywhere in the spec — `build_spec_minutes_model.md`
stops at expected minutes (§5); how minutes combines with goal/assist/
clean-sheet/bonus/defensive-contribution terms into a single expected-points
number is undesigned. Clean sheets and goals-conceded specifically need a
**team-level** model (not a player-level one) that doesn't exist yet —
`docs/README.md`'s "Known gaps" already flags this as the largest missing
term. Team `strength_attack_*`/`strength_defence_*` ratings (in every
archived `bootstrap-static`, not yet in the derived `teams` table) are the
cheap starting point for it, not yet built either.

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
calibration harness, P(starts) including the §4.1 availability refinement)
are done — see "Scripts" above. Step 5 (dedicated feature/filtering module
for API gotchas 1–5) was mostly obviated: this project's own archive only
ever contains finished, `data_checked` gameweeks, so the community archive's
unplayed-fixture trap (gotcha 1) doesn't apply to it.

**2026-09-03 session:** scoped expected points (§ above, "Not yet built")
and found the spec doesn't cover it yet — that's real design work, not a
quick add. Attempted a real pilot of the evidence layer (step 9) against
three clubs' actual GW3 news; found scraped web content genuinely
unreliable without §4.4's contradiction-check in place (see "Non-obvious
facts"), so nothing from that pilot was shipped into predictions. No
browser automation available this session (user declined) — `WebFetch`
alone cannot read `premierleague.com`'s JS-rendered pages.

**2026-09-04 session:** built the raw-archive half of the evidence layer
(step 9's "scrape") for real, plus the contradiction-check that blocked it
before. `pl_content.py` calls premierleague.com's own JSON content API
directly (no browser needed for that part); a headless Chrome your own
code drives (`playwright` + the local machine's installed Chrome, still no
Claude-side browsing tool) handles the minority of club sites that block a
plain request. `news_archiver.py` archives both news and injuries;
`derived.py` gained `news_articles` and `injury_reports`, the latter
resolved against `players`/`teams` with a `contradiction` flag
(`match_injury_player`/`match_team_code`) — validated live against 2026-27
data, catching one genuine case and, along the way, several real name-
matching bugs (an FPL disambiguating-initial prefix, a 3-letter short-code
substring false-positive, multi-word `web_name`s, accented names) fixed
before trusting the result. See docs/README.md's "Known gaps" for the
exact current state and what's still open.

**2026-09-04 session, continued:** finished step 9. `news_extraction.py`
(the LLM extract call), `derived.py`'s `news_claims` (contradiction-checked
the same way as injuries, plus a `target_round` so a claim can later be
joined to the outcome it was about), and `starts_model.py`'s
`predict_gameweek_refined_news` (the third `model_version`) are all built
and unit-tested with an injected/fake client — **not yet run against the
real Anthropic API**, since no key was available this session. `NEWS_PRIORS`
(0.90/0.50/0.35) and `NEWS_SHRINKAGE_K` (10) are reasoned starting guesses
picked with the user, explicitly meant to be checked (and likely revised)
by `scoring.compare_models(["raw_lookup", "refined_availability",
"refined_availability_news"])` once real predictions and outcomes
accumulate — do not treat them as settled without checking recent scoring
output first.

Remaining, per the build order in @README.md: step 8 (expected minutes),
step 10 (optimiser), step 11 (LangGraph agent) — plus the undesigned
expected-points combination step noted above. Which one is next hasn't
been decided — ask rather than assume.
