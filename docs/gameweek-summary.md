# Gameweek summary

A log of what changed each gameweek and what we did about it. Newest first.

## GW3, mid-gameweek (2026-09-05/06)

No transfer this entry — GW3 was already partway played (Saturday's fixtures
done, Sunday's two still to come), so this was a pipeline/model change and a
live pilot, not an end-of-gameweek decision.

We extended the news evidence layer beyond premierleague.com's own feed.
`web_news_archiver.py` builds Brave Search queries straight from the
gameweek's fixtures — predicted lineup, team news, and a press-conference
query for both teams, not just the home side — and archives every result
unfiltered: no allowlist, no denylist. That was a deliberate call, not an
oversight. A curated list built from a handful of post-hoc observations
encodes noise as policy, and a wrongly-excluded source is invisible in a way
a wrongly-included one isn't — a junk article gets caught at extraction and
dilutes a count, a missing one is never seen at all. We did try a direct
scrape of all 20 clubs' own news pages first, and dropped it: 20 of 20
"fetched", but most are client-rendered pages with nothing extractable, and
what did come through was mostly navigation boilerplate. A lighter
replacement stayed — recognising when a Brave result already lands on a
club's own domain, so it still gets tagged as club-official provenance
without a dedicated scrape.

Ran it live, restricted to Sunday's two fixtures (Everton–Man Utd,
Arsenal–Chelsea) as a pilot: 45 pages fetched, 39 usable, 92 claims
extracted, real players correctly matched, nothing fabricated. Also
switched the extraction model to Haiku (a better fit for a narrow,
high-volume classification task, and meaningfully cheaper) — which
surfaced a real bug, since Haiku rejects the `effort` parameter our
Opus-tier requests were sending. Fixed by only sending it to models that
support it.

The more interesting bug: running mid-gameweek meant FPL's `next_gw` had
already rolled over to GW4, even though GW3 wasn't finished. Evidence
gathered about GW3's remaining fixtures was getting mislabelled as GW4
evidence — which would have silently fed fixture-specific claims about the
wrong gameweek's players into next week's real prediction. Fixed by having
web-news articles carry their own known target round through instead of
re-inferring it from the nearest FPL snapshot.

Along the way we found a real weakness in how the model combines
disagreeing claims. One article wrongly said Jack Grealish would miss a
fixture Everton weren't even playing (a stale piece); other, correct
articles said he'd start. The old logic picked a winner by a fixed
precedence order — confirmed-out always won, regardless of how many other
claims contradicted it — so Grealish was hard-gated to 0. Now, when a
player's claims disagree, they're blended into one probability weighted by
how many claims made each category, rather than one category silently
winning. For Grealish specifically that's 0.0 (1 confirmed-out) + 0.90 (1
confirmed-starting) + 0.35 (1 returning-from-injury) + 0.50×3
(rotation-risk), landing at 0.46 instead of 0. Deliberately not yet
weighted by source (club-official vs. everyone else) — a manager has every
incentive to bluff about his starting XI in a presser, unlike about an
injury, so that trust ordering is untested for this claim type and stays a
question for `scoring.py` to answer later, not an assumption to bake in now.

**Watching for GW3 results (Monday):** whether Grealish actually started —
the blended 0.46 vs. the old hard 0 is the cleanest single test of whether
disagreement-blending is an improvement. More broadly, once GW3 is
`data_checked`, re-run the pipeline and `scoring.py --target-round 3` to
compare all three model arms properly; the Sunday-fixture players worth
eyeballing by hand first are Grealish, Saliba, Timber, Rice, Saka, Konsa,
and Mount in `predictions/2026-27/gw03_refined_availability_news_20260905T194809Z.json`.

## GW3 → GW4 (2026-09-04)

This was the first gameweek where we had a P(starts) model at all — and
the first time we used Claude Code, rather than Claude chat, to help pick
the team.

The model has three layers, built to be compared against each other once
results come in:

1. **Raw lookup** — a historical minutes-based table, no live info.
2. **Refined availability** — layers FPL's own status flag
   (`chance_of_playing_next_round`) on top, hard-gating players marked
   unavailable.
3. **Refined + news** — the new piece this week. `news_extraction.py`
   scrapes premierleague.com articles and has an LLM classify each player
   mention into a fixed set of claims (confirmed starting, confirmed out,
   rotation risk, returning from injury), each backed by a verbatim quote
   we check against the source. No guessed probabilities — just labelled
   claims that the code turns into numbers. The point is that team news
   breaks before FPL's status flag catches up, which was our biggest
   source of error. Ran it for real this week: 74 articles processed, all
   three models snapshotted.

**Transfers:** Wilson out — low P(starts). Ndiaye out too, for the
opposite reason — he'd just moved from Everton to Man City and we had no
confidence in his starting probability there yet. In came Stach and
Sangaré, both from Brentford.

**Bench:** injured Rodon, plus Senesi on a low P(starts).

**Did the news layer earn its keep?** Checked the actual GW3 numbers. On
`confirmed_out`, no — all 26 matched claims were already at `p_start = 0.0`
under the plain status-flag model; news just relabelled the method, never
moved a number. But on `returning_from_injury`/`rotation_risk`, yes — 12
players FPL listed as fully fit (status `a`, no doubt flag at all) got
pulled down from a high historical rate once news caught the "back but
being managed" nuance FPL's flag structurally can't express. So the value
is real, just narrower than expected: it's not that news beats FPL to the
injury news, it's that it covers a state FPL doesn't have a flag for.

That's a decent argument for scraping beyond premierleague.com — outlets
with independent reporting (rather than the same official line) should
turn up more of that soft, no-FPL-equivalent signal, which is where the
layer is actually pulling weight.

**Watching for GW4:** whether the news layer actually beats the plain
status-flag model once results are in — its priors are hand-set guesses,
not fitted, and this is the first week there's data to check them against.
