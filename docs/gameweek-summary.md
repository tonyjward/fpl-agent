# Gameweek summary

A log of what changed each gameweek and what we did about it. Newest first.

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

**Watching for GW4:** whether the news layer actually beats the plain
status-flag model once results are in — its priors are hand-set guesses,
not fitted, and this is the first week there's data to check them against.
