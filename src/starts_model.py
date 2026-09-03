"""P(starts): a lookup-table model for whether a player starts a gameweek.

No regression, no optimiser -- count how often each situation resolved each
way, the same design validated in
notebooks/fpl_starts_analysis.ipynb (sections 6-7, walk-forward on 2025-26):
"calibrated + rolling" beats persistence in 96% of mid-season gameweeks.

That validation excludes any row without 4 gameweeks of prior history --
which is every early-season gameweek, in every season, including the one
this project is actually predicting right now. Section 10 of the same
notebook backtests the fix used here: compute `prev`/`roll4` across the
season boundary (last season's tail standing in as history for this
season's first gameweeks) rather than leaving them undefined. Result: the
cross-season table beats a plain prior-season rate by ~40% on Brier from
gameweek 2 of a new season onward -- but *loses* to it at gameweek 1
specifically (last season's final gameweeks are dead-rubber territory, a
poor predictor of a new season's opening XI). `predict_gameweek` routes on
that finding: target round 1 uses the plain prior-season rate; every other
round uses the cross-season lookup table.

Prior-season history comes from the community archive
(vaastav/Fantasy-Premier-League), since the live API holds the current
season only (docs/build_spec_minutes_model.md Section 2.3). Current-season
history comes from this project's own derived.db (derived.py).

Players with no history on either side -- new signings, promotions, youth --
get `cold_start: true` and a pool-average fallback rather than a silent
guess, per docs Section 8b. That's a known-weak placeholder for the
price-and-position prior Section 8b actually specifies, which isn't built
yet.

`predict_gameweek_refined` layers docs Section 4.1's availability routing on
top of `predict_gameweek`'s raw lookup table: a hard gate to 0 for
status in {i, s, u}, and a separate observed-frequency flag table (Section
4.1a) for status == 'd' or a graded chance_of_playing_next_round. Explicitly
**not** `P(available) * P(selected)` -- Section 4.1 shows that fails, since
the flag fields measure fitness, not selection, and can't tell Haaland apart
from a fourth-choice midfielder (both carry status "a", chance null). Every
prediction carries a `model_version` ("raw_lookup" or "refined_availability")
so the two can be scored side by side via scoring.py's compare_models.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import io
import json
import os
import urllib.request
from collections import namedtuple

import pandas as pd

import archiver

COMMUNITY_ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)
PREDICTIONS_DIR = "predictions"

# roll4 bins, matching notebooks/fpl_starts_analysis.ipynb exactly, so a
# fitted table there and one here mean the same thing.
BINS = [-0.01, 0.24, 0.49, 0.74, 1.01]

FittedLookup = namedtuple("FittedLookup", ["r1", "r0", "table"])


def fetch_community_archive(path):
    """GET one file from the community archive and return its raw bytes.
    The only network seam -- tests inject a fake instead.
    """
    req = urllib.request.Request(
        COMMUNITY_ARCHIVE_BASE + "/" + path, headers={"User-Agent": "curl"}
    )
    return urllib.request.urlopen(req, timeout=90).read()


def load_prior_season_starts(fetch, season):
    """(code, GW, y) for every player-gameweek of one full completed
    season, from the community archive. Joined on `code`, not the raw
    `element` id `merged_gw.csv` uses directly -- ids are only stable
    within a season (the same problem this project's own archiver/derived
    layer solves for the live API; `players_raw.csv` carries both per
    season).
    """
    gw = pd.read_csv(io.BytesIO(fetch(season + "/gws/merged_gw.csv")), low_memory=False)
    players = pd.read_csv(io.BytesIO(fetch(season + "/players_raw.csv")))
    id_to_code = players.set_index("id")["code"]
    gw["code"] = gw["element"].map(id_to_code)
    gw_col = "GW" if "GW" in gw.columns else "round"
    gw = gw.rename(columns={gw_col: "GW"})
    gw["y"] = gw["starts"].astype(int)
    return gw[["code", "GW", "y"]].dropna(subset=["code"])


def load_current_season_starts(conn, season):
    """(code, GW, y) for the season being predicted, from this project's
    own derived.db.
    """
    df = pd.read_sql(
        "SELECT code, round AS GW, starts FROM player_gameweek_stats WHERE season = ?",
        conn, params=(season,),
    )
    df["y"] = df["starts"].astype(int)
    return df[["code", "GW", "y"]]


def build_xseason_features(prior_df, current_df):
    """One continuous per-code sequence -- prior season's gameweeks as
    periods 1..N, current season's as N+1 onward -- with `prev`/`roll4`
    computed across the join. See the module docstring for why this beats
    leaving them undefined for early-season rows.
    """
    period_offset = int(prior_df["GW"].max()) if len(prior_df) else 0
    prior = prior_df.copy()
    prior["period"] = prior["GW"]
    current = current_df.copy()
    current["period"] = current["GW"] + period_offset

    combined = pd.concat([prior, current], ignore_index=True)
    combined = combined.sort_values(["code", "period"])
    combined["prev"] = combined.groupby("code")["y"].shift(1)
    combined["roll4"] = combined.groupby("code")["y"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    return combined


def fit_lookup_table(train):
    """Fit the two-number (`prev`-only) and binned (`prev` + `roll4`)
    lookup tables on strictly-prior data -- observed frequencies, nothing
    fitted beyond a group mean.
    """
    r1 = train.loc[train["prev"] == 1, "y"].mean()
    r0 = train.loc[train["prev"] == 0, "y"].mean()
    t = train.copy()
    t["bin"] = pd.cut(t["roll4"], BINS, labels=False)
    table = t.groupby(["prev", "bin"])["y"].agg(["mean", "size"])
    return FittedLookup(r1=r1, r0=r0, table=table)


def predict_p_start(fitted, test, min_cell=50):
    """P(start) per row of `test` (needs `prev` and `roll4` columns),
    using the binned table with a fallback to the two-number table for
    cells with fewer than `min_cell` observations.
    """
    bins = pd.cut(test["roll4"], BINS, labels=False)
    predictions = []
    for prev, bin_ in zip(test["prev"].values, bins):
        fallback = fitted.r1 if prev == 1 else fitted.r0
        key = (prev, bin_)
        if key in fitted.table.index and fitted.table.loc[key, "size"] >= min_cell:
            predictions.append(fitted.table.loc[key, "mean"])
        else:
            predictions.append(fallback)
    return predictions


def next_period_features(combined):
    """For each code, the `prev`/`roll4` that apply to one more period
    appended right after their last observed one -- the actual features to
    predict the *next* gameweek with, not the features attached to their
    last played row (which describe the row before that one). Public
    because scoring.py reuses it to build the persistence baseline.
    """
    ordered = combined.sort_values("period")
    grouped = ordered.groupby("code")["y"]
    prev = grouped.last()
    roll4 = grouped.apply(lambda s: s.tail(4).mean())
    return pd.DataFrame({"prev": prev, "roll4": roll4}).reset_index()


def predict_gameweek(conn, season, prior_season, target_round, fetch=None, min_cell=50):
    """P(start) for every player in `players`, for `target_round` of
    `season`. Trains on everything strictly before that round: the whole
    of `prior_season` plus any already-played rounds of `season`.

    `target_round == 1` uses the plain prior-season rate instead of the
    cross-season lookup table -- see the module docstring. Mainly useful
    for retrospective scoring (docs Section 8.1), since by the time this
    project can predict anything, gameweek 1 is normally already played.

    Returns a DataFrame: code, web_name, p_start, cold_start, n_observed,
    method. `cold_start` marks a player with no history on either side
    (new signing, promotion, youth) -- `p_start` for those is the training
    pool's overall rate, not a real prediction.
    """
    if fetch is None:
        fetch = fetch_community_archive

    prior_df = load_prior_season_starts(fetch, prior_season)
    current_df = load_current_season_starts(conn, season)
    train_current = current_df[current_df["GW"] < target_round]

    prior_rate = prior_df.groupby("code")["y"].mean()
    pool_fallback_rate = float(prior_df["y"].mean()) if len(prior_df) else 0.5
    players = pd.read_sql("SELECT code, web_name FROM players", conn)

    if target_round == 1:
        p_start = players["code"].map(prior_rate)
        cold_start = p_start.isna()
        n_observed = players["code"].map(prior_df.groupby("code").size()).fillna(0).astype(int)
        return pd.DataFrame({
            "code": players["code"],
            "web_name": players["web_name"],
            "p_start": p_start.fillna(pool_fallback_rate).values,
            "cold_start": cold_start.values,
            "n_observed": n_observed.values,
            "method": cold_start.map({True: "cold_start_pool_rate",
                                       False: "prior_season_rate"}).values,
        })

    combined = build_xseason_features(prior_df, train_current)
    train = combined.dropna(subset=["prev", "roll4"])
    fitted = fit_lookup_table(train)
    features = next_period_features(combined)
    n_observed_by_code = train.groupby("code").size()

    merged = players.merge(features, on="code", how="left")
    has_features = merged["prev"].notna()

    p_start = pd.Series(pool_fallback_rate, index=merged.index)
    if has_features.any():
        p_start.loc[has_features] = predict_p_start(
            fitted, merged.loc[has_features], min_cell=min_cell
        )

    return pd.DataFrame({
        "code": merged["code"],
        "web_name": merged["web_name"],
        "p_start": p_start.values,
        "cold_start": (~has_features).values,
        "n_observed": merged["code"].map(n_observed_by_code).fillna(0).astype(int).values,
        "method": has_features.map({True: "cal_rolling_xseason",
                                     False: "cold_start_pool_rate"}).values,
    })


# --------------------------------------------------------------------------
# Availability routing -- docs Section 4.1: a gate, not a product.
# --------------------------------------------------------------------------


FlagTable = namedtuple("FlagTable", ["cells", "pooled"])


def load_availability_for_round(conn, season, round_number):
    """Each player's most recent availability snapshot taken while
    `round_number` was still upcoming (`next_gw == round_number`) -- the
    latest information actually available before that gameweek's deadline.
    Used both to route the round being predicted and, called once per past
    round, to fit the flag table on. Empty if nothing was archived at that
    `next_gw` yet.
    """
    df = pd.read_sql(
        "SELECT code, fetched_at, status, chance_of_playing_next_round AS chance "
        "FROM player_availability_snapshots WHERE season = ? AND next_gw = ?",
        conn, params=(season, round_number),
    )
    if len(df) == 0:
        return df[["code", "status", "chance"]]
    # fetched_at sorts correctly as a plain string (zero-padded
    # "YYYYMMDDTHHMMSSZ"), so the last row per code after sorting is the
    # latest pull -- avoids idxmax(), which errors on string dtype here.
    latest = df.sort_values("fetched_at").groupby("code").tail(1)
    return latest[["code", "status", "chance"]].reset_index(drop=True)


def _flag_bucket(status, chance):
    """One of docs Section 4.1a's flag-table cells. Only meaningful for a
    row already routed to the flag table (status == 'd', or a non-null
    chance_of_playing_next_round < 100) -- status in {i, s, u} is a hard
    gate handled before this and never reaches here.
    """
    if chance == 0:
        return "chance_0"
    if chance == 25:
        return "chance_25"
    if chance == 50:
        return "chance_50"
    if chance == 75:
        return "chance_75"
    return "doubtful_no_chance"  # status == 'd', chance null or 100


def fit_flag_table(conn, season, target_round, combined, period_offset, min_cell=50):
    """Observed P(starts) per (flag bucket, prev) cell, from every played
    round of `season` strictly before `target_round`. Cross the flag level
    with `prev` only, never `roll4` (docs Section 4.1a): the flagged
    population is small and roll4 cells would not fill.

    Only this project's own current-season archive can fit this -- the
    community archive backing `prior_season` carries no status/chance
    fields at all (docs Section 2.3), so there is no cross-season
    equivalent of build_xseason_features here. Expect this to be sparse or
    empty for a season's first several gameweeks; every cell then falls
    back to the pooled "any flag" rate (`.pooled`), and from there to the
    unmodified raw prediction -- provisional by design (docs: "ship the
    routing with provisional values... until [~10 gameweeks accumulate]"),
    not a bug.

    Returns a FlagTable(cells, pooled): `cells` indexed by (bucket, prev),
    `pooled` indexed by prev alone (collapsing bucket, the first fallback).
    Both carry "mean"/"size" columns; empty (but correctly shaped) if there
    is nothing to fit yet.
    """
    empty = pd.DataFrame(columns=["mean", "size"])
    current_rows = combined[combined["period"] > period_offset].copy()
    current_rows["GW"] = current_rows["period"] - period_offset
    current_rows = current_rows[current_rows["GW"] < target_round]

    pairs = []
    for round_number in sorted(current_rows["GW"].unique()):
        round_rows = current_rows.loc[
            current_rows["GW"] == round_number, ["code", "prev", "y"]
        ]
        availability = load_availability_for_round(conn, season, int(round_number))
        if len(availability) == 0:
            continue
        pairs.append(round_rows.merge(availability, on="code", how="inner"))

    if not pairs:
        return FlagTable(cells=empty, pooled=empty)

    all_pairs = pd.concat(pairs, ignore_index=True).dropna(subset=["prev"])
    flagged = all_pairs[
        (all_pairs["status"] == "d") | (all_pairs["chance"] < 100)
    ].copy()
    if len(flagged) == 0:
        return FlagTable(cells=empty, pooled=empty)

    flagged["bucket"] = [
        _flag_bucket(s, c) for s, c in zip(flagged["status"], flagged["chance"])
    ]
    cells = flagged.groupby(["bucket", "prev"])["y"].agg(["mean", "size"])
    pooled = flagged.groupby("prev")["y"].agg(["mean", "size"])
    return FlagTable(cells=cells, pooled=pooled)


def _flag_table_lookup(flag_table, bucket, prev, min_cell):
    """(p_start, method) for one flag-table cell, walking docs Section
    4.1a's fallback chain: cell -> pooled "any flag" rate (same `prev`) ->
    (None, None), meaning the caller keeps the raw lookup prediction.
    """
    cell_key = (bucket, prev)
    if cell_key in flag_table.cells.index and flag_table.cells.loc[cell_key, "size"] >= min_cell:
        return flag_table.cells.loc[cell_key, "mean"], "flag_table"
    if prev in flag_table.pooled.index and flag_table.pooled.loc[prev, "size"] >= min_cell:
        return flag_table.pooled.loc[prev, "mean"], "flag_table_pooled"
    return None, None


def route_predictions_with_availability(predictions, availability, flag_table, min_cell=50):
    """Apply docs Section 4.1's routing on top of raw lookup predictions
    (which must already carry a `prev` column, e.g. from
    next_period_features): a hard gate to 0 for injured/suspended/
    unavailable, the flag table (with its fallback chain) for doubtful/
    graded-chance players, and the raw lookup prediction left untouched for
    everyone else -- never a multiplicative P(available) * P(selected),
    which docs Section 4.1 shows fails.
    """
    merged = predictions.merge(availability, on="code", how="left")
    result = merged.copy()

    unavailable = merged["status"].isin(["i", "s", "u"])
    result.loc[unavailable, "p_start"] = 0.0
    result.loc[unavailable, "method"] = "hard_gate_unavailable"

    doubtful = (~unavailable) & merged["status"].notna() & (
        (merged["status"] == "d") | (merged["chance"] < 100)
    )
    for idx in merged.index[doubtful]:
        prev = merged.at[idx, "prev"]
        if pd.isna(prev):
            continue
        bucket = _flag_bucket(merged.at[idx, "status"], merged.at[idx, "chance"])
        p_start, method = _flag_table_lookup(flag_table, bucket, prev, min_cell)
        if p_start is not None:
            result.at[idx, "p_start"] = p_start
            result.at[idx, "method"] = method

    return result.drop(columns=["status", "chance", "prev"])


def predict_gameweek_refined(conn, season, prior_season, target_round, fetch=None, min_cell=50):
    """predict_gameweek(), refined by docs Section 4.1's availability
    routing -- see the module docstring. Falls back to the unmodified raw
    predictions if there's no availability data archived for
    `target_round` yet, or (predict_gameweek's own round-1 rule) if
    `target_round == 1`, since no current-season history exists yet to
    gate or flag against.

    Returns the same shape as predict_gameweek, with `method` reflecting
    whichever rule actually produced each row's p_start.
    """
    raw = predict_gameweek(conn, season, prior_season, target_round,
                            fetch=fetch, min_cell=min_cell)
    if target_round == 1:
        return raw

    availability = load_availability_for_round(conn, season, target_round)
    if len(availability) == 0:
        return raw

    if fetch is None:
        fetch = fetch_community_archive
    prior_df = load_prior_season_starts(fetch, prior_season)
    current_df = load_current_season_starts(conn, season)
    period_offset = int(prior_df["GW"].max()) if len(prior_df) else 0
    train_current = current_df[current_df["GW"] < target_round]
    combined = build_xseason_features(prior_df, train_current)

    features = next_period_features(combined)[["code", "prev"]]
    flag_table = fit_flag_table(conn, season, target_round, combined, period_offset,
                                 min_cell=min_cell)

    predictions_with_prev = raw.merge(features, on="code", how="left")
    return route_predictions_with_availability(
        predictions_with_prev, availability, flag_table, min_cell=min_cell
    )


# --------------------------------------------------------------------------
# Snapshotting predictions -- can't be reconstructed later, same reasoning
# as the raw archive (docs Section 3.7 / 8a): rebuilding derived.db later
# and predicting "for" a past gameweek would use data that wasn't available
# at the time, silently changing the answer.
# --------------------------------------------------------------------------


def snapshot_predictions(predictions, season, target_round, model_version="raw_lookup",
                          base_dir=PREDICTIONS_DIR, clock=archiver.utcnow):
    """Write `predictions` (a DataFrame from predict_gameweek or
    predict_gameweek_refined) to a write-once, timestamped JSON file.
    Refuses to overwrite an existing snapshot for the same
    season/round/model_version/timestamp, same as the raw archiver.

    `model_version` distinguishes different prediction methods for the same
    gameweek (e.g. "raw_lookup" vs "refined_availability") so derived.py
    keeps the latest of each separately and scoring.py can compare them.
    """
    fetched_at = clock()
    filename = "gw{0:02d}_{1}_{2}.json".format(
        target_round, model_version, archiver.format_timestamp(fetched_at)
    )
    path = os.path.join(base_dir, season, filename)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    if os.path.exists(path):
        raise archiver.ArchiveError(
            "refusing to overwrite existing predictions snapshot: {0}".format(path)
        )
    payload = {
        "season": season,
        "target_round": target_round,
        "model_version": model_version,
        "predicted_at": archiver.format_timestamp(fetched_at),
        "predictions": predictions.to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def _main():
    import argparse
    import sqlite3

    import derived

    parser = argparse.ArgumentParser(
        description="Predict P(starts) for the next unplayed gameweek and "
                    "snapshot the result."
    )
    parser.add_argument("--db-path", default=derived.DERIVED_DB_PATH)
    parser.add_argument("--season", default=None,
                        help="Defaults to the only season in derived.db.")
    parser.add_argument("--prior-season", default=None,
                        help="Defaults to one season before --season.")
    parser.add_argument("--target-round", type=int, default=None,
                        help="Defaults to (max archived round) + 1.")
    parser.add_argument("--predictions-dir", default=PREDICTIONS_DIR)
    args = parser.parse_args()

    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(module_dir))

    conn = sqlite3.connect(args.db_path)
    season = args.season
    if season is None:
        seasons = pd.read_sql(
            "SELECT DISTINCT season FROM player_gameweek_stats", conn
        )["season"]
        if len(seasons) != 1:
            raise SystemExit(
                "derived.db has {0} seasons; pass --season explicitly".format(len(seasons))
            )
        season = seasons.iloc[0]

    prior_season = args.prior_season
    if prior_season is None:
        start_year = int(season[:4]) - 1
        prior_season = "{0}-{1:02d}".format(start_year, (start_year + 1) % 100)

    target_round = args.target_round
    if target_round is None:
        max_round = pd.read_sql(
            "SELECT MAX(round) AS r FROM player_gameweek_stats WHERE season = ?",
            conn, params=(season,),
        )["r"].iloc[0]
        target_round = int(max_round) + 1 if max_round is not None else 1

    # Both versions, every run: this is what lets scoring.py compare them
    # once the gameweek's played, rather than only ever having one to look
    # at (see docs Section 4.1 / starts_model.py module docstring).
    raw = predict_gameweek(conn, season, prior_season, target_round)
    refined = predict_gameweek_refined(conn, season, prior_season, target_round)
    conn.close()

    raw_path = snapshot_predictions(raw, season, target_round,
                                    model_version="raw_lookup",
                                    base_dir=args.predictions_dir)
    refined_path = snapshot_predictions(refined, season, target_round,
                                        model_version="refined_availability",
                                        base_dir=args.predictions_dir)

    n_cold_start = int(raw["cold_start"].sum())
    n_gated = int((refined["method"] == "hard_gate_unavailable").sum())
    n_flagged = int(refined["method"].isin(["flag_table", "flag_table_pooled"]).sum())
    print("{0}: {1} players ({2} cold_start)".format(season, len(raw), n_cold_start))
    print("  raw_lookup           -> {0}".format(raw_path))
    print("  refined_availability -> {0} ({1} hard-gated, {2} flag-table)".format(
        refined_path, n_gated, n_flagged
    ))


if __name__ == "__main__":
    _main()
