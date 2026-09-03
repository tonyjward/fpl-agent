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
# Snapshotting predictions -- can't be reconstructed later, same reasoning
# as the raw archive (docs Section 3.7 / 8a): rebuilding derived.db later
# and predicting "for" a past gameweek would use data that wasn't available
# at the time, silently changing the answer.
# --------------------------------------------------------------------------


def snapshot_predictions(predictions, season, target_round, base_dir=PREDICTIONS_DIR,
                          clock=archiver.utcnow):
    """Write `predictions` (a DataFrame from predict_gameweek) to a
    write-once, timestamped JSON file. Refuses to overwrite an existing
    snapshot for the same season/round/timestamp, same as the raw archiver.
    """
    fetched_at = clock()
    filename = "gw{0:02d}_{1}.json".format(target_round, archiver.format_timestamp(fetched_at))
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

    predictions = predict_gameweek(conn, season, prior_season, target_round)
    conn.close()

    path = snapshot_predictions(predictions, season, target_round,
                                base_dir=args.predictions_dir)
    n_cold_start = int(predictions["cold_start"].sum())
    print("{0}: {1} predictions ({2} cold_start) -> {3}".format(
        season, len(predictions), n_cold_start, path
    ))


if __name__ == "__main__":
    _main()
