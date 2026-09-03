"""Scoring harness: score archived predictions against actual outcomes.

docs/build_spec_minutes_model.md Section 8 (calibration harness), 8.0
(validation protocol), 8.1 (scoring loop), 8b (stratified reporting and the
class-imbalance trap).

**Never report a single pool-wide Brier score.** Section 8b shows Deep
(never-starts players) is ~41% of rows and trivially predictable, which
flatters any pool average into hiding exactly the improvement that matters.
Rotation-stratum Brier is the headline metric; Core/Marginal/Deep are
reported for completeness only. The model must beat all three baselines
(persistence, season-rate, constant 0.9) to justify shipping it (Section 8.1).

Strata are labelled from the same pre-target-round history used to build the
predictions being scored -- never from the round being scored, which would
leak the answer into the stratum that matters most (Section 8.0 rule 2).

Both predictions and actual outcomes must already be archived: predictions
via starts_model.py + derived.py, outcomes via the raw archiver + derived.py
once the gameweek's data_checked. This module only reads derived.db and the
community archive (for the persistence/season-rate baselines, which need
the same cross-season history starts_model.py uses to predict) -- it never
writes anything.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import os

import pandas as pd

import starts_model

STRATA = ["Core", "Rotation", "Marginal", "Deep"]


class ScoringError(Exception):
    """Predictions or actual outcomes aren't archived yet for the round
    being scored.
    """


def brier(p, y):
    pairs = list(zip(p, y))
    if not pairs:
        return float("nan")
    return sum((pi - yi) ** 2 for pi, yi in pairs) / len(pairs)


def accuracy(p, y, threshold=0.5):
    """Share of rows where (p >= threshold) matches the actual outcome.

    A more readable companion to Brier, not a replacement -- it throws away
    calibration (how confident a wrong call was) and, like Brier, is
    meaningless pool-wide: Deep players almost never start, so "always
    guess no" scores near-100% here without the model doing anything.
    Report per stratum, same as Brier (Section 8b).
    """
    pairs = list(zip(p, y))
    if not pairs:
        return float("nan")
    correct = sum(1 for pi, yi in pairs if (pi >= threshold) == bool(yi))
    return correct / len(pairs)


def label_strata(history):
    """code -> one of STRATA, from `history` (code, y) alone. Pass only
    gameweeks strictly before the one being scored -- see module docstring.

    Thresholds per docs Section 8b: Deep = never started, Core >= 50%,
    Rotation 15-50%, Marginal < 15% (but > 0). The denominator is
    gameweeks *registered*, not *available* -- an injured first-choice
    starter drifts toward Rotation. Documented limitation, not fixed here
    (Section 8b: the real fix is archiving `status` over time, which
    derived.py's player_availability_snapshots already does; reclassifying
    from it is future work).
    """
    g = history.groupby("code")["y"].agg(["sum", "size"])
    rate = g["sum"] / g["size"]
    labels = {}
    for code in g.index:
        starts = g.loc[code, "sum"]
        if starts == 0:
            labels[code] = "Deep"
        elif rate.loc[code] >= 0.50:
            labels[code] = "Core"
        elif rate.loc[code] >= 0.15:
            labels[code] = "Rotation"
        else:
            labels[code] = "Marginal"
    return pd.Series(labels, name="stratum")


def load_actual_outcomes(conn, season, target_round):
    """code, y=actually started, for one archived gameweek."""
    df = pd.read_sql(
        "SELECT code, starts FROM player_gameweek_stats "
        "WHERE season = ? AND round = ?",
        conn, params=(season, target_round),
    )
    df["y"] = df["starts"].astype(int)
    return df[["code", "y"]]


def load_scored_predictions(conn, season, target_round):
    """code, p_start, cold_start for one archived gameweek's predictions."""
    return pd.read_sql(
        "SELECT code, p_start, cold_start FROM predictions "
        "WHERE season = ? AND target_round = ?",
        conn, params=(season, target_round),
    )


def score_gameweek(conn, season, prior_season, target_round, fetch=None):
    """Score the archived `predictions` for (season, target_round) against
    actual outcomes, stratified per Section 8b.

    Returns a DataFrame indexed by scope (POOL, then each of STRATA present)
    with columns n, model, model_accuracy, persistence, season_rate,
    constant_0.9, beats_persistence. model_accuracy (share correct at a 0.5
    threshold) is a more readable companion to the Brier columns, not a
    replacement -- see accuracy()'s docstring for why it can't stand alone.

    Baselines are built from the same cross-season history
    starts_model.predict_gameweek used, via the same public functions, so
    "beats persistence" is a fair comparison against what was actually
    available at prediction time -- not a baseline with the benefit of
    hindsight.
    """
    predictions = load_scored_predictions(conn, season, target_round)
    if len(predictions) == 0:
        raise ScoringError(
            "no archived predictions for {0} round {1} -- run starts_model.py "
            "and derived.py first".format(season, target_round)
        )
    outcomes = load_actual_outcomes(conn, season, target_round)
    if len(outcomes) == 0:
        raise ScoringError(
            "no actual outcomes archived for {0} round {1} yet -- this "
            "gameweek hasn't been played/data_checked".format(season, target_round)
        )

    if fetch is None:
        fetch = starts_model.fetch_community_archive
    prior_df = starts_model.load_prior_season_starts(fetch, prior_season)
    current_df = starts_model.load_current_season_starts(conn, season)
    train_current = current_df[current_df["GW"] < target_round]
    combined = starts_model.build_xseason_features(prior_df, train_current)

    strata = label_strata(combined[["code", "y"]])
    persistence_source = starts_model.next_period_features(combined)
    season_rate = combined.groupby("code")["y"].mean()
    pool_fallback_rate = float(prior_df["y"].mean()) if len(prior_df) else 0.5

    scored = predictions.merge(outcomes, on="code", how="inner")
    scored = scored.merge(persistence_source[["code", "prev"]], on="code", how="left")
    scored["persistence"] = scored["prev"].fillna(0.5).clip(0.05, 0.95)
    scored["season_rate"] = scored["code"].map(season_rate).fillna(pool_fallback_rate)
    scored["constant_0.9"] = 0.9
    # A player with no history before this round (a debut) has no stratum
    # label from history either -- reported under Deep rather than dropped,
    # since a labelled-but-untested row beats a silently discarded one.
    scored["stratum"] = scored["code"].map(strata).fillna("Deep")

    rows = []
    for scope in ["POOL"] + STRATA:
        subset = scored if scope == "POOL" else scored[scored["stratum"] == scope]
        if len(subset) == 0:
            continue
        model_brier = brier(subset["p_start"], subset["y"])
        persistence_brier = brier(subset["persistence"], subset["y"])
        rows.append({
            "scope": scope,
            "n": len(subset),
            "model": model_brier,
            "model_accuracy": accuracy(subset["p_start"], subset["y"]),
            "persistence": persistence_brier,
            "season_rate": brier(subset["season_rate"], subset["y"]),
            "constant_0.9": brier(subset["constant_0.9"], subset["y"]),
            "beats_persistence": model_brier < persistence_brier,
        })
    return pd.DataFrame(rows).set_index("scope")


def _main():
    import argparse
    import sqlite3

    import derived

    parser = argparse.ArgumentParser(
        description="Score archived P(starts) predictions against actual outcomes."
    )
    parser.add_argument("--db-path", default=derived.DERIVED_DB_PATH)
    parser.add_argument("--season", default=None,
                        help="Defaults to the only season in derived.db.")
    parser.add_argument("--prior-season", default=None,
                        help="Defaults to one season before --season.")
    parser.add_argument("--target-round", type=int, required=True)
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

    report = score_gameweek(conn, season, prior_season, args.target_round)
    conn.close()

    pd.set_option("display.width", 120)
    print("{0} round {1}, scored against {2}:\n".format(
        season, args.target_round, prior_season
    ))
    print(report.round(4).to_string())


if __name__ == "__main__":
    _main()
