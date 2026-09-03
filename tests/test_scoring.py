"""Tests for scoring.py.

Same faking approach as test_starts_model.py: the community archive is
faked via an injected `fetch(path) -> bytes`, and derived.db is a minimal
sqlite connection carrying just the columns scoring.py reads.
"""

import sqlite3

import pandas as pd
import pytest

import scoring
import starts_model

SEASON = "2026-27"
PRIOR_SEASON = "2025-26"


def make_fake_fetch(responses):
    def fetch(path):
        if path not in responses:
            raise AssertionError("unexpected path: {0}".format(path))
        return responses[path]
    return fetch


def merged_gw_csv(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def players_raw_csv(id_code_pairs):
    return pd.DataFrame(
        [{"id": i, "code": c} for i, c in id_code_pairs]
    ).to_csv(index=False).encode("utf-8")


def make_derived_db(path, players, gameweek_rows, prediction_rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE players (code INTEGER PRIMARY KEY, web_name TEXT)")
    conn.executemany("INSERT INTO players VALUES (?, ?)", players)
    conn.execute("CREATE TABLE player_gameweek_stats (code, season, round, starts)")
    conn.executemany(
        "INSERT INTO player_gameweek_stats VALUES (?, ?, ?, ?)", gameweek_rows
    )
    conn.execute(
        "CREATE TABLE predictions "
        "(code, season, target_round, model_version, p_start, cold_start)"
    )
    conn.executemany(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)", prediction_rows
    )
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# brier
# --------------------------------------------------------------------------


def test_brier_known_values():
    assert scoring.brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert scoring.brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_brier_empty_is_nan():
    assert scoring.brier([], []) != scoring.brier([], [])  # NaN != NaN


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------


def test_accuracy_known_values():
    # p>=0.5 predicts start for rows 0 and 1; actual y matches on row 0 only.
    assert scoring.accuracy([0.6, 0.6, 0.2], [1, 0, 0]) == pytest.approx(2 / 3)


def test_accuracy_empty_is_nan():
    assert scoring.accuracy([], []) != scoring.accuracy([], [])


# --------------------------------------------------------------------------
# label_strata
# --------------------------------------------------------------------------


def test_label_strata_thresholds():
    history = pd.DataFrame({
        "code": (
            [1001] * 10 +   # 8/10 = Core
            [1002] * 10 +   # 2/10 = Rotation
            [1003] * 10 +   # 1/10 = Marginal
            [1004] * 10     # 0/10 = Deep
        ),
        "y": (
            [1] * 8 + [0] * 2 +
            [1] * 2 + [0] * 8 +
            [1] * 1 + [0] * 9 +
            [0] * 10
        ),
    })

    labels = scoring.label_strata(history)

    assert labels[1001] == "Core"
    assert labels[1002] == "Rotation"
    assert labels[1003] == "Marginal"
    assert labels[1004] == "Deep"


# --------------------------------------------------------------------------
# score_gameweek
# --------------------------------------------------------------------------


def test_score_gameweek_raises_if_predictions_missing(tmp_path):
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice")],
        gameweek_rows=[(1001, SEASON, 3, 1)],
        prediction_rows=[],
    )
    fetch = make_fake_fetch({})

    with pytest.raises(scoring.ScoringError, match="no archived 'raw_lookup' predictions"):
        scoring.score_gameweek(conn, SEASON, PRIOR_SEASON, 3, fetch=fetch)
    conn.close()


def test_score_gameweek_raises_if_outcomes_missing(tmp_path):
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice")],
        gameweek_rows=[],
        prediction_rows=[(1001, SEASON, 3, "raw_lookup", 0.8, 0)],
    )
    fetch = make_fake_fetch({})

    with pytest.raises(scoring.ScoringError, match="hasn't been played"):
        scoring.score_gameweek(conn, SEASON, PRIOR_SEASON, 3, fetch=fetch)
    conn.close()


def test_score_gameweek_reports_pool_and_beats_persistence(tmp_path):
    """Alice: nailed on every gameweek last season, benched GW1-2 this
    season (a real drop, not noise), then starts GW3. Persistence (last
    observed = benched) predicts low; the model, given the true P(starts)
    for someone who's usually nailed but was just dropped, should predict
    higher and score better on this single call.
    """
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(
            [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
        ),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice")],
        gameweek_rows=[(1001, SEASON, 1, 0), (1001, SEASON, 2, 0),
                       (1001, SEASON, 3, 1)],
        prediction_rows=[(1001, SEASON, 3, "raw_lookup", 0.75, 0)],
    )

    report = scoring.score_gameweek(conn, SEASON, PRIOR_SEASON, 3, fetch=fetch)
    conn.close()

    assert "POOL" in report.index
    assert report.loc["POOL", "n"] == 1
    assert report.loc["POOL", "model"] < report.loc["POOL", "persistence"]
    assert bool(report.loc["POOL", "beats_persistence"]) is True
    assert report.loc["POOL", "model_accuracy"] == pytest.approx(1.0)  # p=0.75, started


def test_score_gameweek_debut_player_labelled_deep_not_dropped(tmp_path):
    """A player with no history before the round being scored (a debut)
    must still appear in the report -- under Deep, not silently excluded.
    """
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(
            [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
        ),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice"), (9999, "Debutant")],
        gameweek_rows=[(1001, SEASON, 1, 1), (1001, SEASON, 2, 1),
                       (1001, SEASON, 3, 1), (9999, SEASON, 3, 1)],
        prediction_rows=[(1001, SEASON, 3, "raw_lookup", 0.85, 0), (9999, SEASON, 3, "raw_lookup", 0.28, 1)],
    )

    report = scoring.score_gameweek(conn, SEASON, PRIOR_SEASON, 3, fetch=fetch)
    conn.close()

    assert "Deep" in report.index
    assert "Core" in report.index
    assert report.loc["Core", "n"] + report.loc["Deep", "n"] == 2


# --------------------------------------------------------------------------
# compare_models / list_model_versions
# --------------------------------------------------------------------------


def test_list_model_versions(tmp_path):
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice")],
        gameweek_rows=[(1001, SEASON, 3, 1)],
        prediction_rows=[
            (1001, SEASON, 3, "raw_lookup", 0.8, 0),
            (1001, SEASON, 3, "refined_availability", 0.0, 0),
        ],
    )

    versions = scoring.list_model_versions(conn, SEASON, 3)
    conn.close()

    assert sorted(versions) == ["raw_lookup", "refined_availability"]


def test_compare_models_side_by_side(tmp_path):
    """A hard-gated (refined) prediction of 0 for a player who didn't
    start should score better than the raw model's un-gated guess -- and
    compare_models must show both numbers in one table.
    """
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(
            [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
        ),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "InjuredButHistoricallyNailed")],
        gameweek_rows=[(1001, SEASON, 1, 1), (1001, SEASON, 2, 1),
                       (1001, SEASON, 3, 0)],  # didn't start -- injured
        prediction_rows=[
            (1001, SEASON, 3, "raw_lookup", 0.85, 0),           # unaware of injury
            (1001, SEASON, 3, "refined_availability", 0.0, 0),  # hard-gated
        ],
    )

    comparison = scoring.compare_models(
        conn, SEASON, PRIOR_SEASON, 3, ["raw_lookup", "refined_availability"], fetch=fetch
    )
    conn.close()

    assert "raw_lookup_brier" in comparison.columns
    assert "refined_availability_brier" in comparison.columns
    assert (
        comparison.loc["POOL", "refined_availability_brier"]
        < comparison.loc["POOL", "raw_lookup_brier"]
    )
