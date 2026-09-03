"""Tests for starts_model.py.

The community archive is faked via an injected `fetch(path) -> bytes`
callable (never real network); derived.db is faked with a minimal sqlite
connection carrying just the columns starts_model.py actually reads, rather
than going through the full derived.rebuild() pipeline (already covered by
test_derived.py).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

import archiver
import starts_model

SEASON = "2026-27"
PRIOR_SEASON = "2025-26"


def make_fake_fetch(responses):
    def fetch(path):
        if path not in responses:
            raise AssertionError("unexpected path: {0}".format(path))
        return responses[path]
    return fetch


def merged_gw_csv(rows, gw_column="GW"):
    return pd.DataFrame(rows).rename(columns={"GW": gw_column}).to_csv(index=False).encode("utf-8")


def players_raw_csv(id_code_pairs):
    return pd.DataFrame(
        [{"id": i, "code": c} for i, c in id_code_pairs]
    ).to_csv(index=False).encode("utf-8")


def make_derived_db(path, players, gameweek_rows):
    """A minimal sqlite db carrying only the columns starts_model.py
    reads from `players` / `player_gameweek_stats` -- not a real
    derived.rebuild() output, which test_derived.py already covers.
    """
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE players (code INTEGER PRIMARY KEY, web_name TEXT)")
    conn.executemany("INSERT INTO players VALUES (?, ?)", players)
    conn.execute(
        "CREATE TABLE player_gameweek_stats (code, season, round, starts)"
    )
    conn.executemany(
        "INSERT INTO player_gameweek_stats VALUES (?, ?, ?, ?)", gameweek_rows
    )
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# load_prior_season_starts
# --------------------------------------------------------------------------


def test_load_prior_season_starts_joins_on_code_not_element():
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv([
            {"element": 1, "GW": 38, "starts": 1},
            {"element": 2, "GW": 38, "starts": 0},
        ]),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001), (2, 1002)]),
    })

    df = starts_model.load_prior_season_starts(fetch, PRIOR_SEASON)

    assert sorted(df["code"].tolist()) == [1001, 1002]
    assert int(df.loc[df["code"] == 1001, "y"].iloc[0]) == 1
    assert int(df.loc[df["code"] == 1002, "y"].iloc[0]) == 0


def test_load_prior_season_starts_falls_back_to_round_column():
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(
            [{"element": 1, "GW": 1, "starts": 1}], gw_column="round"
        ),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })

    df = starts_model.load_prior_season_starts(fetch, PRIOR_SEASON)

    assert df["GW"].iloc[0] == 1


# --------------------------------------------------------------------------
# load_current_season_starts
# --------------------------------------------------------------------------


def test_load_current_season_starts_reads_from_derived_db(tmp_path):
    conn = make_derived_db(str(tmp_path / "derived.db"), players=[(1001, "Alice")],
                            gameweek_rows=[(1001, SEASON, 1, 1), (1001, SEASON, 2, 0),
                                           (1001, PRIOR_SEASON, 1, 1)])

    df = starts_model.load_current_season_starts(conn, SEASON)
    conn.close()

    assert sorted(df["GW"].tolist()) == [1, 2]
    assert int(df.loc[df["GW"] == 2, "y"].iloc[0]) == 0


# --------------------------------------------------------------------------
# build_xseason_features -- continuity across the season boundary
# --------------------------------------------------------------------------


def test_features_are_continuous_across_season_boundary():
    """GW1 of the new season must see prior-season history in its prev/
    roll4, not NaN -- this is the entire point of the fix.
    """
    prior_df = pd.DataFrame({
        "code": [1001] * 4,
        "GW": [35, 36, 37, 38],
        "y": [1, 1, 0, 1],
    })
    current_df = pd.DataFrame({"code": [1001], "GW": [1], "y": [1]})

    combined = starts_model.build_xseason_features(prior_df, current_df)

    gw1_row = combined[(combined["code"] == 1001) & (combined["period"] == 39)]
    assert gw1_row["prev"].iloc[0] == 1        # GW38's outcome
    assert gw1_row["roll4"].iloc[0] == 0.75    # mean of GW35-38 = [1,1,0,1]


# --------------------------------------------------------------------------
# fit_lookup_table / predict_p_start
# --------------------------------------------------------------------------


def test_fit_and_predict_recovers_observed_frequencies():
    train = pd.DataFrame({
        "prev": [1] * 60 + [0] * 60,
        "roll4": [1.0] * 60 + [0.0] * 60,
        "y": [1] * 54 + [0] * 6 + [0] * 57 + [1] * 3,
    })
    fitted = starts_model.fit_lookup_table(train)

    test = pd.DataFrame({"prev": [1, 0], "roll4": [1.0, 0.0]})
    predictions = starts_model.predict_p_start(fitted, test, min_cell=50)

    assert predictions[0] == pytest.approx(0.90)
    assert predictions[1] == pytest.approx(0.05)


def test_predict_falls_back_to_two_number_table_for_sparse_cells():
    train = pd.DataFrame({
        "prev": [1, 1, 1],
        "roll4": [0.5, 0.5, 1.0],   # the roll4=0.5 cell only has 2 rows
        "y": [1, 0, 1],
    })
    fitted = starts_model.fit_lookup_table(train)

    test = pd.DataFrame({"prev": [1], "roll4": [0.5]})
    predictions = starts_model.predict_p_start(fitted, test, min_cell=50)

    assert predictions[0] == fitted.r1  # fell back, not the 2-row cell's mean


# --------------------------------------------------------------------------
# predict_gameweek
# --------------------------------------------------------------------------


def test_predict_gameweek_round_one_uses_prior_season_rate(tmp_path):
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv([
            {"element": 1, "GW": g, "starts": 1} for g in range(1, 5)
        ] + [
            {"element": 1, "GW": g, "starts": 0} for g in range(5, 39)
        ]),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(str(tmp_path / "derived.db"),
                            players=[(1001, "Alice"), (9999, "NewSigning")],
                            gameweek_rows=[])

    result = starts_model.predict_gameweek(
        conn, SEASON, PRIOR_SEASON, target_round=1, fetch=fetch
    )
    conn.close()

    alice = result[result["code"] == 1001].iloc[0]
    assert alice["p_start"] == pytest.approx(4 / 38)
    assert alice["cold_start"] == False
    assert alice["method"] == "prior_season_rate"

    new_signing = result[result["code"] == 9999].iloc[0]
    assert new_signing["cold_start"] == True
    assert new_signing["method"] == "cold_start_pool_rate"


def test_predict_gameweek_later_round_uses_xseason_lookup_and_flags_cold_start(tmp_path):
    rows = [{"element": 1, "GW": g, "starts": 1} for g in range(1, 39)]
    fetch = make_fake_fetch({
        PRIOR_SEASON + "/gws/merged_gw.csv": merged_gw_csv(rows),
        PRIOR_SEASON + "/players_raw.csv": players_raw_csv([(1, 1001)]),
    })
    conn = make_derived_db(
        str(tmp_path / "derived.db"),
        players=[(1001, "Alice"), (9999, "NewSigning")],
        gameweek_rows=[(1001, SEASON, 1, 1), (1001, SEASON, 2, 1)],
    )

    result = starts_model.predict_gameweek(
        conn, SEASON, PRIOR_SEASON, target_round=3, fetch=fetch
    )
    conn.close()

    alice = result[result["code"] == 1001].iloc[0]
    # Nailed-on every gameweek on both sides of the boundary -> high P(start).
    assert alice["p_start"] > 0.5
    assert alice["method"] == "cal_rolling_xseason"

    new_signing = result[result["code"] == 9999].iloc[0]
    assert new_signing["cold_start"] == True
    assert new_signing["n_observed"] == 0


# --------------------------------------------------------------------------
# snapshot_predictions
# --------------------------------------------------------------------------


def test_snapshot_predictions_write_once(tmp_path):
    predictions = pd.DataFrame([
        {"code": 1001, "web_name": "Alice", "p_start": 0.8, "cold_start": False,
         "n_observed": 10, "method": "cal_rolling_xseason"},
    ])
    base_dir = str(tmp_path / "predictions")
    clock = lambda: datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)

    path = starts_model.snapshot_predictions(
        predictions, SEASON, target_round=3, base_dir=base_dir, clock=clock
    )

    assert os.path.exists(path)
    with open(path) as f:
        payload = json.load(f)
    assert payload["season"] == SEASON
    assert payload["target_round"] == 3
    assert payload["predictions"][0]["web_name"] == "Alice"

    with pytest.raises(archiver.ArchiveError):
        starts_model.snapshot_predictions(
            predictions, SEASON, target_round=3, base_dir=base_dir, clock=clock
        )
