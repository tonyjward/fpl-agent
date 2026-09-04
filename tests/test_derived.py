"""Tests for derived.py: the SQLite layer rebuilt by replaying raw/.

Raw fixtures are written directly (via write_ok_entry below) rather than
through archiver.archive_snapshot, so these tests aren't coupled to the
plausibility gate -- derived.py only cares that the manifest and files
exist, not how they got there.
"""

import gzip
import json
import os
import sqlite3
from datetime import datetime, timezone

import pytest

import archiver
import derived

SEASON = "2026-27"


def write_ok_entry(base_dir, season, endpoint, gw, timestamp, payload,
                    next_gw=None, next_deadline=None):
    content = json.dumps(payload).encode("utf-8")
    path = archiver.snapshot_path(base_dir, season, endpoint, gw, timestamp)
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with gzip.open(path, "wb") as f:
        f.write(content)
    entry = {
        "outcome": "ok",
        "path": path,
        "endpoint": endpoint,
        "season": season,
        "fetched_at": archiver.format_timestamp(timestamp),
        "next_gw": next_gw,
        "current_gw": gw,
        "next_deadline": next_deadline,
        "http_status": 200,
        "bytes_raw": len(content),
        "sha256": archiver.sha256_hex(content),
    }
    archiver.append_manifest_entry(base_dir, season, entry)
    return path


def make_availability_player(code, player_id, web_name, status="a",
                              chance_this=None, chance_next=None, news="",
                              news_added=None, now_cost=55,
                              selected_by_percent="10.0", minutes=90, starts=1,
                              team=1, element_type=3):
    return {
        "code": code, "id": player_id, "web_name": web_name, "team": team,
        "element_type": element_type, "minutes": minutes, "starts": starts,
        "status": status, "chance_of_playing_this_round": chance_this,
        "chance_of_playing_next_round": chance_next, "news": news,
        "news_added": news_added, "now_cost": now_cost,
        "selected_by_percent": selected_by_percent,
    }


def make_teams(*id_code_pairs):
    return [{"id": team_id, "code": code} for team_id, code in id_code_pairs]


def make_player(code, player_id, web_name, minutes, starts, team=1, element_type=3):
    return {
        "code": code, "id": player_id, "web_name": web_name, "team": team,
        "element_type": element_type, "minutes": minutes, "starts": starts,
    }


def make_stats(minutes, starts, total_points=0, bps=0):
    return {
        "minutes": minutes, "starts": starts, "total_points": total_points,
        "bps": bps, "bonus": 0, "goals_scored": 0, "assists": 0,
        "clean_sheets": 0, "goals_conceded": 0, "own_goals": 0,
        "penalties_saved": 0, "penalties_missed": 0, "yellow_cards": 0,
        "red_cards": 0, "saves": 0, "clearances_blocks_interceptions": 0,
        "recoveries": 0, "tackles": 0, "defensive_contribution": 0,
        "expected_goals": "0.00", "expected_assists": "0.00",
        "expected_goal_involvements": "0.00", "expected_goals_conceded": "0.00",
        "in_dreamteam": False, "played": minutes > 0,
    }


T1 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


def seed_consistent_archive(base_dir):
    """Two players, two archived gameweeks, season totals that agree with
    the per-gameweek sums -- the passing baseline every test starts from.
    """
    bootstrap_payload = {
        "teams": make_teams((1, 3)),
        "elements": [
            make_player(1001, 1, "Alice", minutes=180, starts=2),
            make_player(1002, 2, "Bob", minutes=90, starts=1),
        ],
    }
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, T3, bootstrap_payload)

    write_ok_entry(base_dir, SEASON, "event-live", 1, T1, {
        "elements": [
            {"id": 1, "stats": make_stats(90, 1, total_points=6)},
            {"id": 2, "stats": make_stats(90, 1, total_points=2)},
        ],
    })
    write_ok_entry(base_dir, SEASON, "event-live", 2, T2, {
        "elements": [
            {"id": 1, "stats": make_stats(90, 1, total_points=8)},
            {"id": 2, "stats": make_stats(0, 0, total_points=0)},
        ],
    })


def test_rebuild_loads_players_and_gameweek_stats(tmp_path):
    base_dir = str(tmp_path / "raw")
    seed_consistent_archive(base_dir)
    db_path = str(tmp_path / "derived.db")

    seasons = derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )
    assert seasons == [SEASON]

    conn = sqlite3.connect(db_path)
    players = conn.execute(
        "SELECT code, web_name, season_minutes, season_starts "
        "FROM players ORDER BY code"
    ).fetchall()
    assert players == [
        (1001, "Alice", 180, 2),
        (1002, "Bob", 90, 1),
    ]

    rows = conn.execute(
        "SELECT code, round, minutes, starts, total_points "
        "FROM player_gameweek_stats ORDER BY code, round"
    ).fetchall()
    assert rows == [
        (1001, 1, 90, 1, 6),
        (1001, 2, 90, 1, 8),
        (1002, 1, 90, 1, 2),
        (1002, 2, 0, 0, 0),
    ]
    conn.close()


def test_teams_table_and_player_team_code_resolve_through_same_snapshot(tmp_path):
    """players.team_code must be the stable code, resolved via that same
    bootstrap-static snapshot's own teams array -- never the raw, season-
    relative team id (see the docstrings on _load_teams/_load_players for
    why: team id 3 this season need not be the same club next season).
    """
    base_dir = str(tmp_path / "raw")
    bootstrap_payload = {
        "teams": make_teams((1, 3), (2, 7)),  # id 1 = Arsenal (code 3)
        "elements": [
            make_player(1001, 1, "Alice", minutes=90, starts=1, team=1),
        ],
    }
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 1, T3, bootstrap_payload)
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    teams = conn.execute("SELECT code FROM teams ORDER BY code").fetchall()
    team_code = conn.execute(
        "SELECT team_code FROM players WHERE code = 1001"
    ).fetchone()[0]
    conn.close()

    assert teams == [(3,), (7,)]
    assert team_code == 3


def test_rebuild_is_a_full_replace_not_an_append(tmp_path):
    base_dir = str(tmp_path / "raw")
    seed_consistent_archive(base_dir)
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )
    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM player_gameweek_stats").fetchone()[0]
    conn.close()
    assert count == 4


def test_cross_check_passes_on_consistent_archive(tmp_path):
    base_dir = str(tmp_path / "raw")
    seed_consistent_archive(base_dir)
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )  # must not raise


def test_cross_check_fails_on_missed_gameweek(tmp_path):
    """bootstrap-static's season total for Alice (270 minutes, 3 starts)
    implies a third archived gameweek that was never written -- summing the
    two that exist gives only 180/2, so this must fail closed.
    """
    base_dir = str(tmp_path / "raw")
    bootstrap_payload = {
        "elements": [make_player(1001, 1, "Alice", minutes=270, starts=3)],
    }
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 4, T3, bootstrap_payload)
    write_ok_entry(base_dir, SEASON, "event-live", 1, T1, {
        "elements": [{"id": 1, "stats": make_stats(90, 1)}],
    })
    write_ok_entry(base_dir, SEASON, "event-live", 2, T2, {
        "elements": [{"id": 1, "stats": make_stats(90, 1)}],
    })
    db_path = str(tmp_path / "derived.db")

    with pytest.raises(derived.CrossCheckError):
        derived.rebuild(
            base_dir=base_dir, db_path=db_path,
            predictions_dir=str(tmp_path / "predictions"),
        )


def test_event_live_reuses_most_recent_fetch_for_a_round(tmp_path):
    """A gameweek fetched twice (e.g. a rerun) must contribute exactly one
    row to player_gameweek_stats, from the latest fetch.
    """
    base_dir = str(tmp_path / "raw")
    bootstrap_payload = {
        "elements": [make_player(1001, 1, "Alice", minutes=90, starts=1)],
    }
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 2, T3, bootstrap_payload)
    write_ok_entry(base_dir, SEASON, "event-live", 1, T1, {
        "elements": [{"id": 1, "stats": make_stats(45, 0, total_points=1)}],
    })
    later = datetime(2026, 8, 20, 18, 0, 0, tzinfo=timezone.utc)
    write_ok_entry(base_dir, SEASON, "event-live", 1, later, {
        "elements": [{"id": 1, "stats": make_stats(90, 1, total_points=6)}],
    })
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT minutes, starts, total_points FROM player_gameweek_stats"
    ).fetchall()
    conn.close()
    assert rows == [(90, 1, 6)]


def test_gameweek_stats_team_code_reflects_team_at_time_of_transfer(tmp_path):
    """A player who transfers between two gameweeks must have each
    player_gameweek_stats row attributed to the club they were actually on
    when that gameweek was played -- not players.team_code, which only
    ever holds the latest-known club.
    """
    base_dir = str(tmp_path / "raw")
    pull_before_transfer = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
    pull_after_transfer = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    gw1_live = datetime(2026, 8, 21, 20, 0, 0, tzinfo=timezone.utc)
    gw2_live = datetime(2026, 9, 1, 20, 0, 0, tzinfo=timezone.utc)
    teams = make_teams((1, 3), (2, 7))

    write_ok_entry(base_dir, SEASON, "bootstrap-static", 2, pull_before_transfer, {
        "teams": teams,
        "elements": [make_player(1001, 1, "Alice", minutes=90, starts=1, team=1)],
    })
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, pull_after_transfer, {
        "teams": teams,
        "elements": [make_player(1001, 1, "Alice", minutes=180, starts=2, team=2)],
    })
    write_ok_entry(base_dir, SEASON, "event-live", 1, gw1_live, {
        "elements": [{"id": 1, "stats": make_stats(90, 1)}],
    })
    write_ok_entry(base_dir, SEASON, "event-live", 2, gw2_live, {
        "elements": [{"id": 1, "stats": make_stats(90, 1)}],
    })
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT round, team_code FROM player_gameweek_stats "
        "WHERE code = 1001 ORDER BY round"
    ).fetchall()
    latest_team_code = conn.execute(
        "SELECT team_code FROM players WHERE code = 1001"
    ).fetchone()[0]
    conn.close()

    assert rows == [(1, 3), (2, 7)]
    assert latest_team_code == 7  # confirms this isn't what the gw1 row copied


def test_event_live_player_missing_from_latest_bootstrap_is_skipped(tmp_path):
    """A player event-live reports on who the latest bootstrap-static
    snapshot doesn't carry (id/code mapping unavailable) is dropped rather
    than guessed at.
    """
    base_dir = str(tmp_path / "raw")
    bootstrap_payload = {
        "elements": [make_player(1001, 1, "Alice", minutes=90, starts=1)],
    }
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 1, T3, bootstrap_payload)
    write_ok_entry(base_dir, SEASON, "event-live", 1, T1, {
        "elements": [
            {"id": 1, "stats": make_stats(90, 1)},
            {"id": 999, "stats": make_stats(90, 1)},
        ],
    })
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    codes = [row[0] for row in conn.execute("SELECT code FROM player_gameweek_stats")]
    conn.close()
    assert codes == [1001]


def test_rebuild_with_no_archive_produces_empty_tables(tmp_path):
    base_dir = str(tmp_path / "raw")
    db_path = str(tmp_path / "derived.db")

    seasons = derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    assert seasons == []
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweek_stats"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM player_availability_snapshots"
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    conn.close()


def write_predictions_snapshot(predictions_dir, season, target_round, predicted_at, rows,
                                model_version="raw_lookup"):
    directory = os.path.join(predictions_dir, season)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    path = os.path.join(
        directory, "gw{0:02d}_{1}_{2}.json".format(target_round, model_version, predicted_at)
    )
    payload = {
        "season": season, "target_round": target_round, "model_version": model_version,
        "predicted_at": predicted_at, "predictions": rows,
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def test_predictions_table_keeps_only_the_latest_snapshot_per_gameweek(tmp_path):
    """A gameweek predicted twice (e.g. a rerun mid-week) must contribute
    exactly one set of rows to `predictions` -- from the latest snapshot,
    same dedup logic as event-live reruns in player_gameweek_stats.
    """
    base_dir = str(tmp_path / "raw")
    seed_consistent_archive(base_dir)
    db_path = str(tmp_path / "derived.db")
    predictions_dir = str(tmp_path / "predictions")

    write_predictions_snapshot(predictions_dir, SEASON, 3, "20260903T090000Z", [
        {"code": 1001, "p_start": 0.5, "cold_start": False, "n_observed": 10,
         "method": "cal_rolling_xseason"},
    ])
    write_predictions_snapshot(predictions_dir, SEASON, 3, "20260903T180000Z", [
        {"code": 1001, "p_start": 0.8, "cold_start": False, "n_observed": 10,
         "method": "cal_rolling_xseason"},
    ])
    write_predictions_snapshot(predictions_dir, SEASON, 4, "20260903T090000Z", [
        {"code": 1001, "p_start": 0.6, "cold_start": True, "n_observed": 0,
         "method": "cold_start_pool_rate"},
    ])

    derived.rebuild(
        base_dir=base_dir, db_path=db_path, predictions_dir=predictions_dir,
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT target_round, predicted_at, p_start, cold_start "
        "FROM predictions WHERE code = 1001 ORDER BY target_round"
    ).fetchall()
    conn.close()

    assert rows == [
        (3, "20260903T180000Z", 0.8, 0),   # latest of the two GW3 snapshots
        (4, "20260903T090000Z", 0.6, 1),
    ]


def test_predictions_dedup_is_per_model_version_not_just_round(tmp_path):
    """Two different model_versions for the same gameweek (e.g. raw_lookup
    and refined_availability) must both survive -- dedup only collapses
    reruns of the *same* model_version, not different ones.
    """
    base_dir = str(tmp_path / "raw")
    seed_consistent_archive(base_dir)
    db_path = str(tmp_path / "derived.db")
    predictions_dir = str(tmp_path / "predictions")

    write_predictions_snapshot(
        predictions_dir, SEASON, 3, "20260903T090000Z",
        [{"code": 1001, "p_start": 0.5, "cold_start": False, "n_observed": 10,
          "method": "cal_rolling_xseason"}],
        model_version="raw_lookup",
    )
    write_predictions_snapshot(
        predictions_dir, SEASON, 3, "20260903T090000Z",
        [{"code": 1001, "p_start": 0.0, "cold_start": False, "n_observed": 10,
          "method": "hard_gate_unavailable"}],
        model_version="refined_availability",
    )

    derived.rebuild(base_dir=base_dir, db_path=db_path, predictions_dir=predictions_dir)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT model_version, p_start, method FROM predictions "
        "WHERE code = 1001 ORDER BY model_version"
    ).fetchall()
    conn.close()

    assert rows == [
        ("raw_lookup", 0.5, "cal_rolling_xseason"),
        ("refined_availability", 0.0, "hard_gate_unavailable"),
    ]


def test_availability_snapshots_keep_every_pull_not_just_the_latest(tmp_path):
    """Deadline-day is expected to be pulled multiple times specifically to
    catch late-breaking news -- every bootstrap-static snapshot must land
    as its own row, not just whichever one happens to be most recent.
    """
    base_dir = str(tmp_path / "raw")
    morning = datetime(2026, 9, 4, 9, 0, 0, tzinfo=timezone.utc)
    afternoon = datetime(2026, 9, 4, 16, 0, 0, tzinfo=timezone.utc)

    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, morning, {
        "elements": [make_availability_player(
            1001, 1, "Alice", status="d", chance_next=50,
            news="Ankle knock, assessed after training",
        )],
    }, next_gw=3, next_deadline="2026-09-04T17:30:00Z")
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, afternoon, {
        "elements": [make_availability_player(
            1001, 1, "Alice", status="a", chance_next=100, news="",
        )],
    }, next_gw=3, next_deadline="2026-09-04T17:30:00Z")
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT fetched_at, status, chance_of_playing_next_round, news, "
        "next_gw, next_deadline, selected_by_percent "
        "FROM player_availability_snapshots WHERE code = 1001 ORDER BY fetched_at"
    ).fetchall()
    conn.close()

    assert rows == [
        ("20260904T090000Z", "d", 50, "Ankle knock, assessed after training",
         3, "2026-09-04T17:30:00Z", 10.0),
        ("20260904T160000Z", "a", 100, "", 3, "2026-09-04T17:30:00Z", 10.0),
    ]


def test_availability_snapshots_resolve_team_code_and_track_transfers(tmp_path):
    """team_code is resolved through that snapshot's own teams array (never
    the raw, season-relative team id), and a transfer between two archived
    snapshots shows up as a change in team_code on the next row -- the
    reconstruction this table exists for.
    """
    base_dir = str(tmp_path / "raw")
    before = datetime(2026, 8, 20, 9, 0, 0, tzinfo=timezone.utc)
    after = datetime(2026, 9, 1, 9, 0, 0, tzinfo=timezone.utc)

    write_ok_entry(base_dir, SEASON, "bootstrap-static", 2, before, {
        "teams": make_teams((1, 3), (2, 7)),  # id 1 = Arsenal (code 3), id 2 = Aston Villa (code 7)
        "elements": [make_availability_player(
            1001, 1, "Alice", team=1, element_type=2,
        )],
    })
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, after, {
        "teams": make_teams((1, 3), (2, 7)),
        "elements": [make_availability_player(
            1001, 1, "Alice", team=2, element_type=4,  # transferred, reclassified
        )],
    })
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT fetched_at, team_code, element_type "
        "FROM player_availability_snapshots WHERE code = 1001 ORDER BY fetched_at"
    ).fetchall()
    conn.close()

    assert rows == [
        ("20260820T090000Z", 3, 2),  # Arsenal, defender
        ("20260901T090000Z", 7, 4),  # Aston Villa, forward
    ]


# --------------------------------------------------------------------------
# match_team_code / match_injury_player -- pure matching logic
# --------------------------------------------------------------------------

ALL_TEAMS = [
    (3, "Arsenal", "ARS"), (7, "Aston Villa", "AVL"), (36, "Brighton", "BHA"),
    (43, "Man City", "MCI"), (1, "Man Utd", "MUN"), (6, "Spurs", "TOT"),
    (17, "Nott'm Forest", "NFO"), (2, "Leeds", "LEE"), (4, "Newcastle", "NEW"),
]


def test_match_team_code_exact_and_substring():
    assert derived.match_team_code("Arsenal", ALL_TEAMS) == 3
    assert derived.match_team_code("Brighton & Hove Albion", ALL_TEAMS) == 36
    assert derived.match_team_code("Leeds United", ALL_TEAMS) == 2
    assert derived.match_team_code("Newcastle United", ALL_TEAMS) == 4


def test_match_team_code_uses_alias_for_nicknames_with_no_shared_substring():
    assert derived.match_team_code("Manchester City", ALL_TEAMS) == 43
    assert derived.match_team_code("Manchester United", ALL_TEAMS) == 1
    assert derived.match_team_code("Tottenham Hotspur", ALL_TEAMS) == 6
    assert derived.match_team_code("Nottingham Forest", ALL_TEAMS) == 17


def test_match_team_code_returns_none_when_no_team_matches():
    assert derived.match_team_code("Some Championship Club", ALL_TEAMS) is None


def test_match_team_code_short_code_does_not_falsely_substring_match():
    """Confirmed live against 2026-27 data: Chelsea's short_name "CHE" is a
    substring of "manCHEster", which without a minimum length on the
    containment check falsely matched both "Manchester City" and
    "Manchester United" to Chelsea too, making the match ambiguous (and so
    silently returning None instead of the real team).
    """
    teams = ALL_TEAMS + [(8, "Chelsea", "CHE")]
    assert derived.match_team_code("Manchester City", teams) == 43
    assert derived.match_team_code("Manchester United", teams) == 1


PLAYERS = [
    (9001, "Saliba", 3),      # Arsenal
    (9002, "Timber", 3),      # Arsenal
    (9003, "Rodon", 2),       # Leeds
    (9004, "Silva", 3),       # Arsenal -- shares a surname with the Man City one
    (9005, "Silva", 43),      # Man City
]


def test_match_injury_player_matches_within_the_claimed_club():
    code, team_code, contradiction = derived.match_injury_player(
        "William Saliba", club_team_code=3, players=PLAYERS,
    )
    assert (code, team_code, contradiction) == (9001, 3, False)


def test_match_injury_player_strips_disambiguating_initial_within_scope():
    """Confirmed live against 2026-27 data: FPL lists Arsenal's Jurrien
    Timber as web_name "J.Timber" specifically because a *different*
    player (Crystal Palace's) is already bare "Timber". Without stripping
    the initial, the claimed-club scoped match for Arsenal fails and falls
    through to matching Palace's unrelated Timber instead -- a false
    contradiction this test guards against.
    """
    players = [
        (9001, "J.Timber", 3),   # Arsenal -- disambiguated
        (9002, "Timber", 31),    # Crystal Palace -- a different player
    ]
    code, team_code, contradiction = derived.match_injury_player(
        "Jurrien Timber", club_team_code=3, players=players,
    )
    assert (code, team_code, contradiction) == (9001, 3, False)


def test_match_injury_player_matches_a_full_name_web_name():
    """Confirmed live against 2026-27 data: some disambiguated players get
    FPL's full "First Last" as web_name instead of an initial ("Chadi
    Riad", not "Riad") -- single-token surname matching alone never finds
    these at all.
    """
    players = [(9006, "Chadi Riad", 31)]
    code, team_code, contradiction = derived.match_injury_player(
        "Chadi Riad", club_team_code=31, players=players,
    )
    assert (code, team_code, contradiction) == (9006, 31, False)


def test_match_injury_player_strips_repeated_initials():
    """"P.M.Sarr" (double initial) needs the same stripping as a single
    initial -- confirmed live against 2026-27 data (Tottenham's Pape Matar
    Sarr, disambiguated from two other same-surname Sarrs elsewhere).
    """
    players = [(9007, "P.M.Sarr", 6)]
    code, team_code, contradiction = derived.match_injury_player(
        "Pape Matar Sarr", club_team_code=6, players=players,
    )
    assert (code, team_code, contradiction) == (9007, 6, False)


def test_match_injury_player_word_boundary_avoids_partial_word_match():
    """A short web_name must not match as a mere substring of an unrelated
    longer word in the player name.
    """
    players = [(9008, "Sarr", 31)]
    code, team_code, contradiction = derived.match_injury_player(
        "Alassane Sarraf", club_team_code=31, players=players,
    )
    assert (code, team_code, contradiction) == (None, None, False)


def test_match_injury_player_matches_across_accent_differences():
    """Confirmed live against 2026-27 data: CMS text renders a name in
    plain ASCII ("Sangare") where FPL's own web_name carries the accent
    ("I.Sangaré").
    """
    players = [(9009, "I.Sangaré", 17)]
    code, team_code, contradiction = derived.match_injury_player(
        "Ibrahim Sangare", club_team_code=17, players=players,
    )
    assert (code, team_code, contradiction) == (9009, 17, False)


def test_match_injury_player_flags_contradiction_when_matched_elsewhere():
    """The exact loan-transfer scenario docs/CLAUDE.md records: the CMS
    still lists a player under a club he's no longer registered to, per
    this project's own (fresher) players table.
    """
    code, team_code, contradiction = derived.match_injury_player(
        "Joe Rodon", club_team_code=3, players=PLAYERS,  # claimed: Arsenal
    )
    assert (code, team_code, contradiction) == (9003, 2, True)  # actually Leeds


def test_match_injury_player_leaves_ambiguous_surname_unmatched():
    """Two different current players share the surname "Silva" at two
    different clubs, and neither claimed club (a third club here) resolves
    the ambiguity -- guessing would risk a false contradiction flag.
    """
    code, team_code, contradiction = derived.match_injury_player(
        "Bernardo Silva", club_team_code=99, players=PLAYERS,
    )
    assert (code, team_code, contradiction) == (None, None, False)


def test_match_injury_player_no_match_is_not_a_contradiction():
    code, team_code, contradiction = derived.match_injury_player(
        "Nobody Recognisable", club_team_code=3, players=PLAYERS,
    )
    assert (code, team_code, contradiction) == (None, None, False)


# --------------------------------------------------------------------------
# _load_pl_news / _load_pl_injuries -- replaying archived pl-news/pl-injuries
# --------------------------------------------------------------------------


def make_named_teams(*rows):
    """Like make_teams, but carrying the name/short_name fields
    match_team_code needs -- make_teams itself omits them since none of the
    tests it originally served join on team name.
    """
    return [
        {"id": team_id, "code": code, "name": name, "short_name": short_name}
        for team_id, code, name, short_name in rows
    ]


def write_pl_news_entry(base_dir, season, gw, timestamp, articles, external=None):
    payload = {
        "list": {"content": [{"id": aid} for aid in articles]},
        "articles": articles,
        "external": external or {},
    }
    return write_ok_entry(base_dir, season, "pl-news", gw, timestamp, payload)


def write_pl_injuries_entry(base_dir, season, gw, timestamp, hub_items, clubs):
    payload = {"hub": {"items": hub_items}, "clubs": clubs}
    return write_ok_entry(base_dir, season, "pl-injuries", gw, timestamp, payload)


def test_load_pl_news_derives_body_text_from_native_body_and_external_text(tmp_path):
    base_dir = str(tmp_path / "raw")
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, T3, {
        "teams": make_teams((1, 3)),
        "elements": [make_player(1001, 1, "Alice", minutes=90, starts=1)],
    })
    write_pl_news_entry(
        base_dir, SEASON, 3, T3,
        articles={
            "1": {
                "id": 1, "title": "Native", "platform": "PULSE_CMS",
                "date": "2026-09-04T09:00:00Z", "hotlinkUrl": None,
                "body": "<p>" + "Native article prose here. " * 3 + "</p>",
            },
            "2": {
                "id": 2, "title": "Syndicated", "platform": "RSS",
                "date": "2026-09-04T10:00:00Z",
                "hotlinkUrl": "https://club.example/a", "body": None,
                "description": "short teaser",
            },
        },
        external={
            "2": {
                "url": "https://club.example/a", "method": "http",
                "text": "External article prose here too, already extracted.",
            },
        },
    )
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    rows = dict(conn.execute(
        "SELECT article_id, body_text FROM news_articles ORDER BY article_id"
    ).fetchall())
    conn.close()

    assert "Native article prose" in rows[1]
    assert "External article prose" in rows[2]


def test_load_pl_news_falls_back_to_description_with_no_html_at_all(tmp_path):
    base_dir = str(tmp_path / "raw")
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, T3, {
        "teams": make_teams((1, 3)),
        "elements": [make_player(1001, 1, "Alice", minutes=90, starts=1)],
    })
    write_pl_news_entry(
        base_dir, SEASON, 3, T3,
        articles={
            "1": {
                "id": 1, "title": "Live blog", "platform": "PULSE_CMS",
                "date": None, "hotlinkUrl": None, "body": None,
                "description": "short teaser",
            },
        },
    )
    db_path = str(tmp_path / "derived.db")
    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    body_text = conn.execute(
        "SELECT body_text FROM news_articles WHERE article_id = 1"
    ).fetchone()[0]
    conn.close()
    assert body_text == "short teaser"


def test_load_pl_injuries_resolves_club_and_player_with_no_contradiction(tmp_path):
    base_dir = str(tmp_path / "raw")
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, T3, {
        "teams": make_named_teams((1, 3, "Arsenal", "ARS")),
        "elements": [make_player(1001, 1, "Saliba", minutes=90, starts=1, team=1)],
    })
    write_pl_injuries_entry(
        base_dir, SEASON, 3, T3,
        hub_items=[{"response": {"id": 100, "title": "Injury News - Arsenal"}}],
        clubs={"100": {"items": [
            {"response": {
                "type": "promo", "title": "William Saliba",
                "description": "Back",
                "links": [{"promoUrl": "https://www.arsenal.com/x"}],
            }},
        ]}},
    )
    db_path = str(tmp_path / "derived.db")
    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT club_name, club_team_code, player_name, injury, "
        "matched_code, matched_team_code, contradiction FROM injury_reports"
    ).fetchone()
    conn.close()

    assert row == ("Arsenal", 3, "William Saliba", "Back", 1001, 3, 0)


def test_load_pl_injuries_flags_contradiction_against_current_squad(tmp_path):
    """The loan-transfer scenario end to end: the injury hub still lists a
    player under his old club's section, but bootstrap-static (fetched more
    recently, in the same archive) already has him at his new one.
    """
    base_dir = str(tmp_path / "raw")
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, T3, {
        "teams": make_named_teams(
            (1, 3, "Arsenal", "ARS"), (2, 7, "Aston Villa", "AVL"),
        ),
        "elements": [make_player(1001, 1, "Rodon", minutes=90, starts=1, team=2)],
    })
    write_pl_injuries_entry(
        base_dir, SEASON, 3, T3,
        hub_items=[{"response": {"id": 100, "title": "Injury News - Arsenal"}}],
        clubs={"100": {"items": [
            {"response": {
                "type": "promo", "title": "Joe Rodon", "description": "Hamstring",
                "links": [],
            }},
        ]}},
    )
    db_path = str(tmp_path / "derived.db")
    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT club_team_code, matched_code, matched_team_code, contradiction "
        "FROM injury_reports WHERE player_name = 'Joe Rodon'"
    ).fetchone()
    conn.close()

    assert row == (3, 1001, 7, 1)


def test_load_pl_injuries_skips_clubs_with_fetch_errors(tmp_path):
    base_dir = str(tmp_path / "raw")
    write_ok_entry(base_dir, SEASON, "bootstrap-static", 3, T3, {
        "teams": make_teams((1, 3)),
        "elements": [make_player(1001, 1, "Saliba", minutes=90, starts=1, team=1)],
    })
    write_pl_injuries_entry(
        base_dir, SEASON, 3, T3,
        hub_items=[{"response": {"id": 100, "title": "Injury News - Arsenal"}}],
        clubs={"100": {"_fetch_error": "boom", "_club_name": "Arsenal"}},
    )
    db_path = str(tmp_path / "derived.db")
    derived.rebuild(
        base_dir=base_dir, db_path=db_path,
        predictions_dir=str(tmp_path / "predictions"),
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM injury_reports").fetchone()[0]
    conn.close()
    assert count == 0
