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


def write_ok_entry(base_dir, season, endpoint, gw, timestamp, payload):
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
        "next_gw": None,
        "current_gw": gw,
        "next_deadline": None,
        "http_status": 200,
        "bytes_raw": len(content),
        "sha256": archiver.sha256_hex(content),
    }
    archiver.append_manifest_entry(base_dir, season, entry)
    return path


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

    seasons = derived.rebuild(base_dir=base_dir, db_path=db_path)
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


def test_rebuild_is_a_full_replace_not_an_append(tmp_path):
    base_dir = str(tmp_path / "raw")
    seed_consistent_archive(base_dir)
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(base_dir=base_dir, db_path=db_path)
    derived.rebuild(base_dir=base_dir, db_path=db_path)

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM player_gameweek_stats").fetchone()[0]
    conn.close()
    assert count == 4


def test_cross_check_passes_on_consistent_archive(tmp_path):
    base_dir = str(tmp_path / "raw")
    seed_consistent_archive(base_dir)
    db_path = str(tmp_path / "derived.db")

    derived.rebuild(base_dir=base_dir, db_path=db_path)  # must not raise


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
        derived.rebuild(base_dir=base_dir, db_path=db_path)


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

    derived.rebuild(base_dir=base_dir, db_path=db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT minutes, starts, total_points FROM player_gameweek_stats"
    ).fetchall()
    conn.close()
    assert rows == [(90, 1, 6)]


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

    derived.rebuild(base_dir=base_dir, db_path=db_path)

    conn = sqlite3.connect(db_path)
    codes = [row[0] for row in conn.execute("SELECT code FROM player_gameweek_stats")]
    conn.close()
    assert codes == [1001]


def test_rebuild_with_no_archive_produces_empty_tables(tmp_path):
    base_dir = str(tmp_path / "raw")
    db_path = str(tmp_path / "derived.db")

    seasons = derived.rebuild(base_dir=base_dir, db_path=db_path)

    assert seasons == []
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweek_stats"
    ).fetchone()[0] == 0
    conn.close()
