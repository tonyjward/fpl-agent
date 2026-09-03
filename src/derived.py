"""Derived layer: SQLite tables rebuilt by replaying the raw archive.

Per docs/build_spec_minutes_model.md Section 2.3a: the derived layer is
disposable and rebuilt freely, never written to directly or updated
incrementally. If a parsing bug is found here, delete the database and
rebuild -- the raw archive is unaffected and is the only thing that must
never be touched.

Only `player_gameweek_stats` (from event/{gw}/live) is built here. The
availability time series (status/chance/news/price per bootstrap-static
snapshot) is a separate, not-yet-built table -- see docs Section 2.3.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import gzip
import json
import os
import sqlite3

import archiver

DERIVED_DB_PATH = "derived.db"

SCHEMA = """
CREATE TABLE players (
    code INTEGER PRIMARY KEY,
    player_id INTEGER NOT NULL,
    web_name TEXT,
    team INTEGER,
    element_type INTEGER,
    season_minutes INTEGER,
    season_starts INTEGER
);

CREATE TABLE player_gameweek_stats (
    code INTEGER NOT NULL,
    season TEXT NOT NULL,
    round INTEGER NOT NULL,
    minutes INTEGER,
    starts INTEGER,
    total_points INTEGER,
    bps INTEGER,
    bonus INTEGER,
    goals_scored INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    own_goals INTEGER,
    penalties_saved INTEGER,
    penalties_missed INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    saves INTEGER,
    clearances_blocks_interceptions INTEGER,
    recoveries INTEGER,
    tackles INTEGER,
    defensive_contribution INTEGER,
    expected_goals REAL,
    expected_assists REAL,
    expected_goal_involvements REAL,
    expected_goals_conceded REAL,
    in_dreamteam INTEGER,
    played INTEGER,
    PRIMARY KEY (code, season, round),
    FOREIGN KEY (code) REFERENCES players(code)
);
"""

# stats fields copied straight from event-live's `elements[].stats`, in the
# order they're bound into the INSERT below. influence/creativity/threat/
# ict_index are excluded here even though event-live carries them -- they're
# derived composites, not raw match events, and can be added if a feature
# actually needs them.
GAMEWEEK_STAT_FIELDS = [
    "minutes", "starts", "total_points", "bps", "bonus",
    "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves",
    "clearances_blocks_interceptions", "recoveries", "tackles",
    "defensive_contribution",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded",
    "in_dreamteam", "played",
]


class CrossCheckError(Exception):
    """Derived per-gameweek totals don't match bootstrap-static's season
    totals for at least one player -- a gameweek was missed or
    double-counted somewhere in the raw archive or this rebuild.
    """


def _read_gz_json(path):
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _ok_entries(base_dir, season, endpoint):
    """`"ok"` manifest entries for one endpoint, in the order they were
    appended (i.e. oldest fetch first).
    """
    path = archiver.manifest_path(base_dir, season)
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("endpoint") == endpoint and entry.get("outcome") == "ok":
                entries.append(entry)
    return entries


def _latest_by(entries, key):
    """The entry with the largest `key(entry)`, or None if `entries` is
    empty. Timestamps are zero-padded ("YYYYMMDDTHHMMSSZ"), so string
    comparison already sorts them correctly -- no parsing needed.
    """
    latest = None
    for entry in entries:
        if latest is None or key(entry) > key(latest):
            latest = entry
    return latest


def _latest_per_round(entries):
    """One entry per `current_gw`, keeping the most recently fetched if a
    gameweek was archived more than once.
    """
    by_round = {}
    for entry in entries:
        gw = entry.get("current_gw")
        existing = by_round.get(gw)
        if existing is None or entry["fetched_at"] > existing["fetched_at"]:
            by_round[gw] = entry
    return by_round


def _load_players(conn, base_dir, season):
    """Populate `players` from the season's most recent bootstrap-static
    snapshot, and return the id -> code mapping for that snapshot (ids are
    only stable *within* a season, per docs Section 2.3a -- always join
    event-live's `id` back to `code` through this, never across seasons).
    """
    entries = _ok_entries(base_dir, season, "bootstrap-static")
    latest = _latest_by(entries, lambda e: e["fetched_at"])
    if latest is None:
        return {}

    payload = _read_gz_json(latest["path"])
    id_to_code = {}
    rows = []
    for element in payload.get("elements") or []:
        code = element["code"]
        player_id = element["id"]
        id_to_code[player_id] = code
        rows.append((
            code, player_id, element.get("web_name"), element.get("team"),
            element.get("element_type"), element.get("minutes"),
            element.get("starts"),
        ))

    conn.executemany(
        "INSERT INTO players (code, player_id, web_name, team, "
        "element_type, season_minutes, season_starts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return id_to_code


def _load_gameweek_stats(conn, base_dir, season, id_to_code):
    entries = _ok_entries(base_dir, season, "event-live")
    columns = ", ".join(GAMEWEEK_STAT_FIELDS)
    placeholders = ", ".join(["?"] * len(GAMEWEEK_STAT_FIELDS))
    insert_sql = (
        "INSERT INTO player_gameweek_stats (code, season, round, " + columns +
        ") VALUES (?, ?, ?, " + placeholders + ")"
    )

    for entry in _latest_per_round(entries).values():
        gw = entry["current_gw"]
        payload = _read_gz_json(entry["path"])
        rows = []
        for element in payload.get("elements") or []:
            code = id_to_code.get(element["id"])
            if code is None:
                # A player event-live reports on who bootstrap-static's
                # latest snapshot doesn't know about (e.g. removed from the
                # game since). Nothing to join to -- skip rather than guess.
                continue
            stats = element.get("stats") or {}
            rows.append(
                (code, season, gw) +
                tuple(stats.get(field) for field in GAMEWEEK_STAT_FIELDS)
            )
        conn.executemany(insert_sql, rows)


def _seasons_in_archive(base_dir):
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        name for name in os.listdir(base_dir)
        if os.path.isfile(os.path.join(base_dir, name, archiver.MANIFEST_FILENAME))
    )


def cross_check_season_totals(conn, season):
    """Assert every player's summed per-gameweek minutes/starts in the
    derived layer match bootstrap-static's season totals. Per
    docs/build_spec_minutes_model.md Section 2.5: "Any mismatch means a
    missed or double-counted gameweek and must fail the run."

    Only checks players with at least one archived gameweek row -- a player
    bootstrap-static reports minutes for but who never appears in any
    archived event-live file is a backfill gap, not a double-count, and is
    out of scope for this assertion.
    """
    cur = conn.execute(
        "SELECT p.code, p.web_name, p.season_minutes, p.season_starts, "
        "SUM(g.minutes), SUM(g.starts) "
        "FROM players p JOIN player_gameweek_stats g ON g.code = p.code "
        "WHERE g.season = ? "
        "GROUP BY p.code "
        "HAVING p.season_minutes != SUM(g.minutes) "
        "OR p.season_starts != SUM(g.starts)",
        (season,),
    )
    mismatches = cur.fetchall()
    if mismatches:
        lines = [
            "code={0} ({1}): season totals minutes={2} starts={3}, "
            "summed from archive minutes={4} starts={5}".format(*row)
            for row in mismatches
        ]
        raise CrossCheckError(
            "{0}: derived per-gameweek totals disagree with "
            "bootstrap-static's season totals for {1} player(s):\n{2}".format(
                season, len(mismatches), "\n".join(lines)
            )
        )


def build_season(conn, base_dir, season):
    id_to_code = _load_players(conn, base_dir, season)
    _load_gameweek_stats(conn, base_dir, season, id_to_code)


def rebuild(base_dir=archiver.RAW_DIR, db_path=DERIVED_DB_PATH, seasons=None):
    """Rebuild the derived database from scratch by replaying `base_dir`.
    Always a full rebuild, never an incremental update -- see module
    docstring. Returns the season(s) rebuilt.
    """
    if seasons is None:
        seasons = _seasons_in_archive(base_dir)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            "DROP TABLE IF EXISTS player_gameweek_stats;"
            "DROP TABLE IF EXISTS players;"
        )
        conn.executescript(SCHEMA)
        for season in seasons:
            build_season(conn, base_dir, season)
        conn.commit()
        for season in seasons:
            cross_check_season_totals(conn, season)
    finally:
        conn.close()
    return seasons


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild the derived SQLite layer from the raw archive."
    )
    parser.add_argument("--base-dir", default=archiver.RAW_DIR)
    parser.add_argument("--db-path", default=DERIVED_DB_PATH)
    args = parser.parse_args()

    # Run as if invoked from the repo root, regardless of where `python
    # derived.py` was actually launched from -- keeps raw/derived paths
    # consistent with every other entry point.
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(module_dir))
    seasons = rebuild(base_dir=args.base_dir, db_path=args.db_path)
    print("rebuilt {0} from {1}: {2}".format(
        args.db_path, args.base_dir, ", ".join(seasons) or "(no seasons found)"
    ))


if __name__ == "__main__":
    _main()
