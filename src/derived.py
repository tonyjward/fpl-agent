"""Derived layer: SQLite tables rebuilt by replaying the raw archive.

Per docs/build_spec_minutes_model.md Section 2.3a: the derived layer is
disposable and rebuilt freely, never written to directly or updated
incrementally. If a parsing bug is found here, delete the database and
rebuild -- the raw archive is unaffected and is the only thing that must
never be touched.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import gzip
import json
import os
import sqlite3
from datetime import datetime

import archiver

DERIVED_DB_PATH = "derived.db"
PREDICTIONS_DIR = "predictions"

SCHEMA = """
CREATE TABLE teams (
    code INTEGER PRIMARY KEY,
    name TEXT,
    short_name TEXT
);

CREATE TABLE players (
    code INTEGER PRIMARY KEY,
    player_id INTEGER NOT NULL,
    web_name TEXT,
    team_code INTEGER,
    element_type INTEGER,
    season_minutes INTEGER,
    season_starts INTEGER,
    FOREIGN KEY (team_code) REFERENCES teams(code)
);

CREATE TABLE player_gameweek_stats (
    code INTEGER NOT NULL,
    season TEXT NOT NULL,
    round INTEGER NOT NULL,
    team_code INTEGER,
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
    FOREIGN KEY (code) REFERENCES players(code),
    FOREIGN KEY (team_code) REFERENCES teams(code)
);

CREATE TABLE player_availability_snapshots (
    code INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    season TEXT NOT NULL,
    next_gw INTEGER,
    next_deadline TEXT,
    status TEXT,
    chance_of_playing_this_round INTEGER,
    chance_of_playing_next_round INTEGER,
    news TEXT,
    news_added TEXT,
    now_cost INTEGER,
    selected_by_percent REAL,
    team_code INTEGER,
    element_type INTEGER,
    PRIMARY KEY (code, fetched_at),
    FOREIGN KEY (code) REFERENCES players(code)
);

CREATE TABLE predictions (
    code INTEGER NOT NULL,
    season TEXT NOT NULL,
    target_round INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    p_start REAL,
    cold_start INTEGER,
    n_observed INTEGER,
    method TEXT,
    PRIMARY KEY (code, season, target_round, model_version),
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


def _latest_bootstrap_payload(base_dir, season):
    entries = _ok_entries(base_dir, season, "bootstrap-static")
    latest = _latest_by(entries, lambda e: e["fetched_at"])
    if latest is None:
        return None
    return _read_gz_json(latest["path"])


def _load_teams(conn, base_dir, season):
    """Populate `teams` from the season's most recent bootstrap-static
    snapshot, keyed on the stable `code` -- never the raw `id`, which is
    reassigned each season based on that season's promoted/relegated
    composition (team id 3 this season need not be the same club next
    season).
    """
    payload = _latest_bootstrap_payload(base_dir, season)
    if payload is None:
        return
    rows = [
        (team["code"], team.get("name"), team.get("short_name"))
        for team in payload.get("teams") or []
    ]
    conn.executemany(
        "INSERT INTO teams (code, name, short_name) VALUES (?, ?, ?)", rows
    )


def _load_players(conn, base_dir, season):
    """Populate `players` from the season's most recent bootstrap-static
    snapshot, and return the id -> code mapping for that snapshot (ids are
    only stable *within* a season, per docs Section 2.3a -- always join
    event-live's `id` back to `code` through this, never across seasons).

    `team_code` is resolved through that same snapshot's own `teams` array
    for the identical reason -- see `_load_teams`.
    """
    payload = _latest_bootstrap_payload(base_dir, season)
    if payload is None:
        return {}

    team_id_to_code = {
        team["id"]: team["code"] for team in payload.get("teams") or []
    }
    id_to_code = {}
    rows = []
    for element in payload.get("elements") or []:
        code = element["code"]
        player_id = element["id"]
        id_to_code[player_id] = code
        rows.append((
            code, player_id, element.get("web_name"),
            team_id_to_code.get(element.get("team")),
            element.get("element_type"), element.get("minutes"),
            element.get("starts"),
        ))

    conn.executemany(
        "INSERT INTO players (code, player_id, web_name, team_code, "
        "element_type, season_minutes, season_starts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return id_to_code


def _parse_fetched_at(fetched_at):
    return datetime.strptime(fetched_at, "%Y%m%dT%H%M%SZ")


def _team_code_history(base_dir, season):
    """code -> [(fetched_at, team_code), ...], sorted by fetched_at, built
    from every archived bootstrap-static pull. Lets a player_gameweek_stats
    row be attributed to whichever club the player was actually on when
    that gameweek was played, rather than players.team_code, which only
    ever holds the latest-known club.
    """
    entries = _ok_entries(base_dir, season, "bootstrap-static")
    history = {}
    for entry in entries:
        payload = _read_gz_json(entry["path"])
        team_id_to_code = {
            team["id"]: team["code"] for team in payload.get("teams") or []
        }
        for element in payload.get("elements") or []:
            code = element["code"]
            team_code = team_id_to_code.get(element.get("team"))
            history.setdefault(code, []).append((entry["fetched_at"], team_code))
    for rows in history.values():
        rows.sort()
    return history


def _closest_team_code(history_for_code, target_fetched_at):
    """The team_code from whichever bootstrap-static pull was closest in
    time to `target_fetched_at` -- a linear scan, not a bisect, since a
    season's worth of pulls per player is still a small list and rebuild
    cost isn't a concern here (see docs Section 2.3a, "cost is not a
    consideration").
    """
    if not history_for_code:
        return None
    target = _parse_fetched_at(target_fetched_at)
    closest = min(
        history_for_code,
        key=lambda row: abs((_parse_fetched_at(row[0]) - target).total_seconds()),
    )
    return closest[1]


def _load_gameweek_stats(conn, base_dir, season, id_to_code, team_code_history):
    entries = _ok_entries(base_dir, season, "event-live")
    columns = ", ".join(GAMEWEEK_STAT_FIELDS)
    placeholders = ", ".join(["?"] * len(GAMEWEEK_STAT_FIELDS))
    insert_sql = (
        "INSERT INTO player_gameweek_stats (code, season, round, team_code, " +
        columns + ") VALUES (?, ?, ?, ?, " + placeholders + ")"
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
            team_code = _closest_team_code(
                team_code_history.get(code, []), entry["fetched_at"]
            )
            rows.append(
                (code, season, gw, team_code) +
                tuple(stats.get(field) for field in GAMEWEEK_STAT_FIELDS)
            )
        conn.executemany(insert_sql, rows)


def _load_availability_snapshots(conn, base_dir, season):
    """One row per (code, fetched_at) for every archived bootstrap-static
    pull -- the full trajectory, not just the value closest to a deadline.
    Per docs/build_spec_minutes_model.md Section 2.3, these fields are live
    state with no history anywhere else, and deadline-day is expected to be
    pulled multiple times specifically to catch late-breaking news; keeping
    every snapshot is what makes that trajectory queryable later, rather
    than only keeping whichever pull happened to be picked as "the" one.

    `team_code` and `element_type` ride along in the same row so a club
    transfer or position reclassification is reconstructible at whatever
    resolution the archive was pulled at, the same way availability is --
    not a separate change-log, just two more fields on every snapshot.
    `team` is resolved to the stable `team_code` via that same payload's
    own `teams` array, since the raw `team` id is season-relative (see
    `_load_players`'s id -> code note; teams have the identical problem).
    """
    entries = _ok_entries(base_dir, season, "bootstrap-static")
    insert_sql = (
        "INSERT INTO player_availability_snapshots (code, fetched_at, "
        "season, next_gw, next_deadline, status, "
        "chance_of_playing_this_round, chance_of_playing_next_round, "
        "news, news_added, now_cost, selected_by_percent, team_code, "
        "element_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for entry in entries:
        payload = _read_gz_json(entry["path"])
        team_id_to_code = {
            team["id"]: team["code"] for team in payload.get("teams") or []
        }
        rows = []
        for element in payload.get("elements") or []:
            selected = element.get("selected_by_percent")
            rows.append((
                element["code"], entry["fetched_at"], season,
                entry.get("next_gw"), entry.get("next_deadline"),
                element.get("status"),
                element.get("chance_of_playing_this_round"),
                element.get("chance_of_playing_next_round"),
                element.get("news"), element.get("news_added"),
                element.get("now_cost"),
                float(selected) if selected is not None else None,
                team_id_to_code.get(element.get("team")),
                element.get("element_type"),
            ))
        conn.executemany(insert_sql, rows)


def _load_predictions(conn, predictions_dir, season):
    """One row per (code, season, target_round, model_version) -- the
    *latest* snapshot for each (gameweek, model_version) pair, not every
    run. Unlike the raw archive, an unchanged-inputs rerun of
    starts_model.py is a pure duplicate today (no new information), so
    keeping only the latest is a real simplification, not a loss: docs
    Section 8a's scoring loop wants "the prediction as it stood before
    kickoff" -- the last one -- and every run is still on disk under
    predictions/ if a future method (e.g. one that reacts to mid-week news)
    ever makes reruns genuinely differ and that history needs mining.

    `model_version` distinguishes different prediction methods for the same
    gameweek (e.g. "raw_lookup" vs "refined_availability") so scoring.py can
    compare them -- see starts_model.py. Snapshots written before this
    dimension existed have no "model_version" key; treated as "raw_lookup"
    for backward compatibility.
    """
    season_dir = os.path.join(predictions_dir, season)
    if not os.path.isdir(season_dir):
        return

    by_key = {}
    for name in os.listdir(season_dir):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(season_dir, name)) as f:
            payload = json.load(f)
        key = (payload["target_round"], payload.get("model_version", "raw_lookup"))
        existing = by_key.get(key)
        if existing is None or payload["predicted_at"] > existing["predicted_at"]:
            by_key[key] = payload

    insert_sql = (
        "INSERT INTO predictions (code, season, target_round, model_version, "
        "predicted_at, p_start, cold_start, n_observed, method) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for (target_round, model_version), payload in by_key.items():
        rows = [
            (row["code"], season, target_round, model_version,
             payload["predicted_at"], row["p_start"], bool(row["cold_start"]),
             row["n_observed"], row["method"])
            for row in payload["predictions"]
        ]
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


def build_season(conn, base_dir, season, predictions_dir=PREDICTIONS_DIR):
    _load_teams(conn, base_dir, season)
    id_to_code = _load_players(conn, base_dir, season)
    team_code_history = _team_code_history(base_dir, season)
    _load_gameweek_stats(conn, base_dir, season, id_to_code, team_code_history)
    _load_availability_snapshots(conn, base_dir, season)
    _load_predictions(conn, predictions_dir, season)


def rebuild(base_dir=archiver.RAW_DIR, db_path=DERIVED_DB_PATH, seasons=None,
            predictions_dir=PREDICTIONS_DIR):
    """Rebuild the derived database from scratch by replaying `base_dir`.
    Always a full rebuild, never an incremental update -- see module
    docstring. Returns the season(s) rebuilt.
    """
    if seasons is None:
        seasons = _seasons_in_archive(base_dir)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            "DROP TABLE IF EXISTS predictions;"
            "DROP TABLE IF EXISTS player_gameweek_stats;"
            "DROP TABLE IF EXISTS player_availability_snapshots;"
            "DROP TABLE IF EXISTS players;"
            "DROP TABLE IF EXISTS teams;"
        )
        conn.executescript(SCHEMA)
        for season in seasons:
            build_season(conn, base_dir, season, predictions_dir=predictions_dir)
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
    parser.add_argument("--predictions-dir", default=PREDICTIONS_DIR)
    args = parser.parse_args()

    # Run as if invoked from the repo root, regardless of where `python
    # derived.py` was actually launched from -- keeps raw/derived paths
    # consistent with every other entry point.
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(module_dir))
    seasons = rebuild(base_dir=args.base_dir, db_path=args.db_path,
                       predictions_dir=args.predictions_dir)
    print("rebuilt {0} from {1}: {2}".format(
        args.db_path, args.base_dir, ", ".join(seasons) or "(no seasons found)"
    ))


if __name__ == "__main__":
    _main()
