"""Raw archiver for the FPL API.

Writes every fetched payload byte-for-byte to a write-once, gzip-compressed
file and appends one line to a per-season manifest. See
docs/build_spec_minutes_model.md Section 2.3a and docs/README.md, "Raw
archiver", for the full spec this implements.

Two invariants everything downstream depends on:
  - the raw layer is never parsed, edited, or overwritten -- only appended to.
  - the manifest is the index of the archive: every "ok" line must point at
    a real file with a matching hash, and every raw file must have a line.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone

import api

RAW_DIR = "raw"
MANIFEST_FILENAME = "manifest.jsonl"

ENDPOINT_PATHS = {
    "bootstrap-static": "bootstrap-static/",
    "fixtures": "fixtures/",
}

EVENT_LIVE_PATH_TEMPLATE = "event/{0}/live/"


class ArchiveError(Exception):
    """A fetch or write failed. The failure has already been recorded in the
    manifest as an "outcome": "failed" line before this is raised.
    """


class PlausibilityError(ArchiveError):
    """A payload was structurally implausible and was not archived.

    Distinct from a parsing error: this checks the payload is *usable*
    (right shape, right cardinality, the fields this component exists to
    capture are actually present), never what the fields mean. Checking
    plausibility and extracting fields are different operations -- this
    does not violate "store bytes verbatim, never parse".
    """


def utcnow():
    return datetime.now(timezone.utc)


def format_timestamp(dt):
    return dt.strftime("%Y%m%dT%H%M%SZ")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Plausibility gate
# --------------------------------------------------------------------------


def check_bootstrap_static(payload):
    """Raise PlausibilityError if a bootstrap-static payload looks truncated,
    corrupted, or is missing the fields this whole component exists to
    capture.

    The last check -- at least one element carrying availability signal --
    is the point of the component: a truncated response or a CDN error page
    would otherwise archive happily and pass every structural check, and if
    the availability fields silently disappeared from the API this would
    keep producing valid-looking, worthless files.
    """
    elements = payload.get("elements") or []
    teams = payload.get("teams") or []
    events = payload.get("events") or []

    if len(elements) <= 300:
        raise PlausibilityError(
            "bootstrap-static: {0} elements, expected > 300".format(len(elements))
        )
    if len(teams) != 20:
        raise PlausibilityError(
            "bootstrap-static: {0} teams, expected 20".format(len(teams))
        )

    next_events = [e for e in events if e.get("is_next")]
    if not events or len(next_events) != 1:
        raise PlausibilityError(
            "bootstrap-static: expected exactly one is_next event among "
            "{0}, found {1}".format(len(events), len(next_events))
        )

    has_availability_signal = any(
        e.get("chance_of_playing_next_round") is not None or e.get("news")
        for e in elements
    )
    if not has_availability_signal:
        raise PlausibilityError(
            "bootstrap-static: no element carries a non-null "
            "chance_of_playing_next_round or non-empty news -- availability "
            "fields may have disappeared from the API"
        )


def season_from_bootstrap(payload):
    """Infer the "YYYY-YY" season label from the payload itself.

    bootstrap-static carries no season field, but events[].deadline_time
    does. Derive from the earliest deadline, never from the wall clock: the
    season a snapshot belongs to is a property of the data, not of when the
    archiver happened to run.
    """
    deadlines = sorted(
        e["deadline_time"] for e in payload.get("events") or []
        if e.get("deadline_time")
    )
    if not deadlines:
        raise PlausibilityError("no deadline_time in events; cannot infer season")
    first = datetime.strptime(deadlines[0], "%Y-%m-%dT%H:%M:%SZ")
    start = first.year if first.month >= 7 else first.year - 1
    return "{0}-{1:02d}".format(start, (start + 1) % 100)


def gw_state_from_bootstrap(payload):
    """Return (next_gw, current_gw, next_deadline) read from a
    bootstrap-static payload's `events` list.
    """
    next_gw = None
    current_gw = None
    next_deadline = None
    for event in payload.get("events") or []:
        if event.get("is_next"):
            next_gw = event.get("id")
            next_deadline = event.get("deadline_time")
        if event.get("is_current"):
            current_gw = event.get("id")
    return next_gw, current_gw, next_deadline


def is_data_checked(events, gw):
    for event in events:
        if event.get("id") == gw:
            return bool(event.get("data_checked"))
    return False


# --------------------------------------------------------------------------
# Paths and manifest
# --------------------------------------------------------------------------


def snapshot_path(base_dir, season, endpoint, gw, timestamp):
    filename = "{0}.json.gz".format(format_timestamp(timestamp))
    return os.path.join(
        base_dir, season, endpoint, "gw{0:02d}".format(gw), filename
    )


def manifest_path(base_dir, season):
    return os.path.join(base_dir, season, MANIFEST_FILENAME)


def append_manifest_entry(base_dir, season, entry):
    """Append one JSON line to the season's manifest. A single write() of one
    line terminated by "\\n", in append mode -- an interrupted write must not
    corrupt a prior line or leave a torn partial line, which is what keeping
    this to one write() call per append protects.
    """
    path = manifest_path(base_dir, season)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    line = json.dumps(entry, sort_keys=True) + "\n"
    with open(path, "a") as f:
        f.write(line)


def _record_failure(base_dir, season, endpoint, fetched_at, http_status, reason,
                     exc_class=ArchiveError):
    """Record a failed fetch/write attempt and raise. Write ordering here
    doesn't matter the way it does for a success: there is no file to
    reference, so `path` and `sha256` are null.

    `exc_class` lets callers preserve a more specific failure type (e.g.
    PlausibilityError) for whoever calls archive_snapshot, rather than
    collapsing every failure to the same generic error.
    """
    entry = {
        "outcome": "failed",
        "endpoint": endpoint,
        "season": season,
        "fetched_at": format_timestamp(fetched_at),
        "http_status": http_status,
        "path": None,
        "sha256": None,
        "bytes_raw": None,
        "reason": reason,
    }
    append_manifest_entry(base_dir, season, entry)
    raise exc_class(reason)


def _write_snapshot_file(path, content):
    """Write `content` gzip-compressed to `path`. Refuses to overwrite an
    existing file -- write-once, no "latest wins" behaviour anywhere here.
    """
    if os.path.exists(path):
        raise ArchiveError(
            "refusing to overwrite existing raw file: {0}".format(path)
        )
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with gzip.open(path, "wb") as f:
        f.write(content)


def _last_ok_hash(base_dir, season, endpoint):
    """The (sha256, path) of the most recent successfully-written snapshot
    for this endpoint, or (None, None). Used only to flag a repeat in the
    manifest as a warning -- never to skip a write.
    """
    path = manifest_path(base_dir, season)
    if not os.path.exists(path):
        return None, None
    last_hash = None
    last_path = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("endpoint") == endpoint and entry.get("outcome") == "ok":
                last_hash = entry.get("sha256")
                last_path = entry.get("path")
    return last_hash, last_path


def _finalize(base_dir, season, endpoint, gw, next_gw, current_gw,
              next_deadline, fetched_at, http_status, content):
    """Write the raw file, then append the manifest line -- in that order.
    A crash between the two leaves an orphaned file, which verify_archive
    can detect and which is repairable by regenerating the missing line from
    the file itself. The reverse ordering would leave a manifest line
    pointing at data that was never written, which is not recoverable, so
    this ordering is never reversed.
    """
    file_path = snapshot_path(base_dir, season, endpoint, gw, fetched_at)
    try:
        _write_snapshot_file(file_path, content)
    except ArchiveError as exc:
        _record_failure(base_dir, season, endpoint, fetched_at, http_status, str(exc))

    digest = sha256_hex(content)
    entry = {
        "outcome": "ok",
        "path": file_path,
        "endpoint": endpoint,
        "season": season,
        "fetched_at": format_timestamp(fetched_at),
        "next_gw": next_gw,
        "current_gw": current_gw,
        "next_deadline": next_deadline,
        "http_status": http_status,
        "bytes_raw": len(content),
        "sha256": digest,
    }

    prior_hash, prior_path = _last_ok_hash(base_dir, season, endpoint)
    if prior_hash is not None and prior_hash == digest:
        entry["duplicate_of"] = prior_path

    append_manifest_entry(base_dir, season, entry)
    return entry


# --------------------------------------------------------------------------
# Fetch + archive
# --------------------------------------------------------------------------


def archive_snapshot(endpoint, http_get, season=None, base_dir=RAW_DIR,
                      gw_state=None, clock=utcnow):
    """Fetch one endpoint and archive the response verbatim.

    `http_get(path)` must return `(status_code, content_bytes)` for the
    given API path and is the only I/O seam -- tests inject a fake instead
    of making a real HTTP call.

    For "bootstrap-static", gw_state is derived from the payload itself and
    the `gw_state` argument is ignored. Every other endpoint must be given
    the state to file under: only bootstrap-static's response carries the
    season's `events` list, so gw{NN} for fixtures is a snapshot-time label
    supplied by the caller, not something derived from the fixtures payload.

    `season` is likewise only derivable from a bootstrap-static payload
    (see season_from_bootstrap) -- omit it there to auto-detect from the
    fetched data, never from the wall clock. Every other endpoint must be
    given the season explicitly, for the same reason it must be given
    gw_state.
    """
    if endpoint != "bootstrap-static" and season is None:
        raise ValueError(
            "{0}: season must be supplied explicitly -- it cannot be "
            "derived from this endpoint's own payload".format(endpoint)
        )

    path = ENDPOINT_PATHS[endpoint]
    status, content = http_get(path)
    fetched_at = clock()

    if status != 200:
        if season is None:
            # bootstrap-static, but nothing was fetched to derive a season
            # from -- there is no per-season manifest to file this under.
            raise ArchiveError(
                "bootstrap-static: fetch failed with status {0} before a "
                "season could be derived from the payload".format(status)
            )
        _record_failure(
            base_dir, season, endpoint, fetched_at, status,
            "fetch failed with status {0}".format(status),
        )

    if endpoint == "bootstrap-static":
        try:
            payload = json.loads(content.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            reason = "response is not valid JSON: {0}".format(exc)
            if season is None:
                raise PlausibilityError(reason)
            _record_failure(base_dir, season, endpoint, fetched_at, status, reason,
                             exc_class=PlausibilityError)

        if season is None:
            season = season_from_bootstrap(payload)

        try:
            check_bootstrap_static(payload)
        except PlausibilityError as exc:
            _record_failure(base_dir, season, endpoint, fetched_at, status, str(exc),
                             exc_class=PlausibilityError)
        next_gw, current_gw, next_deadline = gw_state_from_bootstrap(payload)
    else:
        if gw_state is None:
            raise ValueError(
                "{0}: gw_state must be supplied explicitly -- it cannot be "
                "derived from this endpoint's own payload".format(endpoint)
            )
        next_gw, current_gw, next_deadline = gw_state

    return _finalize(
        base_dir, season, endpoint, next_gw, next_gw, current_gw,
        next_deadline, fetched_at, status, content,
    )


def archive_event_live(gw, events, http_get, season, base_dir=RAW_DIR,
                        clock=utcnow):
    """Archive event/{gw}/live, gated on `data_checked` for that gameweek.

    Capturing on `finished` instead would risk freezing provisional bonus
    permanently, since nothing re-reads it later (docs/README.md Section
    8a). Returns None without fetching anything if the gameweek isn't
    verified yet.
    """
    if not is_data_checked(events, gw):
        return None

    path = EVENT_LIVE_PATH_TEMPLATE.format(gw)
    status, content = http_get(path)
    fetched_at = clock()

    if status != 200:
        _record_failure(
            base_dir, season, "event-live", fetched_at, status,
            "fetch failed with status {0}".format(status),
        )

    return _finalize(
        base_dir, season, "event-live", gw, None, gw, None,
        fetched_at, status, content,
    )


def make_http_get(session):
    """Adapt a requests.Session (see api.py) to the `(status, content)`
    seam archive_snapshot and archive_event_live expect.
    """

    def http_get(path):
        resp = session.get("{0}/{1}".format(api.BASE_URL, path), timeout=15)
        return resp.status_code, resp.content

    return http_get


def run_daily_archive(session, season=None, base_dir=RAW_DIR, clock=utcnow):
    """Archive bootstrap-static, then fixtures under the gameweek state that
    bootstrap-static reported. This is the entry point the daily Airflow
    task (and the T-3h pull) calls.

    `season` is optional: bootstrap-static's own `events[].deadline_time`
    resolves it (season_from_bootstrap), and that resolved value -- not a
    second guess -- is what fixtures gets filed under, so both endpoints
    land in the same season directory from a single bootstrap-static fetch.
    """
    http_get = make_http_get(session)
    bootstrap_entry = archive_snapshot(
        "bootstrap-static", http_get, season=season, base_dir=base_dir, clock=clock
    )
    resolved_season = bootstrap_entry["season"]
    gw_state = (
        bootstrap_entry["next_gw"],
        bootstrap_entry["current_gw"],
        bootstrap_entry["next_deadline"],
    )
    fixtures_entry = archive_snapshot(
        "fixtures", http_get, season=resolved_season, base_dir=base_dir,
        gw_state=gw_state, clock=clock,
    )
    return [bootstrap_entry, fixtures_entry]


# --------------------------------------------------------------------------
# Verification -- can the manifest rebuild the archive?
# --------------------------------------------------------------------------


def verify_archive(base_dir, season):
    """Check the manifest against the files actually on disk.

    Raises ArchiveError describing every problem found if they've diverged:
    an "ok" line pointing at a missing file, a file whose hash doesn't match
    its recorded sha256, or a raw file on disk with no "ok" line pointing at
    it (an orphan). This is what the derived layer's reproducibility
    guarantee rests on -- the archive is only as good as its index.
    """
    problems = []
    seen_paths = set()

    manifest_file = manifest_path(base_dir, season)
    if os.path.exists(manifest_file):
        with open(manifest_file) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("outcome") != "ok":
                    continue
                file_path = entry["path"]
                # abspath, not normpath: a manifest can (and, via the CLI's
                # cwd-independent base_dir, sometimes does) mix relative and
                # absolute path strings for the same file. Resolving both
                # sides to an absolute path before comparing is what keeps
                # that a non-issue instead of a false "orphan".
                seen_paths.add(os.path.abspath(file_path))
                if not os.path.exists(file_path):
                    problems.append(
                        "manifest line {0} points at missing file {1}".format(
                            lineno, file_path
                        )
                    )
                    continue
                with gzip.open(file_path, "rb") as gz:
                    content = gz.read()
                if sha256_hex(content) != entry.get("sha256"):
                    problems.append("sha256 mismatch for {0}".format(file_path))

    season_dir = os.path.join(base_dir, season)
    if os.path.isdir(season_dir):
        for root, _dirs, files in os.walk(season_dir):
            for name in files:
                if not name.endswith(".json.gz"):
                    continue
                file_path = os.path.abspath(os.path.join(root, name))
                if file_path not in seen_paths:
                    problems.append(
                        "orphaned raw file with no manifest line: {0}".format(
                            file_path
                        )
                    )

    if problems:
        raise ArchiveError("archive and manifest have diverged:\n" + "\n".join(problems))


# --------------------------------------------------------------------------
# Command-line entry point
# --------------------------------------------------------------------------


def _repo_root():
    """<repo root>, resolved from this file's own location (src/archiver.py)
    rather than the current working directory.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(module_dir)


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Archive today's bootstrap-static and fixtures verbatim."
    )
    parser.add_argument(
        "--base-dir", default=RAW_DIR,
        help="Raw archive root, relative to the repo root (default: raw)",
    )
    parser.add_argument(
        "--season", default=None,
        help="Season label, e.g. 2026-27. Auto-detected from the fetched "
             "bootstrap-static payload if omitted -- this is the normal case.",
    )
    args = parser.parse_args()

    # Run as if invoked from the repo root, regardless of where `python
    # archiver.py` was actually launched from -- e.g. from src/ itself.
    # This keeps manifest paths relative to the repo root ("raw/2026-27/...",
    # matching every entry written any other way), rather than baking in
    # whatever absolute path this particular machine happens to use.
    os.chdir(_repo_root())

    session = api.new_session()
    results = run_daily_archive(session, season=args.season, base_dir=args.base_dir)
    for result in results:
        print("{0}: {1} -> {2}".format(
            result["endpoint"], result["outcome"], result["path"]
        ))


if __name__ == "__main__":
    _main()
