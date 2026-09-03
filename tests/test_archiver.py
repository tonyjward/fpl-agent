"""Tests for archiver, written against the 20-item list in
docs/build_spec_minutes_model.md-derived plan (see the raw-archiver plan
in the project history). Each test function's docstring names which item
it covers.
"""

import gzip
import json
import os
import re
from datetime import datetime, timezone

import pytest

import archiver

SEASON = "2026-27"


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


def make_bootstrap_payload(n_elements=350, n_teams=20, next_gw=3, current_gw=2,
                            next_deadline="2026-09-05T17:30:00Z", flagged=True):
    elements = []
    for i in range(n_elements):
        if flagged and i == 0:
            chance = 50
            news = "Ankle knock, assessed after training"
        else:
            chance = None
            news = ""
        elements.append({
            "id": i + 1,
            "chance_of_playing_next_round": chance,
            "news": news,
        })
    teams = [{"id": i + 1, "name": "Team {0}".format(i + 1)} for i in range(n_teams)]
    events = [
        {"id": current_gw, "is_next": False, "is_current": True,
         "deadline_time": "2026-08-29T17:30:00Z", "data_checked": True},
        {"id": next_gw, "is_next": True, "is_current": False,
         "deadline_time": next_deadline, "data_checked": False},
    ]
    return {"elements": elements, "teams": teams, "events": events, "chips": []}


class SequenceClock(object):
    """A fake clock that returns each timestamp in order, once."""

    def __init__(self, timestamps):
        self._timestamps = list(timestamps)

    def __call__(self):
        return self._timestamps.pop(0)


def make_fake_http_get(responses):
    """responses: {path: (status, content_bytes)}. Records every path
    requested on `.calls` so tests can assert what was and wasn't fetched.
    """
    calls = []

    def http_get(path):
        calls.append(path)
        return responses[path]

    http_get.calls = calls
    return http_get


def read_manifest_lines(base_dir, season):
    path = archiver.manifest_path(str(base_dir), season)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_manifest_raw_lines(base_dir, season):
    path = archiver.manifest_path(str(base_dir), season)
    with open(path) as f:
        return f.readlines()


def all_raw_files(base_dir):
    found = []
    for root, _dirs, files in os.walk(str(base_dir)):
        for name in files:
            if name.endswith(".json.gz"):
                found.append(os.path.join(root, name))
    return found


FIXED = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# 1. Round-trip fidelity
# --------------------------------------------------------------------------


def test_round_trip_fidelity(tmp_path):
    payload = make_bootstrap_payload()
    # Pretty-printed with a specific key order: if the archiver ever
    # re-serialises via json.dumps(parsed_payload) instead of storing the
    # original bytes, this formatting is lost and the test catches it.
    content = json.dumps(payload, indent=2, sort_keys=False).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    with gzip.open(entry["path"], "rb") as f:
        stored = f.read()
    assert stored == content


# --------------------------------------------------------------------------
# 2. Path naming uses is_next, not is_current
# --------------------------------------------------------------------------


def test_path_naming_uses_is_next_not_current(tmp_path):
    payload = make_bootstrap_payload(next_gw=4, current_gw=3)
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    assert os.path.join("bootstrap-static", "gw04") in entry["path"]
    assert "gw03" not in entry["path"]


# --------------------------------------------------------------------------
# 2a. Filename includes seconds; two same-minute fetches don't collide
# --------------------------------------------------------------------------


def test_filename_seconds_avoid_same_minute_collision(tmp_path):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    t1 = datetime(2026, 9, 4, 12, 0, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 45, tzinfo=timezone.utc)
    clock = SequenceClock([t1, t2])

    entry1 = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    entry2 = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    assert entry1["path"] != entry2["path"]
    assert os.path.exists(entry1["path"])
    assert os.path.exists(entry2["path"])


# --------------------------------------------------------------------------
# 3. Manifest line completeness
# --------------------------------------------------------------------------


def test_manifest_line_completeness(tmp_path):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    lines = read_manifest_lines(tmp_path, SEASON)
    assert len(lines) == 1
    required_fields = [
        "path", "endpoint", "fetched_at", "next_gw", "current_gw",
        "next_deadline", "http_status", "bytes_raw", "sha256",
    ]
    for field in required_fields:
        assert field in lines[0], field
    assert lines[0]["path"] is not None
    assert lines[0]["next_gw"] == 3
    assert lines[0]["current_gw"] == 2


# --------------------------------------------------------------------------
# 4. Manifest is append-only
# --------------------------------------------------------------------------


def test_manifest_append_only(tmp_path):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    t1 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 5, tzinfo=timezone.utc)
    clock = SequenceClock([t1, t2])

    archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    raw_after_first = read_manifest_raw_lines(tmp_path, SEASON)

    archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    raw_after_second = read_manifest_raw_lines(tmp_path, SEASON)

    assert raw_after_second[0] == raw_after_first[0]
    assert len(raw_after_second) == len(raw_after_first) + 1


# --------------------------------------------------------------------------
# 5. sha256 correctness
# --------------------------------------------------------------------------


def test_sha256_correctness(tmp_path):
    import hashlib

    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    assert entry["sha256"] == hashlib.sha256(content).hexdigest()


# --------------------------------------------------------------------------
# 6. Identical consecutive payloads are still written, just flagged
# --------------------------------------------------------------------------


def test_duplicate_consecutive_payloads_both_written_and_flagged(tmp_path):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    t1 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 1, tzinfo=timezone.utc)
    clock = SequenceClock([t1, t2])

    entry1 = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    entry2 = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    assert entry1["path"] != entry2["path"]
    with gzip.open(entry1["path"], "rb") as f:
        content1 = f.read()
    with gzip.open(entry2["path"], "rb") as f:
        content2 = f.read()
    assert content1 == content2 == content

    assert "duplicate_of" not in entry1
    assert entry2.get("duplicate_of") == entry1["path"]


# --------------------------------------------------------------------------
# 7. Write-once / no overwrite on a true same-second collision
# --------------------------------------------------------------------------


def test_write_once_refuses_same_second_collision(tmp_path):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED, FIXED])

    entry1 = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    with gzip.open(entry1["path"], "rb") as f:
        original_bytes = f.read()

    with pytest.raises(archiver.ArchiveError):
        archiver.archive_snapshot(
            "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
        )

    with gzip.open(entry1["path"], "rb") as f:
        after_bytes = f.read()
    assert after_bytes == original_bytes


# --------------------------------------------------------------------------
# 8. Endpoint scope: bootstrap-static and fixtures only, never element-summary
# --------------------------------------------------------------------------


def test_endpoint_scope_bootstrap_and_fixtures_only(tmp_path):
    payload = make_bootstrap_payload()
    bootstrap_content = json.dumps(payload).encode("utf-8")
    fixtures_content = json.dumps([{"id": 1, "event": 3}]).encode("utf-8")
    http_get = make_fake_http_get({
        "bootstrap-static/": (200, bootstrap_content),
        "fixtures/": (200, fixtures_content),
    })
    t1 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 1, tzinfo=timezone.utc)
    clock = SequenceClock([t1, t2])

    bootstrap_entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    gw_state = (
        bootstrap_entry["next_gw"],
        bootstrap_entry["current_gw"],
        bootstrap_entry["next_deadline"],
    )
    archiver.archive_snapshot(
        "fixtures", http_get, SEASON, base_dir=str(tmp_path), gw_state=gw_state,
        clock=clock,
    )

    assert http_get.calls == ["bootstrap-static/", "fixtures/"]
    assert not any("element-summary" in call for call in http_get.calls)


# --------------------------------------------------------------------------
# 9. event/{gw}/live gated on data_checked
# --------------------------------------------------------------------------


def test_event_live_gated_on_data_checked(tmp_path):
    live_content = json.dumps({"elements": []}).encode("utf-8")
    http_get = make_fake_http_get({"event/2/live/": (200, live_content)})

    not_checked = [{"id": 2, "data_checked": False}]
    clock1 = SequenceClock([FIXED])
    result = archiver.archive_event_live(
        2, not_checked, http_get, SEASON, base_dir=str(tmp_path), clock=clock1
    )
    assert result is None
    assert http_get.calls == []

    checked = [{"id": 2, "data_checked": True}]
    clock2 = SequenceClock([FIXED])
    result = archiver.archive_event_live(
        2, checked, http_get, SEASON, base_dir=str(tmp_path), clock=clock2
    )
    assert result is not None
    assert http_get.calls == ["event/2/live/"]


# --------------------------------------------------------------------------
# 10. Fetch failure is surfaced, not silent
# --------------------------------------------------------------------------


def test_fetch_failure_recorded_and_raised(tmp_path):
    http_get = make_fake_http_get({"bootstrap-static/": (503, b"<html>error</html>")})
    clock = SequenceClock([FIXED])

    with pytest.raises(archiver.ArchiveError):
        archiver.archive_snapshot(
            "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
        )

    lines = read_manifest_lines(tmp_path, SEASON)
    assert len(lines) == 1
    assert lines[0]["outcome"] == "failed"
    assert lines[0]["http_status"] == 503
    assert lines[0]["path"] is None
    assert lines[0]["sha256"] is None
    assert all_raw_files(tmp_path) == []


# --------------------------------------------------------------------------
# 11. Schema-change resilience
# --------------------------------------------------------------------------


def test_schema_change_resilience(tmp_path):
    payload = make_bootstrap_payload()
    payload["a_field_fpl_adds_next_season"] = {"nested": [1, 2, 3]}
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    with gzip.open(entry["path"], "rb") as f:
        stored = f.read()
    stored_payload = json.loads(stored.decode("utf-8"))
    assert stored_payload["a_field_fpl_adds_next_season"] == {"nested": [1, 2, 3]}


# --------------------------------------------------------------------------
# 12. Multi-endpoint manifest correctness
# --------------------------------------------------------------------------


def test_multi_endpoint_manifest_correctness(tmp_path):
    payload = make_bootstrap_payload()
    bootstrap_content = json.dumps(payload).encode("utf-8")
    fixtures_content = json.dumps([{"id": 1, "event": 3}]).encode("utf-8")
    http_get = make_fake_http_get({
        "bootstrap-static/": (200, bootstrap_content),
        "fixtures/": (200, fixtures_content),
    })
    t1 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 1, tzinfo=timezone.utc)
    clock = SequenceClock([t1, t2])

    bootstrap_entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    gw_state = (
        bootstrap_entry["next_gw"],
        bootstrap_entry["current_gw"],
        bootstrap_entry["next_deadline"],
    )
    archiver.archive_snapshot(
        "fixtures", http_get, SEASON, base_dir=str(tmp_path), gw_state=gw_state,
        clock=clock,
    )

    lines = read_manifest_lines(tmp_path, SEASON)
    assert len(lines) == 2
    endpoints = set(line["endpoint"] for line in lines)
    assert endpoints == {"bootstrap-static", "fixtures"}
    paths = set(line["path"] for line in lines)
    assert len(paths) == 2


# --------------------------------------------------------------------------
# 13. Crash between file write and manifest append leaves an orphan,
#     never a dangling reference
# --------------------------------------------------------------------------


def test_crash_between_write_and_manifest_leaves_orphan(tmp_path, monkeypatch):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash after the raw file was written")

    monkeypatch.setattr(archiver, "append_manifest_entry", boom)

    with pytest.raises(RuntimeError):
        archiver.archive_snapshot(
            "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
        )

    expected_path = archiver.snapshot_path(
        str(tmp_path), SEASON, "bootstrap-static", 3, FIXED
    )
    assert os.path.exists(expected_path)
    # No manifest line was written -- the orphan is exactly what verify_archive
    # (which never calls append_manifest_entry) must detect.
    assert not os.path.exists(archiver.manifest_path(str(tmp_path), SEASON))

    with pytest.raises(archiver.ArchiveError) as excinfo:
        archiver.verify_archive(str(tmp_path), SEASON)
    assert "orphan" in str(excinfo.value)


# --------------------------------------------------------------------------
# 14. Timestamp format
# --------------------------------------------------------------------------


def test_timestamp_format(tmp_path):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    assert re.search(r"\d{8}T\d{6}Z\.json\.gz$", entry["path"])
    assert re.match(r"^\d{8}T\d{6}Z$", entry["fetched_at"])
    assert entry["fetched_at"] == "20260904T120000Z"


# --------------------------------------------------------------------------
# 15. fetched_at reflects response arrival, not request start
# --------------------------------------------------------------------------


def test_fetched_at_recorded_after_response_arrives(tmp_path):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    order = []

    def http_get(path):
        order.append("http_get")
        return 200, content

    def clock():
        order.append("clock")
        return FIXED

    archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )

    assert order.index("http_get") < order.index("clock")


# --------------------------------------------------------------------------
# 16. Plausibility gate rejects a truncated/malformed payload
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mutate,label", [
    (lambda p: p.__setitem__("elements", p["elements"][:12]), "too_few_elements"),
    (lambda p: p.__setitem__("teams", p["teams"][:4]), "wrong_team_count"),
    (lambda p: p.__setitem__("events", []), "no_events"),
    (lambda p: p.__setitem__("events", p["events"] + [dict(p["events"][-1])]),
     "duplicate_is_next"),
])
def test_plausibility_gate_rejects_truncated_payload(tmp_path, mutate, label):
    payload = make_bootstrap_payload()
    mutate(payload)
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    with pytest.raises(archiver.ArchiveError):
        archiver.archive_snapshot(
            "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
        )

    assert all_raw_files(tmp_path) == [], label
    lines = read_manifest_lines(tmp_path, SEASON)
    assert len(lines) == 1
    assert lines[0]["outcome"] == "failed"


# --------------------------------------------------------------------------
# 17. Plausibility gate rejects an availability-field blackout
# --------------------------------------------------------------------------


def test_plausibility_gate_rejects_availability_blackout(tmp_path):
    payload = make_bootstrap_payload(flagged=False)
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    with pytest.raises(archiver.PlausibilityError):
        archiver.archive_snapshot(
            "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
        )
    assert all_raw_files(tmp_path) == []


# --------------------------------------------------------------------------
# 18. Plausibility gate accepts a normal payload
# --------------------------------------------------------------------------


def test_plausibility_gate_accepts_normal_payload(tmp_path):
    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    assert entry["outcome"] == "ok"
    assert os.path.exists(entry["path"])


# --------------------------------------------------------------------------
# 19. Manifest round-trip reconstructs the archive
# --------------------------------------------------------------------------


def test_manifest_round_trip_reconstructs_archive(tmp_path):
    payload = make_bootstrap_payload()
    bootstrap_content = json.dumps(payload).encode("utf-8")
    fixtures_content = json.dumps([{"id": 1, "event": 3}]).encode("utf-8")
    http_get = make_fake_http_get({
        "bootstrap-static/": (200, bootstrap_content),
        "fixtures/": (200, fixtures_content),
    })
    t1 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 5, tzinfo=timezone.utc)  # dedup-flagged repeat
    t3 = datetime(2026, 9, 4, 15, 0, 0, tzinfo=timezone.utc)
    clock = SequenceClock([t1, t2, t3])

    e1 = archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=str(tmp_path), clock=clock
    )
    gw_state = (e1["next_gw"], e1["current_gw"], e1["next_deadline"])
    archiver.archive_snapshot(
        "fixtures", http_get, SEASON, base_dir=str(tmp_path), gw_state=gw_state,
        clock=clock,
    )

    failing_http_get = make_fake_http_get({"bootstrap-static/": (503, b"error")})
    clock4 = SequenceClock([datetime(2026, 9, 4, 16, 0, 0, tzinfo=timezone.utc)])
    with pytest.raises(archiver.ArchiveError):
        archiver.archive_snapshot(
            "bootstrap-static", failing_http_get, SEASON, base_dir=str(tmp_path),
            clock=clock4,
        )

    lines = read_manifest_lines(tmp_path, SEASON)
    assert len(lines) == 4
    assert sum(1 for line in lines if line["outcome"] == "failed") == 1

    # Should not raise: every ok line resolves to a real, hash-matching file,
    # and every file on disk has a corresponding ok line.
    archiver.verify_archive(str(tmp_path), SEASON)


# --------------------------------------------------------------------------
# Season auto-detection from the payload (not the wall clock)
# --------------------------------------------------------------------------


def test_season_from_bootstrap_derives_from_earliest_deadline():
    payload = make_bootstrap_payload(next_deadline="2026-09-05T17:30:00Z")
    # First event (current_gw) carries the season's opening deadline.
    payload["events"][0]["deadline_time"] = "2026-08-14T17:30:00Z"
    assert archiver.season_from_bootstrap(payload) == "2026-27"


def test_season_from_bootstrap_ignores_wall_clock(tmp_path):
    """The season is read off the payload's own deadlines, not off when the
    archiver happens to run -- so a snapshot fetched deep into next season's
    calendar for whatever reason (backfill, a delayed run) still files under
    the season the data itself describes, not the season implied by "now".
    """
    payload = make_bootstrap_payload()
    payload["events"][0]["deadline_time"] = "2026-08-14T17:30:00Z"
    payload["events"][1]["deadline_time"] = "2026-08-21T17:30:00Z"
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    # Clock is far outside the season the payload describes.
    far_future_clock = SequenceClock([datetime(2027, 6, 1, tzinfo=timezone.utc)])

    entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, base_dir=str(tmp_path), clock=far_future_clock
    )

    assert entry["season"] == "2026-27"


def test_season_from_bootstrap_rejects_payload_with_no_deadlines():
    payload = make_bootstrap_payload()
    for event in payload["events"]:
        event["deadline_time"] = None
    with pytest.raises(archiver.PlausibilityError):
        archiver.season_from_bootstrap(payload)


def test_archive_snapshot_auto_detects_season_when_omitted(tmp_path):
    payload = make_bootstrap_payload()
    payload["events"][0]["deadline_time"] = "2026-08-14T17:30:00Z"
    payload["events"][1]["deadline_time"] = "2026-09-05T17:30:00Z"
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    clock = SequenceClock([FIXED])

    entry = archiver.archive_snapshot(
        "bootstrap-static", http_get, base_dir=str(tmp_path), clock=clock
    )

    assert entry["season"] == "2026-27"
    assert os.path.join(str(tmp_path), "2026-27") in entry["path"]


def test_non_bootstrap_endpoint_requires_explicit_season(tmp_path):
    http_get = make_fake_http_get({"fixtures/": (200, b"[]")})
    clock = SequenceClock([FIXED])

    with pytest.raises(ValueError):
        archiver.archive_snapshot(
            "fixtures", http_get, base_dir=str(tmp_path), gw_state=(3, 2, "x"),
            clock=clock,
        )


class FakeResponse(object):
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content


class FakeSession(object):
    """Stands in for requests.Session for run_daily_archive tests -- avoids
    a real network call while exercising the same `session.get(url,
    timeout=...)` seam make_http_get relies on.
    """

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        for path, response in self.responses.items():
            if url.endswith(path):
                return FakeResponse(*response)
        raise AssertionError("unexpected url: {0}".format(url))


def test_run_daily_archive_detects_season_once_and_reuses_it(tmp_path):
    payload = make_bootstrap_payload()
    payload["events"][0]["deadline_time"] = "2026-08-14T17:30:00Z"
    payload["events"][1]["deadline_time"] = "2026-09-05T17:30:00Z"
    bootstrap_content = json.dumps(payload).encode("utf-8")
    fixtures_content = json.dumps([{"id": 1, "event": 3}]).encode("utf-8")
    session = FakeSession({
        "bootstrap-static/": (200, bootstrap_content),
        "fixtures/": (200, fixtures_content),
    })
    clock = SequenceClock([FIXED, FIXED])

    entries = archiver.run_daily_archive(session, base_dir=str(tmp_path), clock=clock)

    # Exactly one fetch per endpoint -- no second bootstrap-static call just
    # to peek at the season before the real one.
    assert session.calls == [
        "{0}/bootstrap-static/".format(archiver.api.BASE_URL),
        "{0}/fixtures/".format(archiver.api.BASE_URL),
    ]
    assert entries[0]["season"] == "2026-27"
    assert entries[1]["season"] == "2026-27"
    assert "2026-27" in entries[1]["path"]


def test_bootstrap_fetch_failure_with_no_season_raises_without_filing(tmp_path):
    """If the very first fetch of the day fails outright, there is no
    payload to derive a season from and therefore no per-season manifest to
    file the failure under. This must still raise -- just without a
    manifest line, since there's nowhere correct to put one.
    """
    http_get = make_fake_http_get({"bootstrap-static/": (503, b"error")})
    clock = SequenceClock([FIXED])

    with pytest.raises(archiver.ArchiveError):
        archiver.archive_snapshot(
            "bootstrap-static", http_get, base_dir=str(tmp_path), clock=clock
        )

    assert not os.path.isdir(str(tmp_path)) or all_raw_files(tmp_path) == []


# --------------------------------------------------------------------------
# Regression: mixed absolute/relative base_dir must not confuse verify_archive
# --------------------------------------------------------------------------


def test_verify_archive_reconciles_absolute_and_relative_paths(tmp_path, monkeypatch):
    """A prior version of the CLI resolved base_dir to an absolute path so
    it would work regardless of invocation directory, which left the
    manifest with an absolute `path` while every other entry used a
    base_dir-relative one. verify_archive flagged the absolute-path entry as
    an orphan even though the file was exactly where it said. Mixing the two
    representations for files that genuinely exist must never trip the
    check -- only a real missing file or hash mismatch should.
    """
    monkeypatch.chdir(str(tmp_path))
    relative_base_dir = "archive"
    absolute_base_dir = os.path.abspath(relative_base_dir)

    payload = make_bootstrap_payload()
    content = json.dumps(payload).encode("utf-8")
    http_get = make_fake_http_get({"bootstrap-static/": (200, content)})
    t1 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 5, tzinfo=timezone.utc)
    clock = SequenceClock([t1, t2])

    archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=relative_base_dir, clock=clock
    )
    archiver.archive_snapshot(
        "bootstrap-static", http_get, SEASON, base_dir=absolute_base_dir, clock=clock
    )

    lines = read_manifest_lines(relative_base_dir, SEASON)
    assert not os.path.isabs(lines[0]["path"])
    assert os.path.isabs(lines[1]["path"])

    archiver.verify_archive(relative_base_dir, SEASON)
    archiver.verify_archive(absolute_base_dir, SEASON)


# --------------------------------------------------------------------------
# Direct script execution: `python archiver.py` from within src/
# --------------------------------------------------------------------------


def test_direct_script_execution_resolves_sibling_import():
    """Regression test for running `python archiver.py` directly from src/,
    where it must still find its sibling `api` module on sys.path. --help
    exits before any network call, so this only exercises import resolution
    and argument parsing.
    """
    import subprocess
    import sys

    script_dir = os.path.dirname(os.path.abspath(archiver.__file__))
    result = subprocess.run(
        [sys.executable, "archiver.py", "--help"],
        cwd=script_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stderr = result.stderr.decode("utf-8")
    assert "ImportError" not in stderr
    assert "attempted relative import" not in stderr
    assert result.returncode == 0
    assert b"usage" in result.stdout.lower()
