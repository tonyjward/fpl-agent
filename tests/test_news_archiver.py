"""Tests for news_archiver.py: the write-once raw archive for
premierleague.com's content API (pl_content.py).
"""

from datetime import datetime, timezone

import pytest

import archiver
import news_archiver
import pl_content

SEASON = "2026-27"


class SequenceClock(object):
    def __init__(self, timestamps):
        self._timestamps = list(timestamps)

    def __call__(self):
        return self._timestamps.pop(0)


def ts(n):
    return datetime(2026, 9, n, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Plausibility gates
# --------------------------------------------------------------------------


def test_check_pl_news_raises_on_empty_list():
    with pytest.raises(archiver.PlausibilityError):
        news_archiver.check_pl_news({"list": {"content": []}})


def test_check_pl_news_accepts_nonempty_list():
    news_archiver.check_pl_news({"list": {"content": [{"id": 1}]}})


def test_check_pl_injuries_raises_when_too_few_clubs_fetched():
    payload = {"clubs": {str(i): {"title": "x"} for i in range(5)}}
    with pytest.raises(archiver.PlausibilityError):
        news_archiver.check_pl_injuries(payload)


def test_check_pl_injuries_ignores_fetch_errors_in_the_success_count():
    clubs = {str(i): {"title": "x"} for i in range(15)}
    clubs["99"] = {"_fetch_error": "boom", "_club_name": "Somewhere"}
    news_archiver.check_pl_injuries({"clubs": clubs})  # does not raise


# --------------------------------------------------------------------------
# fetch_pl_news
# --------------------------------------------------------------------------


def test_fetch_pl_news_archives_list_native_and_syndicated_articles(
    tmp_path, monkeypatch,
):
    base_dir = str(tmp_path)
    list_payload = {"content": [
        {"id": 1, "title": "Native"},
        {"id": 2, "title": "Syndicated"},
    ]}
    detail_by_id = {
        1: {"id": 1, "body": "<p>" + "x" * 50 + "</p>", "hotlinkUrl": None},
        2: {"id": 2, "body": None, "hotlinkUrl": "https://club.example/a"},
    }

    external_html = "<p>" + "External club-site prose. " * 5 + "</p>"
    monkeypatch.setattr(pl_content, "get_json", lambda path: list_payload)
    monkeypatch.setattr(pl_content, "get_article", lambda aid: detail_by_id[aid])
    monkeypatch.setattr(
        pl_content, "fetch_raw_html", lambda url, **kw: (external_html, "http"),
    )

    entry = news_archiver.fetch_pl_news(
        base_dir, SEASON, gw=3, clock=SequenceClock([ts(4)]),
    )

    assert entry["outcome"] == "ok"
    payload = news_archiver._read_gz_json(entry["path"])
    assert set(payload["articles"].keys()) == {"1", "2"}
    assert "1" not in payload["external"]
    # Stored as extracted text, not the raw page HTML -- see news_archiver's
    # comment on why (docs/README.md's "fetch markdown and archive it").
    assert payload["external"]["2"]["url"] == "https://club.example/a"
    assert payload["external"]["2"]["method"] == "http"
    assert "External club-site prose" in payload["external"]["2"]["text"]
    assert "<p>" not in payload["external"]["2"]["text"]


def test_fetch_pl_news_skips_articles_already_captured(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    list_payload = {"content": [{"id": 1, "title": "Old"}, {"id": 2, "title": "New"}]}
    get_article_calls = []

    def fake_get_article(article_id):
        get_article_calls.append(article_id)
        return {"id": article_id, "body": "<p>" + "y" * 50 + "</p>", "hotlinkUrl": None}

    monkeypatch.setattr(pl_content, "get_json", lambda path: list_payload)
    monkeypatch.setattr(pl_content, "get_article", fake_get_article)

    # First run captures both articles.
    news_archiver.fetch_pl_news(base_dir, SEASON, gw=3, clock=SequenceClock([ts(3)]))
    assert sorted(get_article_calls) == [1, 2]

    # Second run against the same (unchanged) list should not re-fetch
    # either article's detail.
    get_article_calls.clear()
    news_archiver.fetch_pl_news(base_dir, SEASON, gw=3, clock=SequenceClock([ts(4)]))
    assert get_article_calls == []


def test_fetch_pl_news_raises_when_list_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pl_content, "get_json", lambda path: {"content": []})
    with pytest.raises(archiver.PlausibilityError):
        news_archiver.fetch_pl_news(str(tmp_path), SEASON, gw=3)


def test_fetch_pl_news_records_per_article_fetch_errors_without_aborting(
    tmp_path, monkeypatch,
):
    list_payload = {"content": [{"id": 1}, {"id": 2}]}

    def fake_get_article(article_id):
        if article_id == 1:
            raise ValueError("boom")
        return {"id": 2, "body": "<p>" + "z" * 50 + "</p>", "hotlinkUrl": None}

    monkeypatch.setattr(pl_content, "get_json", lambda path: list_payload)
    monkeypatch.setattr(pl_content, "get_article", fake_get_article)

    entry = news_archiver.fetch_pl_news(str(tmp_path), SEASON, gw=3)
    payload = news_archiver._read_gz_json(entry["path"])
    assert payload["articles"]["1"] == {"_fetch_error": "boom"}
    assert payload["articles"]["2"]["id"] == 2


# --------------------------------------------------------------------------
# fetch_pl_injuries
# --------------------------------------------------------------------------


def test_fetch_pl_injuries_archives_hub_and_every_club(tmp_path, monkeypatch):
    hub_payload = {
        "items": (
            [
                {"response": {"id": 100, "title": "Injury News - Arsenal"}},
                {"response": {"id": 101, "title": "Injury News - Chelsea"}},
            ]
            + [
                {"response": {"id": 200 + i, "title": "Injury News - Club {0}".format(i)}}
                for i in range(13)
            ]
        )
    }
    club_payloads = {
        100: {"items": [{"response": {"type": "promo", "title": "P1"}}]},
        101: {"items": []},
    }

    def fake_get_json(path):
        if str(pl_content.INJURY_HUB_PLAYLIST_ID) in path:
            return hub_payload
        for club_id, payload in club_payloads.items():
            if "/{0}?".format(club_id) in path:
                return payload
        return {"items": []}

    monkeypatch.setattr(pl_content, "get_json", fake_get_json)

    entry = news_archiver.fetch_pl_injuries(str(tmp_path), SEASON, gw=3)
    payload = news_archiver._read_gz_json(entry["path"])
    assert payload["hub"] == hub_payload
    assert payload["clubs"]["100"] == club_payloads[100]
    assert payload["clubs"]["101"] == club_payloads[101]
    assert len(payload["clubs"]) == 15


def test_fetch_pl_injuries_records_per_club_fetch_errors_without_aborting(
    tmp_path, monkeypatch,
):
    hub_payload = {
        "items": [
            {"response": {"id": i, "title": "Injury News - Club {0}".format(i)}}
            for i in range(20)
        ]
    }

    def fake_get_json(path):
        if str(pl_content.INJURY_HUB_PLAYLIST_ID) in path:
            return hub_payload
        if "/0?" in path:
            raise ValueError("boom")
        return {"items": []}

    monkeypatch.setattr(pl_content, "get_json", fake_get_json)

    entry = news_archiver.fetch_pl_injuries(str(tmp_path), SEASON, gw=3)
    payload = news_archiver._read_gz_json(entry["path"])
    assert payload["clubs"]["0"]["_fetch_error"] == "boom"
    assert payload["clubs"]["0"]["_club_name"] == "Club 0"
    assert payload["clubs"]["1"] == {"items": []}


def test_fetch_pl_injuries_raises_when_too_few_clubs_resolved(tmp_path, monkeypatch):
    hub_payload = {"items": [
        {"response": {"id": 1, "title": "Injury News - Only One"}}
    ]}
    monkeypatch.setattr(pl_content, "get_json", lambda path: (
        hub_payload if str(pl_content.INJURY_HUB_PLAYLIST_ID) in path else {"items": []}
    ))
    with pytest.raises(archiver.PlausibilityError):
        news_archiver.fetch_pl_injuries(str(tmp_path), SEASON, gw=3)


# --------------------------------------------------------------------------
# gw-state helper
# --------------------------------------------------------------------------


def test_latest_bootstrap_gw_state_reads_most_recent_entry(tmp_path):
    base_dir = str(tmp_path)
    for gw, next_gw, fetched_at in [(2, 3, "20260901T000000Z"), (3, 4, "20260908T000000Z")]:
        entry = {
            "outcome": "ok", "path": "x", "endpoint": "bootstrap-static",
            "season": SEASON, "fetched_at": fetched_at, "next_gw": next_gw,
            "current_gw": gw, "next_deadline": None, "http_status": 200,
            "bytes_raw": 1, "sha256": "abc",
        }
        archiver.append_manifest_entry(base_dir, SEASON, entry)

    next_gw, current_gw = news_archiver._latest_bootstrap_gw_state(base_dir, SEASON)
    assert (next_gw, current_gw) == (4, 3)


def test_latest_bootstrap_gw_state_returns_none_when_nothing_archived(tmp_path):
    assert news_archiver._latest_bootstrap_gw_state(str(tmp_path), SEASON) == (None, None)
