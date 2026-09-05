"""Tests for web_news_archiver.py: the write-once raw archive for
fixture-driven Brave web search results and each club's own official news
page.
"""

import gzip
import json
from datetime import datetime, timezone

import pytest

import archiver
import pl_content
import web_news_archiver

SEASON = "2026-27"


class SequenceClock(object):
    def __init__(self, timestamps):
        self._timestamps = list(timestamps)

    def __call__(self):
        return self._timestamps.pop(0)


def ts(n):
    return datetime(2026, 9, n, 12, 0, 0, tzinfo=timezone.utc)


def _seed_raw(base_dir, season, endpoint, gw, fetched_at, payload):
    """Write a raw snapshot + manifest entry for an upstream endpoint
    (bootstrap-static/fixtures) the way archiver.py itself would, so
    web_news_archiver's own readers (_latest_ok_payload) have something to
    find.
    """
    content = json.dumps(payload).encode("utf-8")
    path = archiver.snapshot_path(base_dir, season, endpoint, gw, fetched_at)
    import os
    os.makedirs(os.path.dirname(path))
    with gzip.open(path, "wb") as f:
        f.write(content)
    archiver.append_manifest_entry(base_dir, season, {
        "outcome": "ok", "path": path, "endpoint": endpoint, "season": season,
        "fetched_at": archiver.format_timestamp(fetched_at),
    })


BOOTSTRAP = {"teams": [
    {"id": 1, "name": "Leeds"},
    {"id": 2, "name": "Arsenal"},
    {"id": 3, "name": "Spurs"},
]}
FIXTURES = [
    {"event": 3, "team_h": 1, "team_a": 2},
    {"event": 3, "team_h": 3, "team_a": 2},
    {"event": 2, "team_h": 2, "team_a": 1},
]


def _seed_bootstrap_and_fixtures(base_dir):
    _seed_raw(base_dir, SEASON, "bootstrap-static", 3, ts(1), BOOTSTRAP)
    _seed_raw(base_dir, SEASON, "fixtures", 3, ts(1), FIXTURES)


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------


def test_build_queries_orders_home_first_and_covers_both_pressers():
    queries = dict(web_news_archiver.build_queries("Leeds", "Arsenal"))
    assert queries["predicted_lineup"] == "Leeds Arsenal predicted lineup"
    assert queries["team_news"] == "Leeds Arsenal team news"
    assert queries["home_press_conference"] == "Leeds press conference"
    assert queries["away_press_conference"] == "Arsenal press conference"


def test_build_queries_applies_search_alias():
    queries = dict(web_news_archiver.build_queries("Spurs", "Nott'm Forest"))
    assert queries["predicted_lineup"] == "Tottenham Nottingham Forest predicted lineup"
    assert queries["home_press_conference"] == "Tottenham press conference"


def test_fixtures_for_round_filters_by_event():
    pairs = web_news_archiver.fixtures_for_round(FIXTURES, 3)
    assert pairs == [(1, 2), (3, 2)]


def test_fixtures_for_round_excludes_finished_fixtures_by_default():
    fixtures = [
        {"id": 1, "event": 3, "team_h": 1, "team_a": 2, "finished": True},
        {"id": 2, "event": 3, "team_h": 3, "team_a": 2, "finished": False},
    ]
    assert web_news_archiver.fixtures_for_round(fixtures, 3) == [(3, 2)]


def test_fixtures_for_round_fixture_ids_overrides_the_finished_filter():
    fixtures = [
        {"id": 1, "event": 3, "team_h": 1, "team_a": 2, "finished": True},
        {"id": 2, "event": 3, "team_h": 3, "team_a": 2, "finished": False},
    ]
    pairs = web_news_archiver.fixtures_for_round(fixtures, 3, fixture_ids=[1])
    assert pairs == [(1, 2)]


# --------------------------------------------------------------------------
# Plausibility gates
# --------------------------------------------------------------------------


def test_check_web_news_raises_when_every_query_returns_nothing():
    payload = {"fixtures": {
        "Leeds-Arsenal": {"queries": {
            "predicted_lineup": {"query": "x", "search_result": {"web": {"results": []}}},
        }},
    }}
    with pytest.raises(archiver.PlausibilityError):
        web_news_archiver.check_web_news(payload)


def test_check_web_news_accepts_at_least_one_result():
    payload = {"fixtures": {
        "Leeds-Arsenal": {"queries": {
            "predicted_lineup": {
                "query": "x",
                "search_result": {"web": {"results": [{"url": "https://a.example"}]}},
            },
        }},
    }}
    web_news_archiver.check_web_news(payload)  # does not raise


# --------------------------------------------------------------------------
# is_club_domain
# --------------------------------------------------------------------------


def test_is_club_domain_true_for_a_known_club_site():
    assert web_news_archiver.is_club_domain("https://www.arsenal.com/news/some-article")
    assert web_news_archiver.is_club_domain("https://arsenal.com/news/some-article")


def test_is_club_domain_false_for_a_third_party_site():
    assert not web_news_archiver.is_club_domain("https://www.sportsmole.co.uk/football/x")


# --------------------------------------------------------------------------
# fetch_web_news
# --------------------------------------------------------------------------


def _fake_brave_search(results_by_query):
    def brave_search(query, api_key, count=6):
        urls = results_by_query.get(query, [])
        return {"web": {"results": [{"url": u, "title": "t"} for u in urls]}}
    return brave_search


def test_fetch_web_news_raises_without_upstream_fixtures(tmp_path):
    with pytest.raises(archiver.ArchiveError):
        web_news_archiver.fetch_web_news(
            str(tmp_path), SEASON, gw=3, target_round=3, brave_api_key="key",
        )


def test_fetch_web_news_archives_queries_and_fetches_result_pages(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    _seed_bootstrap_and_fixtures(base_dir)

    results_by_query = {
        "Leeds Arsenal predicted lineup": ["https://sportsmole.example/a"],
        "Leeds Arsenal team news": ["https://skysports.example/b"],
        "Leeds press conference": ["https://leedsunited.example/c"],
        "Arsenal press conference": ["https://arsenal.example/d"],
        "Tottenham Arsenal predicted lineup": [],
        "Tottenham Arsenal team news": [],
        "Tottenham press conference": [],
    }
    monkeypatch.setattr(
        pl_content, "fetch_raw_html",
        lambda url, **kw: ("<p>" + "prose about the match " * 5 + "</p>", "http"),
    )

    entry = web_news_archiver.fetch_web_news(
        base_dir, SEASON, gw=3, target_round=3, brave_api_key="key",
        clock=SequenceClock([ts(2)]),
        brave_search=_fake_brave_search(results_by_query),
    )

    payload = web_news_archiver._read_gz_json(entry["path"])
    # Fixture keys use bootstrap-static's own team names (unaliased) --
    # aliasing (Spurs -> Tottenham) only applies to the search query string.
    assert set(payload["fixtures"].keys()) == {"Leeds-Arsenal", "Spurs-Arsenal"}

    leeds_fixture = payload["fixtures"]["Leeds-Arsenal"]
    assert leeds_fixture["queries"]["predicted_lineup"]["query"] == (
        "Leeds Arsenal predicted lineup"
    )
    fetched = payload["fetched_articles"]
    assert "https://sportsmole.example/a" in fetched
    assert "prose about the match" in fetched["https://sportsmole.example/a"]["text"]
    assert "<p>" not in fetched["https://sportsmole.example/a"]["text"]
    # No allowlist/denylist applied -- every returned URL gets fetched, sportsmole included.
    assert set(fetched.keys()) == {
        "https://sportsmole.example/a", "https://skysports.example/b",
        "https://leedsunited.example/c", "https://arsenal.example/d",
    }


def test_fetch_web_news_records_per_query_search_errors_without_aborting(tmp_path, monkeypatch):
    import requests

    base_dir = str(tmp_path)
    _seed_bootstrap_and_fixtures(base_dir)
    monkeypatch.setattr(pl_content, "fetch_raw_html", lambda url, **kw: (None, "blocked"))

    def flaky_search(query, api_key, count=6):
        if "predicted lineup" in query:
            raise requests.exceptions.RequestException("boom")
        if query == "Arsenal Leeds team news":
            return {"web": {"results": [{"url": "https://skysports.example/x"}]}}
        return {"web": {"results": []}}

    entry = web_news_archiver.fetch_web_news(
        base_dir, SEASON, gw=3, target_round=2, brave_api_key="key",
        brave_search=flaky_search,
    )
    payload = web_news_archiver._read_gz_json(entry["path"])
    fixture = payload["fixtures"]["Arsenal-Leeds"]
    assert fixture["queries"]["predicted_lineup"]["_fetch_error"] == "boom"


def test_fetch_web_news_does_not_refetch_a_url_seen_in_a_prior_run(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    _seed_bootstrap_and_fixtures(base_dir)

    fetch_calls = []

    def fake_fetch_raw_html(url, **kw):
        fetch_calls.append(url)
        return "<p>" + "some prose here indeed " * 5 + "</p>", "http"

    monkeypatch.setattr(pl_content, "fetch_raw_html", fake_fetch_raw_html)
    results_by_query = {
        "Arsenal Leeds predicted lineup": ["https://x.example/same"],
    }
    brave_search = _fake_brave_search(results_by_query)

    web_news_archiver.fetch_web_news(
        base_dir, SEASON, gw=2, target_round=2, brave_api_key="key",
        clock=SequenceClock([ts(2)]), brave_search=brave_search,
    )
    assert fetch_calls == ["https://x.example/same"]

    fetch_calls.clear()
    web_news_archiver.fetch_web_news(
        base_dir, SEASON, gw=2, target_round=2, brave_api_key="key",
        clock=SequenceClock([ts(3)]), brave_search=brave_search,
    )
    assert fetch_calls == []
