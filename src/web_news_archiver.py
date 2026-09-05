"""Raw archiver for a wider-web evidence source than premierleague.com's own
feed: Brave Search results built from the target gameweek's fixtures.

Same write-once/gzip/manifest discipline as news_archiver.py, filed under one
new endpoint, "web-news". Per docs/README.md's News scraper requirements --
"fetch markdown and archive it, do not extract inline" -- everything
captured here is fetched verbatim (Brave's raw JSON result list, and the
extracted-but-unjudged text of every page fetched off the back of it), never
ranked or reconciled against players/teams. That happens downstream:
derived.py loads these articles the same way it loads pl-news, and
news_extraction.py classifies them, same taxonomy either way.

**2026-09-06 revision:** an earlier version of this module also directly
scraped a fixed list of 20 club news-index pages ("club-news" endpoint).
Dropped after a live check against Leeds's own site turned up mostly
navigation/boilerplate noise, and separately most of the 20 pages are
client-rendered SPAs that a plain fetch can't see through at all (20/20
"fetched", 8/20 extracted any text, most of *that* under 60 characters --
see git history for the full module if reviving this). A club's own words
still reach this pipeline two other ways instead: premierleague.com's own
feed already syndicates official articles it links to (news_archiver.py,
via `hotlinkUrl`), and is_club_domain() below flags a Brave result that
happens to land on a club's own site so derived.py can tag it with the same
provenance an intentional scrape would have gotten -- without spending a
fetch on 19 index pages that turn out empty for every one that's useful.

Deliberately no domain allowlist or denylist on which *results* get fetched
-- see the project's own GW2 finding (docs/README.md "A finding the model
does not capture" / Provenance): a curated list built from a handful of
post-hoc observations encodes noise as policy, and a wrongly-excluded
source is invisible in a way a wrongly-included one isn't (a junk article
gets caught by extraction and dilutes a count; a missing article is never
seen at all). Every result's domain, position and date is archived
regardless of whether its page gets fetched, so per-outlet accuracy can be
measured against scoring.py's Rotation stratum once enough gameweeks have
played out, and *that* -- not a guess made from a handful of data points --
is what should eventually decide which sources are worth weighting up or
down (see derived.py's `_source_tier`). is_club_domain() is a narrower,
separate thing: not a judgment about quality, just recognising a club's own
domain so its provenance tier reflects whose words they actually are.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import gzip
import json
import os

try:
    from urllib.parse import urlparse
except ImportError:  # pragma: no cover - Python 2, not this project's target
    from urlparse import urlparse

import requests

import archiver
import pl_content

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
RESULTS_PER_QUERY = 6

# A handful of club names that are a poor *search* term even though they're
# exactly right for matching against this project's own archive
# (derived.py's own _TEAM_NAME_ALIASES solves the mirror-image problem: an
# external source's full name -> this project's short form). "Spurs" and
# "Nott'm Forest" are FPL's own short_name/name, but outlets overwhelmingly
# write "Tottenham" and "Nottingham Forest" in headlines -- searching FPL's
# literal string under-retrieves for exactly those two clubs. Every other
# 2026-27 club's `name` field (e.g. "Man City", "Man Utd") already reads as
# the commonly-searched form, so no override is needed for them.
_SEARCH_ALIASES = {
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
}

# Every 2026-27 top-flight club's own domain, used only to *recognise* a
# result already on a club's own site (see is_club_domain) -- not to fetch
# from directly, since that's exactly the scrape this revision removed.
# Same hand-maintained-list caveat as before: expected to need occasional
# fixing as a club changes domains, and a stale entry just means that one
# club's results under-tag rather than the run failing.
CLUB_DOMAINS = frozenset([
    "arsenal.com", "avfc.co.uk", "afcb.co.uk", "brentfordfc.com",
    "brightonandhovealbion.com", "chelseafc.com", "ccfc.co.uk", "cpfc.co.uk",
    "evertonfc.com", "fulhamfc.com", "hcafc.com", "itfc.co.uk",
    "leedsunited.com", "liverpoolfc.com", "mancity.com", "manutd.com",
    "nufc.co.uk", "nottinghamforest.co.uk", "tottenhamhotspur.com", "safc.com",
])


def is_club_domain(url):
    """True if `url`'s host is one of the 20 clubs' own official domains
    (see CLUB_DOMAINS) -- a manager's own words, however this pipeline
    found them, per docs/README.md's Provenance rule.
    """
    host = (urlparse(url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host in CLUB_DOMAINS


def _write_snapshot(base_dir, season, endpoint, gw, fetched_at, payload):
    """Write one JSON payload gzip-compressed and append its manifest line.
    Duplicated from news_archiver.py's identical helper rather than
    imported -- see that module's docstring for why (private to its own
    module, kept local).
    """
    content = json.dumps(payload).encode("utf-8")
    path = archiver.snapshot_path(base_dir, season, endpoint, gw, fetched_at)
    if os.path.exists(path):
        raise archiver.ArchiveError(
            "refusing to overwrite existing raw file: {0}".format(path)
        )
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with gzip.open(path, "wb") as f:
        f.write(content)

    entry = {
        "outcome": "ok",
        "path": path,
        "endpoint": endpoint,
        "season": season,
        "fetched_at": archiver.format_timestamp(fetched_at),
        "next_gw": None,
        "current_gw": gw,
        "next_deadline": None,
        "http_status": 200,
        "bytes_raw": len(content),
        "sha256": archiver.sha256_hex(content),
    }
    archiver.append_manifest_entry(base_dir, season, entry)
    return entry


def _read_gz_json(path):
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _latest_ok_payload(base_dir, season, endpoint):
    """The most recently archived, successfully-fetched payload for
    `endpoint` in `season`'s manifest (e.g. the latest bootstrap-static or
    fixtures snapshot) -- None if nothing archived yet. This module reads
    the raw archive rather than derived.db so it has no dependency on the
    derived layer having been rebuilt first, same boundary news_archiver.py
    keeps.
    """
    path = archiver.manifest_path(base_dir, season)
    if not os.path.exists(path):
        return None
    latest = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("endpoint") != endpoint or entry.get("outcome") != "ok":
                continue
            if latest is None or entry["fetched_at"] > latest["fetched_at"]:
                latest = entry
    if latest is None:
        return None
    return _read_gz_json(latest["path"])


def fixtures_for_round(fixtures_payload, target_round, fixture_ids=None,
                        only_unfinished=True):
    """[(team_h_id, team_a_id), ...] for `target_round`'s fixtures.

    Searching "predicted lineup" for a fixture that has already kicked off
    (or finished) burns a query on a question that's no longer live -- by
    default this drops any fixture already flagged `finished`, so a
    mid-gameweek run (some Saturday fixtures played, Sunday's still ahead)
    only spends its budget on what's still genuinely upcoming. `fixture_ids`
    overrides that filter entirely with an explicit allowlist of fixture
    ids -- for a deliberate one-off run against a *specific* subset (e.g.
    "just Sunday's two games", to compare the evidence gathered against
    Monday's actual results), since `finished` is only as reliable as
    however recently fixtures.py was last archived.
    """
    fixtures = [f for f in fixtures_payload if f.get("event") == target_round]
    if fixture_ids is not None:
        fixtures = [f for f in fixtures if f.get("id") in fixture_ids]
    elif only_unfinished:
        fixtures = [f for f in fixtures if not f.get("finished")]
    return [(f["team_h"], f["team_a"]) for f in fixtures]


def _search_name(team_name):
    return _SEARCH_ALIASES.get(team_name, team_name)


def build_queries(home_name, away_name):
    """The query label/string pairs to run for one fixture. Both teams get
    an equal share of the press-conference query -- an earlier draft of
    this only searched the home team's presser, which would have
    systematically under-served every away-team player's evidence. Order
    matters for the other two: outlets write "Home vs Away", so team_h is
    always named first, matching how a search engine's own phrase-matching
    actually retrieves.
    """
    home = _search_name(home_name)
    away = _search_name(away_name)
    return [
        ("predicted_lineup", "{0} {1} predicted lineup".format(home, away)),
        ("team_news", "{0} {1} team news".format(home, away)),
        ("home_press_conference", "{0} press conference".format(home)),
        ("away_press_conference", "{0} press conference".format(away)),
    ]


def default_brave_search(query, api_key, count=RESULTS_PER_QUERY):
    """GET one query from the Brave Search API. The only network seam for
    search -- tests inject a fake instead, same pattern as
    starts_model.fetch_community_archive.
    """
    resp = requests.get(
        BRAVE_SEARCH_URL,
        params={"q": query, "count": count},
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _result_urls(search_payload):
    results = ((search_payload.get("web") or {}).get("results")) or []
    return [r.get("url") for r in results if r.get("url")]


def _previously_fetched_urls(base_dir, season):
    """URLs already fetched-and-extracted in some prior "web-news" snapshot
    this season -- a search result that ranks for days shouldn't be
    re-fetched every time this runs. Mirrors news_archiver.py's
    _previously_captured_article_ids.
    """
    seen = set()
    path = archiver.manifest_path(base_dir, season)
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("endpoint") != "web-news" or entry.get("outcome") != "ok":
                continue
            payload = _read_gz_json(entry["path"])
            for url in (payload.get("fetched_articles") or {}):
                seen.add(url)
    return seen


def check_web_news(payload):
    """Raise unless at least one fixture returned at least one search
    result -- otherwise indistinguishable from "Brave is down" rather than
    "a quiet news day", per docs/README.md's "Empty or near-empty responses
    must raise".
    """
    fixtures = payload.get("fixtures") or {}
    any_results = any(
        _result_urls(query_payload.get("search_result") or {})
        for fixture in fixtures.values()
        for query_payload in (fixture.get("queries") or {}).values()
    )
    if fixtures and not any_results:
        raise archiver.PlausibilityError(
            "web-news: every fixture query returned 0 results"
        )


def fetch_web_news(base_dir, season, gw, target_round, brave_api_key,
                    clock=archiver.utcnow, session=None,
                    results_per_query=RESULTS_PER_QUERY,
                    brave_search=default_brave_search,
                    fixture_ids=None, only_unfinished=True):
    """For every fixture in `target_round` (see fixtures_for_round for how
    `fixture_ids`/`only_unfinished` narrow that set), run build_queries()'s
    four searches, archive every result's metadata verbatim, and
    fetch-and-extract (never rank, never filter -- see the module
    docstring) up to `results_per_query` result pages per query. Raises if
    neither fixtures nor bootstrap-static has been archived yet for
    `season`, since there is nothing to build a query from.
    """
    bootstrap = _latest_ok_payload(base_dir, season, "bootstrap-static")
    fixtures_payload = _latest_ok_payload(base_dir, season, "fixtures")
    if bootstrap is None or fixtures_payload is None:
        raise archiver.ArchiveError(
            "web-news: need an archived bootstrap-static and fixtures "
            "snapshot for season {0} before this can build a query "
            "-- run archiver.py first".format(season)
        )
    id_to_name = dict((t["id"], t["name"]) for t in bootstrap.get("teams") or [])
    already_fetched = _previously_fetched_urls(base_dir, season)

    fixtures = {}
    fetched_articles = {}
    fixture_pairs = fixtures_for_round(
        fixtures_payload, target_round,
        fixture_ids=fixture_ids, only_unfinished=only_unfinished,
    )
    for team_h_id, team_a_id in fixture_pairs:
        home_name = id_to_name.get(team_h_id, str(team_h_id))
        away_name = id_to_name.get(team_a_id, str(team_a_id))
        fixture_key = "{0}-{1}".format(home_name, away_name)
        queries = {}
        for label, query in build_queries(home_name, away_name):
            try:
                search_payload = brave_search(query, brave_api_key, count=results_per_query)
            except requests.exceptions.RequestException as exc:
                queries[label] = {"query": query, "_fetch_error": str(exc)}
                continue
            queries[label] = {"query": query, "search_result": search_payload}
            for url in _result_urls(search_payload)[:results_per_query]:
                if url in fetched_articles or url in already_fetched:
                    continue
                try:
                    html, method = pl_content.fetch_raw_html(url, session=session)
                except requests.exceptions.RequestException as exc:
                    fetched_articles[url] = {"_fetch_error": str(exc)}
                    continue
                if html is None:
                    fetched_articles[url] = {"_fetch_error": "blocked", "method": method}
                    continue
                fetched_articles[url] = {
                    "method": method, "text": pl_content.html_to_text(html),
                }
        fixtures[fixture_key] = {
            "team_h": team_h_id, "team_a": team_a_id, "queries": queries,
        }

    payload = {"target_round": target_round, "fixtures": fixtures,
               "fetched_articles": fetched_articles}
    check_web_news(payload)
    fetched_at = clock()
    return _write_snapshot(base_dir, season, "web-news", gw, fetched_at, payload)


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Archive Brave web-search evidence for the next "
                    "gameweek's fixtures."
    )
    parser.add_argument("--base-dir", default=archiver.RAW_DIR)
    parser.add_argument("--season", required=True)
    parser.add_argument(
        "--gw", type=int, default=None,
        help="Target gameweek, and the raw-file label. Defaults to the most "
             "recently archived bootstrap-static's next_gw -- pass this "
             "explicitly for a gameweek already in progress, since next_gw "
             "rolls over to the *following* gameweek as soon as the current "
             "one's deadline passes, well before it's finished.",
    )
    parser.add_argument(
        "--fixture-ids", default=None,
        help="Comma-separated fixture ids (from the archived fixtures "
             "payload's `id` field) to restrict the search to -- e.g. for a "
             "gameweek already partly played, just the fixtures still "
             "ahead. Overrides the default `finished` filter entirely.",
    )
    parser.add_argument("--results-per-query", type=int, default=RESULTS_PER_QUERY)
    args = parser.parse_args()
    fixture_ids = None
    if args.fixture_ids:
        fixture_ids = [int(x) for x in args.fixture_ids.split(",")]

    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(module_dir))

    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if not brave_api_key:
        raise SystemExit(
            "BRAVE_API_KEY is not set -- required to call the Brave Search API"
        )

    gw = args.gw
    if gw is None:
        bootstrap = _latest_ok_payload(args.base_dir, args.season, "bootstrap-static")
        if bootstrap is None:
            raise SystemExit(
                "no archived bootstrap-static found for season {0} -- run "
                "archiver.py first, or pass --gw explicitly".format(args.season)
            )
        next_gw, _current_gw, _deadline = archiver.gw_state_from_bootstrap(bootstrap)
        if next_gw is None:
            raise SystemExit("bootstrap-static has no upcoming gameweek (season over?)")
        gw = next_gw

    web_entry = fetch_web_news(args.base_dir, args.season, gw, gw, brave_api_key,
                                results_per_query=args.results_per_query,
                                fixture_ids=fixture_ids)
    print("web-news: {0} -> {1}".format(web_entry["outcome"], web_entry["path"]))


if __name__ == "__main__":
    _main()
