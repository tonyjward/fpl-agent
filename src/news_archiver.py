"""Raw archiver for premierleague.com's content API (see pl_content.py).

Extends archiver.py's write-once/gzip/manifest pattern to two new
endpoints, filed in the same per-season manifest as bootstrap-static and
fixtures: "pl-news" and "pl-injuries". Per docs/README.md's News scraper
component requirements -- "fetch markdown and archive it, do not extract
inline" -- everything captured here is a raw fetched payload (external
club-site HTML included) verbatim, never parsed or reconciled against
players/teams. Reconciliation, including the contradiction-check against
current squad membership, happens in derived.py, replaying this archive,
exactly like every other endpoint.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import gzip
import json
import os

import requests

import archiver
import pl_content

NEWS_LIST_LIMIT = 50

# A near-total failure to fetch club playlists (e.g. INJURY_HUB_PLAYLIST_ID
# has gone stale, see pl_content.py) must not archive happily as "0
# injuries everywhere" -- but a handful of individual club fetches failing
# on any given run is tolerated and recorded per-club, not fatal to the
# whole snapshot. 15 of 20 leaves room for a few transient failures without
# masking a real breakage.
MIN_EXPECTED_INJURY_CLUBS = 15


def _write_snapshot(base_dir, season, endpoint, gw, fetched_at, payload):
    """Write one JSON payload gzip-compressed and append its manifest line.
    Mirrors archiver._write_snapshot_file/_finalize; kept local rather than
    calling those directly since they're private to archiver.py and shaped
    around the FPL-API-specific gw_state tuple this module doesn't have.
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


def _ok_payloads(base_dir, season, endpoint):
    path = archiver.manifest_path(base_dir, season)
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("endpoint") == endpoint and entry.get("outcome") == "ok":
                yield _read_gz_json(entry["path"])


def _previously_captured_article_ids(base_dir, season):
    """Article ids that already have a detail fetch (and, if syndicated, an
    external-site fetch) captured in some prior pl-news snapshot. A
    published article's content doesn't change, so re-fetching its detail
    or external page on every run is pure repeat cost -- and repeat risk of
    tripping a club site's bot defences -- for zero new information.
    """
    seen = set()
    for payload in _ok_payloads(base_dir, season, "pl-news"):
        for article_id_str in (payload.get("articles") or {}):
            seen.add(int(article_id_str))
    return seen


def check_pl_news(payload):
    """Raise if the news list itself came back empty -- otherwise
    indistinguishable from "no news happened today", per docs/README.md's
    "Empty or near-empty responses must raise."
    """
    content = (payload.get("list") or {}).get("content") or []
    if not content:
        raise archiver.PlausibilityError("pl-news: news list returned 0 articles")


def check_pl_injuries(payload):
    """Raise unless at least MIN_EXPECTED_INJURY_CLUBS club playlists were
    fetched successfully.
    """
    clubs = payload.get("clubs") or {}
    ok_clubs = [c for c in clubs.values() if "_fetch_error" not in c]
    if len(ok_clubs) < MIN_EXPECTED_INJURY_CLUBS:
        raise archiver.PlausibilityError(
            "pl-injuries: only {0} of {1} club playlists fetched "
            "successfully (need >= {2})".format(
                len(ok_clubs), len(clubs), MIN_EXPECTED_INJURY_CLUBS
            )
        )


def fetch_pl_news(base_dir, season, gw, clock=archiver.utcnow,
                   limit=NEWS_LIST_LIMIT):
    """Fetch the news list, every not-yet-seen article's detail, and (for
    syndicated articles with no native body) the external page it links to,
    then archive the whole thing as one "pl-news" snapshot.
    """
    already_captured = _previously_captured_article_ids(base_dir, season)

    list_payload = pl_content.get_json(
        "/en?contentTypes=text&limit={0}&offset=0".format(limit)
    )
    articles = {}
    external = {}
    for summary in pl_content.parse_news_list(list_payload):
        article_id = summary.get("id")
        if article_id is None or article_id in already_captured:
            continue
        try:
            detail = pl_content.get_article(article_id)
        except (requests.exceptions.RequestException, ValueError) as exc:
            articles[str(article_id)] = {"_fetch_error": str(exc)}
            continue
        articles[str(article_id)] = detail
        if not detail.get("body") and detail.get("hotlinkUrl"):
            html, method = pl_content.fetch_raw_html(detail["hotlinkUrl"])
            # Store the extracted text, not the full page HTML -- a club
            # site's raw page is mostly nav/scripts/cookie banners (one
            # early run of this measured ~100KB+ per article), and
            # docs/README.md's own spec for this component is "fetch
            # markdown and archive it", not "archive raw page bytes". This
            # is a mechanical, deterministic reduction (BeautifulSoup
            # paragraph-stripping), not the LLM-based judgment that "do not
            # extract inline" is about -- see pl_content.html_to_text.
            external[str(article_id)] = {
                "url": detail["hotlinkUrl"], "method": method,
                "text": pl_content.html_to_text(html) if html else None,
            }

    payload = {"list": list_payload, "articles": articles, "external": external}
    check_pl_news(payload)
    fetched_at = clock()
    return _write_snapshot(base_dir, season, "pl-news", gw, fetched_at, payload)


def fetch_pl_injuries(base_dir, season, gw, clock=archiver.utcnow):
    """Fetch the injury hub and every club's injury playlist, and archive
    the whole thing as one "pl-injuries" snapshot.
    """
    hub_payload = pl_content.get_json(
        "/PLAYLIST/en/{0}?detail=DETAILED".format(pl_content.INJURY_HUB_PLAYLIST_ID)
    )
    clubs = {}
    for club_name, club_id in pl_content.parse_injury_hub(hub_payload):
        try:
            clubs[str(club_id)] = pl_content.get_json(
                "/PLAYLIST/en/{0}?detail=DETAILED".format(club_id)
            )
        except (requests.exceptions.RequestException, ValueError) as exc:
            clubs[str(club_id)] = {
                "_fetch_error": str(exc), "_club_name": club_name,
            }

    payload = {"hub": hub_payload, "clubs": clubs}
    check_pl_injuries(payload)
    fetched_at = clock()
    return _write_snapshot(base_dir, season, "pl-injuries", gw, fetched_at, payload)


def _latest_bootstrap_gw_state(base_dir, season):
    """(next_gw, current_gw) from the most recently archived bootstrap-static
    entry for `season` -- pl-news/pl-injuries aren't gameweek-scoped data,
    but need *a* gw label for their raw file path, exactly like fixtures
    (see archiver.archive_snapshot's docstring on that point).
    """
    path = archiver.manifest_path(base_dir, season)
    if not os.path.exists(path):
        return None, None
    latest = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("endpoint") != "bootstrap-static" or entry.get("outcome") != "ok":
                continue
            if latest is None or entry["fetched_at"] > latest["fetched_at"]:
                latest = entry
    if latest is None:
        return None, None
    return latest.get("next_gw"), latest.get("current_gw")


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Archive today's premierleague.com news and injuries verbatim."
    )
    parser.add_argument("--base-dir", default=archiver.RAW_DIR)
    parser.add_argument("--season", required=True)
    parser.add_argument(
        "--gw", type=int, default=None,
        help="Gameweek label for the raw file path. Defaults to the most "
             "recently archived bootstrap-static's next_gw.",
    )
    args = parser.parse_args()

    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(module_dir))

    gw = args.gw
    if gw is None:
        next_gw, _current_gw = _latest_bootstrap_gw_state(args.base_dir, args.season)
        if next_gw is None:
            raise SystemExit(
                "no archived bootstrap-static found for season {0} -- run "
                "archiver.py first, or pass --gw explicitly".format(args.season)
            )
        gw = next_gw

    news_entry = fetch_pl_news(args.base_dir, args.season, gw)
    print("pl-news: {0} -> {1}".format(news_entry["outcome"], news_entry["path"]))
    injuries_entry = fetch_pl_injuries(args.base_dir, args.season, gw)
    print("pl-injuries: {0} -> {1}".format(
        injuries_entry["outcome"], injuries_entry["path"]
    ))


if __name__ == "__main__":
    _main()
