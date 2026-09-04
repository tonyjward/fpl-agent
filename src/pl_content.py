"""Thin client for premierleague.com's public content API.

This is the JSON API the premierleague.com website itself calls client-side
to render /en/news and /en/latest-player-injuries -- there is no published
API documentation for it. Found by pulling the site's JS bundle
(resources/v1.52.5/scripts/bundle-es.min.js) and reading its RTK Query
`contentApi` endpoint config, which resolves to base URL
`https://api.premierleague.com/content/premierleague` (from
`window.PULSE.envPaths.api`). No authentication is required.

Mirrors api.py's session/get_json shape.
"""

import os
import shutil

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://api.premierleague.com/content/premierleague"

# A club site can put a JS challenge in front of a plain HTTP request (seen:
# mancity.com, fronted by Cloudflare with `Cf-Mitigated: challenge`) -- a
# real, locally-launched headless browser gets past this, a bare `requests`
# call cannot. This is a browser this code drives itself, not any
# Claude-side browsing tool. Points at whatever Chrome/Chromium is already
# installed on the machine rather than a Playwright-managed download, so it
# degrades to no-op (see _fetch_rendered_html) on a machine with neither.
_CHROME_EXECUTABLE = (
    os.environ.get("PL_CONTENT_CHROME_PATH")
    or shutil.which("google-chrome")
    or shutil.which("chromium")
    or shutil.which("chromium-browser")
)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# The `data-playlist-id` on the `data-widget="injury-news/injury-news"`
# element embedded in /en/latest-player-injuries' static HTML shell (the
# page's listing/table is client-rendered, but this container div -- and
# its id -- is server-rendered, so it's readable without executing any JS).
# This "hub" playlist's items are one sub-playlist per club; each club
# sub-playlist's own id is what get_club_injuries needs.
#
# This id is a CMS content id, not a per-season or per-matchweek value, so
# it's expected to stay stable across gameweeks and seasons. If
# list_injury_playlists() starts returning nothing, re-fetch
# https://www.premierleague.com/en/latest-player-injuries and grep the raw
# HTML for `data-widget="injury-news/injury-news"` to find the current id.
INJURY_HUB_PLAYLIST_ID = 4509826


def new_session():
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


_session = new_session()


def get_json(path):
    """GET a path under the content API base URL and return the parsed JSON."""
    resp = _session.get("{0}{1}".format(BASE_URL, path), timeout=15)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------
# News articles
# --------------------------------------------------------------------------


def parse_news_list(payload):
    """Extract the list of article summaries from a news-list payload.

    Each summary carries id, title, description, date, tags (club/team
    references live here) and hotlinkUrl, but not the article body -- fetch
    get_article(id) for that.
    """
    return payload.get("content") or []


def list_news(limit=20, offset=0):
    """Return one page of news article summaries, newest first."""
    path = "/en?contentTypes=text&limit={0}&offset={1}".format(limit, offset)
    return parse_news_list(get_json(path))


def get_article(article_id):
    """Return the full article detail for one id from list_news(), including
    the `body` field (article HTML) for natively-published items -- see
    get_article_text for items where `body` is null instead.
    """
    return get_json("/text/en/{0}".format(article_id))


def html_to_text(html):
    """Strip an article's `body` HTML down to its paragraph text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) > 40]
    return "\n\n".join(paragraphs)


def _fetch_rendered_html(url, timeout_ms=30000):
    """Load `url` in a real, locally-launched headless Chrome and return the
    rendered HTML, for sites that block a plain HTTP request behind a JS
    challenge. Returns None (rather than raising) if no Chrome/Chromium
    executable is available on this machine -- callers treat that the same
    as any other extraction failure and fall back to a shorter text.
    """
    if not _CHROME_EXECUTABLE:
        return None
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=_CHROME_EXECUTABLE, headless=True,
            args=["--no-sandbox"],
        )
        try:
            page = browser.new_page(user_agent=_USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            return page.content()
        finally:
            browser.close()


def fetch_raw_html(url, session=None, allow_browser_fallback=True):
    """Fetch `url` and return (html, method), trying a plain HTTP request
    first and falling back to a real headless browser (see
    _fetch_rendered_html) if that's blocked -- e.g. a Cloudflare challenge,
    seen on mancity.com. `method` is "http" or "browser" on success; on
    total failure returns (None, "blocked").

    This is the raw fetch step behind fetch_external_article_text, split
    out so callers that need the actual bytes -- news_archiver.py archives
    them verbatim, per the raw layer's "never parse before archiving"
    invariant -- don't have to duplicate the fetch/fallback logic.
    """
    session = session or _session
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text, "http"
    except requests.exceptions.RequestException:
        if not allow_browser_fallback:
            return None, "blocked"
        html = _fetch_rendered_html(url)
        if html is None:
            return None, "blocked"
        return html, "browser"


def fetch_external_article_text(hotlink_url, session=None,
                                 allow_browser_fallback=True):
    """Best-effort fetch-and-extract of the article text from a club's own
    site, for syndicated (RSS/External) news items whose `body` from the
    content API is null -- the real text lives at `hotlinkUrl` instead.

    Club sites vary in markup, so extraction itself is a generic heuristic
    (every <p> in the page long enough to plausibly be prose, not
    boilerplate), not a per-club parser -- it can return noise or miss text
    on any given site. Treat the result as best-effort, not authoritative.
    """
    html, _method = fetch_raw_html(
        hotlink_url, session=session, allow_browser_fallback=allow_browser_fallback,
    )
    if html is None:
        return ""
    return html_to_text(html)


def get_article_text(article, session=None, allow_browser_fallback=True):
    """Return the best available body text for an article dict from
    get_article(): the parsed native `body` HTML if present, otherwise a
    best-effort fetch from the syndication source at `hotlinkUrl` (see
    fetch_external_article_text), falling back to the short `description`
    if that source has nothing extractable (e.g. it's blocked outright, or
    it's itself a client-rendered page with no <p> tags in the raw HTML --
    some club sites are, just as premierleague.com's own listing pages
    are).
    """
    body = article.get("body")
    if body:
        return html_to_text(body)
    hotlink_url = article.get("hotlinkUrl")
    if hotlink_url:
        text = fetch_external_article_text(
            hotlink_url, session=session,
            allow_browser_fallback=allow_browser_fallback,
        )
        if text:
            return text
    return article.get("description") or ""


# --------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------


_INJURY_TITLE_PREFIX = "Injury News - "


def parse_injury_hub(payload):
    """Extract [(club_name, club_playlist_id), ...] from the injury hub
    playlist payload (see INJURY_HUB_PLAYLIST_ID).
    """
    clubs = []
    for item in payload.get("items") or []:
        response = item.get("response") or {}
        club_id = response.get("id")
        title = response.get("title") or ""
        if club_id is None:
            continue
        if title.startswith(_INJURY_TITLE_PREFIX):
            title = title[len(_INJURY_TITLE_PREFIX):]
        clubs.append((title, club_id))
    return clubs


def list_injury_playlists(hub_playlist_id=INJURY_HUB_PLAYLIST_ID):
    """Return [(club_name, club_playlist_id), ...], one entry per club."""
    payload = get_json("/PLAYLIST/en/{0}?detail=DETAILED".format(hub_playlist_id))
    return parse_injury_hub(payload)


def parse_club_injuries(payload):
    """Extract [{"player", "injury", "link"}, ...] from one club's injury
    playlist payload (see list_injury_playlists).
    """
    injuries = []
    for item in payload.get("items") or []:
        response = item.get("response") or {}
        if response.get("type") != "promo":
            continue
        links = response.get("links") or []
        injuries.append({
            "player": response.get("title"),
            "injury": response.get("description"),
            "link": links[0].get("promoUrl") if links else None,
        })
    return injuries


def get_club_injuries(club_playlist_id):
    """Return the current injury list for one club, as
    [{"player", "injury", "link"}, ...].
    """
    path = "/PLAYLIST/en/{0}?detail=DETAILED".format(club_playlist_id)
    return parse_club_injuries(get_json(path))


def get_all_injuries(hub_playlist_id=INJURY_HUB_PLAYLIST_ID):
    """Return {club_name: [{"player", "injury", "link"}, ...]} across every
    club. Makes one request per club (20) plus one for the hub.
    """
    return {
        club_name: get_club_injuries(club_id)
        for club_name, club_id in list_injury_playlists(hub_playlist_id)
    }
