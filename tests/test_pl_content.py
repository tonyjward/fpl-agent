"""Tests for pl_content's parsing functions. These take payload dicts
directly (shaped like real responses captured from the live API) rather
than mocking requests, matching archiver.py's split between fetch and pure
parsing/validation logic.
"""

import requests

import pl_content


class FakeResponse(object):
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession(object):
    def __init__(self, text):
        self._text = text
        self.requested_url = None

    def get(self, url, timeout=None):
        self.requested_url = url
        return FakeResponse(self._text)


def test_parse_news_list_extracts_content():
    payload = {
        "pageInfo": {"page": 0, "numPages": 0, "pageSize": 2, "numEntries": 0},
        "content": [
            {"id": 1, "type": "text", "title": "A", "tags": []},
            {"id": 2, "type": "text", "title": "B", "tags": []},
        ],
    }
    articles = pl_content.parse_news_list(payload)
    assert [a["id"] for a in articles] == [1, 2]


def test_parse_news_list_handles_missing_content():
    assert pl_content.parse_news_list({}) == []


def test_parse_injury_hub_strips_title_prefix_and_keeps_id():
    payload = {
        "items": [
            {"response": {"id": 4509803, "title": "Injury News - Arsenal"}},
            {"response": {"id": 4509804, "title": "Injury News - Aston Villa"}},
        ]
    }
    clubs = pl_content.parse_injury_hub(payload)
    assert clubs == [("Arsenal", 4509803), ("Aston Villa", 4509804)]


def test_parse_injury_hub_skips_items_without_id():
    payload = {"items": [{"response": {"title": "Injury News - Nobody"}}]}
    assert pl_content.parse_injury_hub(payload) == []


def test_parse_club_injuries_extracts_player_injury_link():
    payload = {
        "items": [
            {
                "response": {
                    "type": "promo",
                    "title": "William Saliba",
                    "description": "Back",
                    "links": [{"promoUrl": "https://www.arsenal.com/news/x"}],
                }
            },
            {
                "response": {
                    "type": "promo",
                    "title": "Jurrien Timber",
                    "description": "Groin",
                    "links": [],
                }
            },
        ]
    }
    injuries = pl_content.parse_club_injuries(payload)
    assert injuries == [
        {"player": "William Saliba", "injury": "Back",
         "link": "https://www.arsenal.com/news/x"},
        {"player": "Jurrien Timber", "injury": "Groin", "link": None},
    ]


def test_parse_club_injuries_skips_non_promo_items():
    payload = {"items": [{"response": {"type": "playlist", "title": "Ignored"}}]}
    assert pl_content.parse_club_injuries(payload) == []


def test_get_article_text_prefers_native_body():
    article = {"body": "<p>" + "x" * 50 + "</p>", "hotlinkUrl": "https://example.com",
               "description": "short"}
    assert pl_content.get_article_text(article) == "x" * 50


def test_get_article_text_strips_script_and_short_fragments():
    html = (
        "<script>var x = 1;</script>"
        "<p>Nav</p>"
        "<p>" + "This is a real paragraph of article prose. " * 3 + "</p>"
    )
    article = {"body": html}
    text = pl_content.get_article_text(article)
    assert "var x" not in text
    assert "Nav" not in text
    assert "real paragraph" in text


def test_get_article_text_falls_back_to_hotlink_fetch_when_body_missing():
    html = "<p>" + "Full text from the club site itself, syndicated in. " * 2 + "</p>"
    session = FakeSession(html)
    article = {"body": None, "hotlinkUrl": "https://clubsite.example/news/1",
               "description": "short teaser"}

    text = pl_content.get_article_text(article, session=session)

    assert session.requested_url == "https://clubsite.example/news/1"
    assert "Full text from the club site" in text


def test_get_article_text_falls_back_to_description_when_nothing_else():
    article = {"body": None, "hotlinkUrl": None, "description": "short teaser"}
    assert pl_content.get_article_text(article) == "short teaser"


class BlockingSession(object):
    """Simulates a club site that actively rejects the request (e.g. a 403
    from bot detection), rather than one that just renders client-side.
    """

    def get(self, url, timeout=None):
        raise requests.exceptions.HTTPError("403 Client Error: Forbidden")


def test_get_article_text_falls_back_to_description_when_blocked_and_no_browser():
    article = {"body": None, "hotlinkUrl": "https://clubsite.example/news/1",
               "description": "short teaser"}
    session = BlockingSession()
    text = pl_content.get_article_text(article, session=session,
                                        allow_browser_fallback=False)
    assert text == "short teaser"


def test_get_article_text_uses_headless_browser_when_request_is_blocked(monkeypatch):
    """A blocked plain request (e.g. mancity.com's Cloudflare challenge)
    should fall through to a real headless browser rather than giving up
    straight to the description.
    """
    html = "<p>" + "Full text rendered only via a real browser session. " * 2 + "</p>"
    monkeypatch.setattr(pl_content, "_fetch_rendered_html", lambda url, **kw: html)

    article = {"body": None, "hotlinkUrl": "https://clubsite.example/news/1",
               "description": "short teaser"}
    session = BlockingSession()

    text = pl_content.get_article_text(article, session=session)

    assert "Full text rendered only via a real browser" in text


def test_fetch_rendered_html_is_noop_without_a_chrome_executable(monkeypatch):
    monkeypatch.setattr(pl_content, "_CHROME_EXECUTABLE", None)
    assert pl_content._fetch_rendered_html("https://clubsite.example/news/1") is None


def test_get_article_text_falls_back_to_description_when_hotlink_yields_nothing():
    """Some club sites are themselves client-rendered (no <p> in the raw
    HTML), just like premierleague.com's own listing pages -- the short
    description should still come through rather than an empty string.
    """
    session = FakeSession("<div id=\"app\"></div>")
    article = {"body": None, "hotlinkUrl": "https://clubsite.example/news/1",
               "description": "short teaser"}
    assert pl_content.get_article_text(article, session=session) == "short teaser"
