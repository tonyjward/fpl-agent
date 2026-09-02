"""Thin client for the public FPL API.

The API rejects requests that don't look like they come from a browser
(docs/spec_v3.odt §3.2), so every request goes through a session carrying a
real User-Agent.
"""

import requests

BASE_URL = "https://fantasy.premierleague.com/api"

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


_session = new_session()


def get_json(path: str) -> dict:
    """GET a path under the FPL API base URL and return the parsed JSON."""
    resp = _session.get(f"{BASE_URL}/{path}", timeout=15)
    resp.raise_for_status()
    return resp.json()
