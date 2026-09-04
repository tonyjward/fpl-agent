"""LLM extraction of start-relevant claims from scraped news article text.

The only LLM call in this project's evidence layer (docs/README.md's "Where
the LLM is used" -- reading prose, nothing else). Its job is narrow and
deliberately does not include producing a probability: classify each
player mention in one article's text into a small fixed taxonomy
(CLAIM_CATEGORIES), with a verbatim quote backing the claim. Turning a
category into a number that adjusts P(starts) is a separate, code-owned
step (see starts_model.py) -- this keeps "never treat a flag percentage
as a probability" (docs/README.md) from recurring in a worse form, where
an LLM invents an ungrounded, unauditable float per article.

Output is a write-once JSON artifact under EXTRACTIONS_DIR, mirroring
starts_model.py's snapshot_predictions -- not part of the raw archive (an
LLM call is neither free nor byte-reproducible, so it can't be regenerated
by derived.py's rebuild the way every raw-archive-backed table is) and not
silently re-run: derived.py's _load_news_claims reads these files back,
exactly like it reads predictions/.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import json
import os

import archiver

EXTRACTIONS_DIR = "extractions"

MODEL = "claude-opus-5"

# A category is a discrete, auditable label -- never a number. What number
# (if any) a category maps to for P(starts) is starts_model.py's decision,
# made once in code and revisited via scoring.py, not the LLM's per-article
# guess. Deliberately narrow: a category that isn't confidently one of
# these is not extracted at all, rather than forcing a bad fit.
CLAIM_CATEGORIES = [
    "confirmed_starting",
    "confirmed_out",
    "rotation_risk",
    "returning_from_injury",
]

_SYSTEM_PROMPT = """You classify player-availability claims in a football \
(soccer) news article. You are not predicting anything and you must not \
invent a probability, percentage, or confidence score of any kind -- your \
only output is a short list of discrete claims.

For every player the article makes an explicit, start-relevant claim \
about, emit one entry with:
- "player_name": the player's name as written in the article
- "category": exactly one of {categories}
- "quote": the exact sentence or clause from the article that supports \
this claim, copied verbatim (do not paraphrase, do not summarize)

Category meanings:
- confirmed_starting: the article states the player will start, or is in \
the starting lineup, for the next fixture
- confirmed_out: the article states the player is unavailable, ruled out, \
or will definitely not play the next fixture
- rotation_risk: the article mentions rotation, resting, or managing the \
player's minutes around the next fixture
- returning_from_injury: the article discusses the player returning from \
injury/suspension with some caveat about game time or fitness, without a \
clear starting/out claim

Only emit a claim when the article is genuinely explicit -- a passing \
mention of a player's name with no start-relevant statement is not a \
claim. If the article makes no such claim about any player, return an \
empty list. Do not guess, and do not infer a claim from silence.""".format(
    categories=", ".join(CLAIM_CATEGORIES)
)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string"},
                    "category": {"type": "string", "enum": CLAIM_CATEGORIES},
                    "quote": {"type": "string"},
                },
                "required": ["player_name", "category", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def _normalize_for_substring_check(text):
    return " ".join((text or "").lower().split())


def _verify_quote(quote, body_text):
    """A claim's quote must actually appear in the source article --
    a cheap, mechanical guard against the model fabricating a claim rather
    than reading one, independent of whether the category itself is right.
    Whitespace-normalized, case-insensitive: the model may reflow line
    breaks even when copying "verbatim".
    """
    return _normalize_for_substring_check(quote) in _normalize_for_substring_check(body_text)


def build_request(article, model=MODEL, effort="low"):
    """The exact request payload sent to the Messages API for one article,
    exposed separately from extract_claims so it can be inspected/tested
    without a network call. `effort="low"` by default: this is a
    high-volume (one call per article, ~20-50/day), narrowly-scoped
    classification task -- exactly the workload shape the model reference
    says does well at low effort, not a coding or long-horizon task.
    """
    content = "Title: {0}\n\n{1}".format(
        article.get("title") or "", article.get("body_text") or "",
    )
    return {
        "model": model,
        "max_tokens": 2048,
        "system": _SYSTEM_PROMPT,
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA},
        },
        "messages": [{"role": "user", "content": content}],
    }


def parse_response(response, body_text):
    """Extract and validate the claims list from a Messages API response.
    Drops (rather than raises on) any claim whose quote doesn't verify
    against `body_text` -- one bad claim in a batch shouldn't sink the
    others, but an unverifiable one must never reach the derived layer
    silently trusted.
    """
    text = next(block.text for block in response.content if block.type == "text")
    payload = json.loads(text)
    claims = []
    for claim in payload.get("claims") or []:
        if _verify_quote(claim.get("quote"), body_text):
            claims.append(claim)
    return claims


def extract_claims(article, client, model=MODEL, effort="low"):
    """Classify player claims in one article's text. `article` is a row
    from news_articles (needs at least "title" and "body_text"). `client`
    is an anthropic.Anthropic()-shaped client, passed in rather than
    constructed here so tests can inject a fake -- see make_http_get in
    archiver.py for the same seam pattern.
    """
    request = build_request(article, model=model, effort=effort)
    response = client.messages.create(**request)
    return parse_response(response, article.get("body_text") or "")


# --------------------------------------------------------------------------
# Persistence -- write-once JSON artifact, mirroring
# starts_model.snapshot_predictions
# --------------------------------------------------------------------------


def save_extraction(article_id, claims, season, model=MODEL,
                     base_dir=EXTRACTIONS_DIR, clock=archiver.utcnow):
    """Write one article's extracted claims to a write-once, timestamped
    JSON file under base_dir/season/. Refuses to overwrite, same as every
    other write-once artifact in this project.
    """
    fetched_at = clock()
    filename = "article_{0}_{1}.json".format(
        article_id, archiver.format_timestamp(fetched_at)
    )
    path = os.path.join(base_dir, season, filename)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    if os.path.exists(path):
        raise archiver.ArchiveError(
            "refusing to overwrite existing extraction snapshot: {0}".format(path)
        )
    payload = {
        "article_id": article_id,
        "season": season,
        "model": model,
        "extracted_at": archiver.format_timestamp(fetched_at),
        "claims": claims,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def extract_and_save(article, season, client, model=MODEL, effort="low",
                      base_dir=EXTRACTIONS_DIR, clock=archiver.utcnow):
    """extract_claims() then save_extraction() -- the one call a daily job
    makes per not-yet-extracted article.
    """
    claims = extract_claims(article, client, model=model, effort=effort)
    return save_extraction(
        article["article_id"], claims, season, model=model,
        base_dir=base_dir, clock=clock,
    )


def _already_extracted_article_ids(base_dir, season):
    """Article ids that already have an extraction file on disk -- an
    article's text doesn't change once archived, so re-extracting it is
    pure repeat LLM cost for zero new information, same reasoning as
    news_archiver.py's own dedup of the external-site fetch.
    """
    season_dir = os.path.join(base_dir, season)
    if not os.path.isdir(season_dir):
        return set()
    ids = set()
    for name in os.listdir(season_dir):
        if name.startswith("article_") and name.endswith(".json"):
            with open(os.path.join(season_dir, name)) as f:
                ids.add(json.load(f)["article_id"])
    return ids


def _main():
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(
        description="Extract start-relevant claims from not-yet-extracted "
                    "news articles in derived.db."
    )
    parser.add_argument("--db-path", default="derived.db")
    parser.add_argument("--season", required=True)
    parser.add_argument("--base-dir", default=EXTRACTIONS_DIR)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Max not-yet-extracted articles to process this run (cost control).",
    )
    args = parser.parse_args()

    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(module_dir))

    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "the 'anthropic' package isn't installed -- run `uv add anthropic`"
        )
    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant auth login

    conn = sqlite3.connect(args.db_path)
    already_done = _already_extracted_article_ids(args.base_dir, args.season)
    cur = conn.execute(
        "SELECT article_id, title, body_text FROM news_articles WHERE season = ?",
        (args.season,),
    )
    columns = [d[0] for d in cur.description]
    articles = [
        dict(zip(columns, row)) for row in cur.fetchall()
        if row[0] not in already_done
    ][:args.limit]
    conn.close()

    n_claims = 0
    for article in articles:
        path = extract_and_save(article, args.season, client, model=args.model,
                                 base_dir=args.base_dir)
        with open(path) as f:
            n_claims += len(json.load(f)["claims"])

    print("{0}: extracted {1} article(s), {2} claim(s) -> {3}/{4}/".format(
        args.season, len(articles), n_claims, args.base_dir, args.season,
    ))


if __name__ == "__main__":
    _main()
