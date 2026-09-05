"""Tests for news_extraction.py: the LLM classification step, and its
write-once persistence.
"""

import json
import os

import pytest

import archiver
import news_extraction

SEASON = "2026-27"


class FakeTextBlock(object):
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse(object):
    def __init__(self, content):
        self.content = content


class FakeMessages(object):
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class FakeClient(object):
    def __init__(self, response):
        self.messages = FakeMessages(response)


class SequenceClock(object):
    def __init__(self, timestamps):
        self._timestamps = list(timestamps)

    def __call__(self):
        return self._timestamps.pop(0)


ARTICLE = {
    "article_id": 42,
    "title": "Farke provides Rodon injury update",
    "body_text": (
        "Daniel Farke confirmed Joe Rodon has a hamstring injury and will "
        "be out for eight to 10 weeks. He also said Jayden Bogle will "
        "start on Saturday after returning from a knock."
    ),
}


def response_with_claims(claims):
    return FakeResponse([FakeTextBlock(json.dumps({"claims": claims}))])


# --------------------------------------------------------------------------
# build_request / parse_response
# --------------------------------------------------------------------------


def test_build_request_never_asks_for_a_probability():
    request = news_extraction.build_request(ARTICLE)
    # The one hard requirement: nothing in the schema or prompt admits a
    # numeric confidence/probability field for the model to fill in.
    schema_str = json.dumps(request["extra_body"]["output_config"]["format"]["schema"])
    assert "probability" not in schema_str
    assert "confidence" not in schema_str
    assert "percent" not in schema_str.lower()
    # The prompt explicitly prohibits inventing one, rather than merely
    # not mentioning the concept at all.
    assert "must not invent a probability" in request["system"].lower()
    assert set(request["extra_body"]["output_config"]["format"]["schema"]
               ["properties"]["claims"]["items"]["properties"]["category"]["enum"]) \
        == set(news_extraction.CLAIM_CATEGORIES)


def test_build_request_includes_title_and_body():
    request = news_extraction.build_request(ARTICLE)
    content = request["messages"][0]["content"]
    assert ARTICLE["title"] in content
    assert "Rodon" in content


def test_build_request_omits_effort_for_models_that_reject_it():
    # Confirmed live (2026-09-05): claude-haiku-4-5 400s on
    # output_config.effort -- "This model does not support the effort
    # parameter." Must be left out entirely, not sent as e.g. None.
    request = news_extraction.build_request(ARTICLE, model="claude-haiku-4-5")
    assert "effort" not in request["extra_body"]["output_config"]


def test_build_request_includes_effort_for_models_that_support_it():
    request = news_extraction.build_request(ARTICLE, model="claude-opus-5", effort="low")
    assert request["extra_body"]["output_config"]["effort"] == "low"


def test_parse_response_extracts_verified_claims():
    claims = [
        {"player_name": "Joe Rodon", "category": "confirmed_out",
         "quote": "Joe Rodon has a hamstring injury"},
        {"player_name": "Jayden Bogle", "category": "confirmed_starting",
         "quote": "Jayden Bogle will start on Saturday"},
    ]
    result = news_extraction.parse_response(
        response_with_claims(claims), ARTICLE["body_text"],
    )
    assert result == claims


def test_parse_response_drops_claims_with_unverifiable_quotes():
    claims = [
        {"player_name": "Joe Rodon", "category": "confirmed_out",
         "quote": "Joe Rodon has a hamstring injury"},
        {"player_name": "Someone Else", "category": "confirmed_starting",
         "quote": "This sentence was never in the article"},
    ]
    result = news_extraction.parse_response(
        response_with_claims(claims), ARTICLE["body_text"],
    )
    assert [c["player_name"] for c in result] == ["Joe Rodon"]


def test_parse_response_quote_check_is_whitespace_and_case_insensitive():
    claims = [{
        "player_name": "Joe Rodon", "category": "confirmed_out",
        "quote": "  JOE   RODON has a hamstring\ninjury  ",
    }]
    result = news_extraction.parse_response(
        response_with_claims(claims), ARTICLE["body_text"],
    )
    assert len(result) == 1


def test_parse_response_handles_no_claims():
    result = news_extraction.parse_response(response_with_claims([]), ARTICLE["body_text"])
    assert result == []


class FakeResponseWithStopReason(object):
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


def test_parse_response_raises_truncated_error_when_cut_off_by_max_tokens():
    """Confirmed live: a long article can exceed max_tokens mid-generation,
    breaking output_config's "always valid JSON" guarantee since that
    assumes generation actually finishes.
    """
    truncated = FakeResponseWithStopReason(
        [FakeTextBlock('{"claims": [{"player_name": "Joe Rodon", "categ')],
        stop_reason="max_tokens",
    )
    with pytest.raises(news_extraction.TruncatedResponseError):
        news_extraction.parse_response(truncated, ARTICLE["body_text"])


def test_parse_response_reraises_plain_json_error_when_not_truncated():
    """A malformed response NOT explained by hitting max_tokens is a
    genuine parsing bug, not the known truncation case -- must not be
    silently reclassified as the same thing.
    """
    broken = FakeResponseWithStopReason(
        [FakeTextBlock("not json at all")], stop_reason="end_turn",
    )
    with pytest.raises(ValueError):
        news_extraction.parse_response(broken, ARTICLE["body_text"])


# --------------------------------------------------------------------------
# extract_claims -- the injected-client seam
# --------------------------------------------------------------------------


def test_extract_claims_calls_client_and_returns_verified_claims():
    claims = [{"player_name": "Joe Rodon", "category": "confirmed_out",
               "quote": "Joe Rodon has a hamstring injury"}]
    client = FakeClient(response_with_claims(claims))

    result = news_extraction.extract_claims(ARTICLE, client)

    assert result == claims
    assert client.messages.last_kwargs["model"] == news_extraction.MODEL
    # MODEL (claude-haiku-4-5) is in _MODELS_WITHOUT_EFFORT -- see
    # test_build_request_omits_effort_for_models_that_reject_it.
    assert "effort" not in client.messages.last_kwargs["extra_body"]["output_config"]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_save_extraction_writes_write_once_json(tmp_path):
    base_dir = str(tmp_path)
    claims = [{"player_name": "Joe Rodon", "category": "confirmed_out",
               "quote": "x"}]
    clock = SequenceClock([__import__("datetime").datetime(
        2026, 9, 4, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc,
    )])

    path = news_extraction.save_extraction(
        42, claims, SEASON, base_dir=base_dir, clock=clock,
    )

    assert os.path.exists(path)
    with open(path) as f:
        payload = json.load(f)
    assert payload["article_id"] == 42
    assert payload["season"] == SEASON
    assert payload["claims"] == claims
    assert payload["model"] == news_extraction.MODEL


def test_save_extraction_refuses_to_overwrite(tmp_path):
    base_dir = str(tmp_path)
    ts = __import__("datetime").datetime(
        2026, 9, 4, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc,
    )
    news_extraction.save_extraction(42, [], SEASON, base_dir=base_dir,
                                     clock=SequenceClock([ts]))
    with pytest.raises(archiver.ArchiveError):
        news_extraction.save_extraction(42, [], SEASON, base_dir=base_dir,
                                         clock=SequenceClock([ts]))


def test_already_extracted_article_ids_reads_from_disk(tmp_path):
    base_dir = str(tmp_path)
    ts = __import__("datetime").datetime(
        2026, 9, 4, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc,
    )
    news_extraction.save_extraction(1, [], SEASON, base_dir=base_dir,
                                     clock=SequenceClock([ts]))
    news_extraction.save_extraction(2, [], SEASON, base_dir=base_dir,
                                     clock=SequenceClock([ts]))
    assert news_extraction._already_extracted_article_ids(base_dir, SEASON) == {1, 2}


def test_already_extracted_article_ids_empty_when_no_season_dir(tmp_path):
    assert news_extraction._already_extracted_article_ids(str(tmp_path), SEASON) == set()


def test_extract_and_save_round_trips(tmp_path):
    base_dir = str(tmp_path)
    claims = [{"player_name": "Joe Rodon", "category": "confirmed_out",
               "quote": "Joe Rodon has a hamstring injury"}]
    client = FakeClient(response_with_claims(claims))
    ts = __import__("datetime").datetime(
        2026, 9, 4, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc,
    )

    path = news_extraction.extract_and_save(
        ARTICLE, SEASON, client, base_dir=base_dir, clock=SequenceClock([ts]),
    )

    with open(path) as f:
        payload = json.load(f)
    assert payload["claims"] == claims
