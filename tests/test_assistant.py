"""The assistant's safety rails: the number check and argument validation (no network)."""
import json

from reach120 import assistant, numbercheck


def test_numbers_from_tool_results_pass():
    tool = json.dumps({"share_beyond_120_min_pct": 12.34, "beyond_120_min": 104512, "budget_naira": 2000000000,
                       "share": 0.0712, "lga": "Ward 7 (Dala)"})
    text = ("About 12.3% of women (104,512) are more than 2 hours away. The plan costs ₦2.00 billion; "
            "7.1% remain. Ward 7 is worst.")
    res = numbercheck.verify(text, [tool, "budget 2 billion, target 2 hours"])
    assert res.ok, res.unverified


def test_invented_numbers_fail():
    tool = json.dumps({"beyond_120_min": 104512})
    res = numbercheck.verify("Roughly 150,000 women, or 15% of them, are far away.", [tool])
    assert not res.ok
    assert set(res.unverified) == {"150,000", "15%"}


def test_list_markers_are_not_numbers():
    tool = json.dumps({"rows": [{"lga": "Gwale", "share_beyond_120_min_pct": 40.5}]})
    res = numbercheck.verify("Worst LGAs:\n1. Gwale, 40.5%\n2) none else", [tool])
    assert res.ok, res.unverified


def test_rounding_must_match_the_precision_shown():
    tool = json.dumps({"value": 1249.6})
    assert numbercheck.verify("1,250 women", [tool]).ok
    assert not numbercheck.verify("1,260 women", [tool]).ok


def test_argument_validation():
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in assistant.SCHEMAS}
    rank = schemas["rank_upgrades"]
    assert assistant.validate({"budget_naira": 2e9, "scenario": "base"}, rank) == []
    assert assistant.validate({"scenario": "base"}, rank) == ["missing budget_naira"]
    assert assistant.validate({"budget_naira": -5}, rank) == ["budget_naira must be at least 0"]
    assert assistant.validate({"budget_naira": 1, "scenario": "everything"}, rank)[0].startswith("scenario must be one of")
    assert assistant.validate({"budget_naira": 1, "drop_tables": True}, rank) == ["unknown argument drop_tables"]
    assert assistant.validate({"budget_naira": 1, "tier_costs_naira": {"phc": "cheap"}}, rank) == ["tier_costs_naira.phc must be a number"]


def test_assistant_is_off_without_a_key(monkeypatch):
    for var in ("REACH120_LLM_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert assistant.LLMClient.from_settings() is None
