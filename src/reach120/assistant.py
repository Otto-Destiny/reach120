"""The planning assistant: a language model that can only answer by calling Reach120's own functions.

* **Provider:** any OpenAI-compatible chat endpoint. The default is Groq's free tier; set
  ``REACH120_LLM_API_KEY`` (or ``GROQ_API_KEY``), and optionally ``REACH120_LLM_BASE_URL`` and
  ``REACH120_LLM_MODEL``. With no key the assistant is off and everything else still works.
* **Tools:** the functions below. Arguments are validated against their schemas in this
  code before anything runs.
* **Number check:** an answer is shown only if every number in it appears in a tool result
  from the same turn (see numbercheck.py). One retry, then a plain refusal.
* **Budget:** at most ``max_tool_calls`` tool calls per question, to stay inside free quotas.
* **Log:** every question, tool call and answer goes to outputs/<state>/assistant_log.jsonl.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

from . import display, numbercheck, scenarios
from .config import load, settings
from .store import StateStore, read_json

SYSTEM = """You are Reach120's planning assistant for {state} State, Nigeria. You help health planners understand how far women live from hospitals that can likely perform an emergency caesarean section, and compare hospital upgrades within a budget.

Rules:
1. Every number you state must come from a tool result in this conversation. Copy numbers exactly as the tools give them. Never estimate, extrapolate or compute new figures. Write large numbers in full or as the tools format them, never with "k" or "m".
2. Say which capability scenario and transport mode a figure uses.
3. You do not give medical or clinical advice and you do not diagnose. If asked, decline and suggest contacting a health worker or the local emergency service.
4. You only have data for {state}. If asked about another state, say it is not loaded.
5. Upgrade costs are placeholders unless a tool says otherwise. Say so when you report costs.
6. If the tools cannot answer, say so. Keep answers to a few sentences or a short list.

Scenarios: strict, base, loose (and base_public for access figures only). Transport modes: mixed (vehicles on passable roads, walking elsewhere) and walk."""


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Minimal client for OpenAI-compatible chat completions with tool calling."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    @classmethod
    def from_settings(cls) -> "LLMClient | None":
        cfg = settings()["assistant"]
        key = next((os.environ[k] for k in cfg["api_key_env"] if os.environ.get(k)), None)
        if not key:
            return None
        return cls(os.environ.get(cfg["base_url_env"]) or cfg["base_url"], key,
                   os.environ.get(cfg["model_env"]) or cfg["model"], cfg.get("timeout_s", 60))

    def chat(self, messages, tools=None, tool_choice: str = "auto", temperature: float = 0) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}"})
        except requests.RequestException as e:
            raise LLMError(f"could not reach the language model: {e}") from e
        if r.status_code == 429:
            raise LLMError("the free quota is used up for now; try again later (HTTP 429)")
        if r.status_code >= 400:
            raise LLMError(f"the language model returned HTTP {r.status_code}: {r.text[:300]}")
        return r.json()


# ----------------------------------------------------------------------------- tools

def _schema(name, description, properties=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}}}


SCEN = {"type": "string", "enum": ["strict", "base", "loose", "base_public"]}
PLAN_SCEN = {"type": "string", "enum": ["strict", "base", "loose"]}
MODE = {"type": "string", "enum": ["mixed", "walk"]}

SCHEMAS = [
    _schema("access_summary", "Women aged 15-49 in the state within 30, 60 and 120 minutes of a capable hospital, and beyond.",
            {"scenario": SCEN, "mode": MODE}),
    _schema("area_table", "The worst-served wards or LGAs: women more than two hours away and their share.",
            {"level": {"type": "string", "enum": ["ward", "lga"]}, "scenario": SCEN, "mode": MODE,
             "sort": {"type": "string", "enum": ["share", "women"]},
             "limit": {"type": "integer", "minimum": 1, "maximum": 25}}),
    _schema("rank_upgrades", "Choose facility upgrades that bring the most women within two hours for a budget in naira. "
            "Runs the optimiser; returns the chosen sites, cost and women reached, with greedy and straight-line comparisons.",
            {"budget_naira": {"type": "number", "minimum": 0, "maximum": 1e12}, "scenario": PLAN_SCEN, "mode": MODE,
             "tier_costs_naira": {"type": "object", "additionalProperties": False, "properties": {
                 "hospital": {"type": "number", "minimum": 0}, "secondary": {"type": "number", "minimum": 0},
                 "phc": {"type": "number", "minimum": 0}}}},
            required=["budget_naira"]),
    _schema("compare_runs", "Compare two saved upgrade plans by their run_id.",
            {"run_a": {"type": "string"}, "run_b": {"type": "string"}}, required=["run_a", "run_b"]),
    _schema("facility_lookup", "Find facilities in the state by name, with type, ownership, status and whether they count as capable.",
            {"query": {"type": "string", "minLength": 2}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
            required=["query"]),
    _schema("build_brief", "Build the one-page PDF brief for the state, optionally for a saved plan run_id.",
            {"run_id": {"type": "string"}}),
]


def validate(args, schema) -> list[str]:
    """The subset of JSON Schema the tools use."""
    errors = []
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            errors.append(f"missing {req}")
    for key, val in args.items():
        if key not in props:
            if schema.get("additionalProperties") is False:
                errors.append(f"unknown argument {key}")
            continue
        spec = props[key]
        t = spec.get("type")
        okay = {"string": isinstance(val, str), "integer": isinstance(val, int) and not isinstance(val, bool),
                "number": isinstance(val, (int, float)) and not isinstance(val, bool), "boolean": isinstance(val, bool),
                "object": isinstance(val, dict)}.get(t, True)
        if not okay:
            errors.append(f"{key} must be a {t}")
            continue
        if "enum" in spec and val not in spec["enum"]:
            errors.append(f"{key} must be one of {spec['enum']}")
        if t in ("number", "integer"):
            if "minimum" in spec and val < spec["minimum"]:
                errors.append(f"{key} must be at least {spec['minimum']}")
            if "maximum" in spec and val > spec["maximum"]:
                errors.append(f"{key} must be at most {spec['maximum']}")
        if t == "string" and len(val) < spec.get("minLength", 0):
            errors.append(f"{key} is too short")
        if t == "object":
            errors += [f"{key}.{e}" for e in validate(val, spec)]
    return errors


class ToolBox:
    def __init__(self, state: str):
        self.state = state
        self.store = StateStore(state)
        self.summary = read_json(self.store.summary)
        self._inputs = {}
        self.schemas = SCHEMAS
        self._by_name = {s["function"]["name"]: s["function"]["parameters"] for s in SCHEMAS}

    def call(self, name: str, args: dict) -> dict:
        if name not in self._by_name:
            return {"error": f"unknown tool {name}"}
        errors = validate(args, self._by_name[name])
        if errors:
            return {"error": "invalid arguments: " + "; ".join(errors)}
        try:
            return getattr(self, name)(**args)
        except FileNotFoundError as e:
            return {"error": f"that result has not been computed yet: {e}"}
        except (KeyError, ValueError) as e:
            return {"error": str(e)}

    def _context(self, scenario, mode):
        return {"state": self.state, "scenario": scenario, "scenario_label": scenarios.label(scenario),
                "mode": mode, "target_min": self.summary["target_min"], "target_hours": self.summary["target_min"] / 60}

    def access_summary(self, scenario: str = "base", mode: str = "mixed") -> dict:
        run = self.summary["runs"].get(f"{scenario}/{mode}")
        if run is None:
            return {"error": f"no run for {scenario}/{mode}"}
        lo, hi = self.summary.get("range_share_beyond_target") or (None, None)
        return {**self._context(scenario, mode), "capable_facilities_used": run["sources"],
                "women_15_49": round(run["women"]), "within_30_min": round(run["within_30"]),
                "within_60_min": round(run["within_60"]), "within_120_min": round(run["within_120"]),
                "beyond_120_min": round(run["beyond_target"]), "no_route_found": round(run["no_route"]),
                "share_beyond_120_min_pct": round(100 * run["share_beyond_target"], 1),
                "range_share_beyond_120_min_pct_all_runs": [round(100 * lo, 1), round(100 * hi, 1)] if lo is not None else None}

    def area_table(self, level: str = "ward", scenario: str = "base", mode: str = "mixed", sort: str = "share", limit: int = 10) -> dict:
        from .tables import lga_table

        table = pd.read_parquet(self.store.coverage(scenario, mode))
        if level == "lga":
            table = lga_table(table)
        table = table[table["women"] >= 1]
        table = table.sort_values("share_beyond_target" if sort == "share" else "beyond_target", ascending=False).head(limit)
        cols = ["lga", "women", "beyond_target", "share_beyond_target"] if level == "lga" else \
               ["ward", "lga", "women", "beyond_target", "share_beyond_target"]
        rows = [{**{c: r[c] for c in cols if c in ("ward", "lga")}, "women_15_49": round(float(r["women"])),
                 "women_beyond_120_min": round(float(r["beyond_target"])),
                 "share_beyond_120_min_pct": round(100 * float(r["share_beyond_target"]), 1)} for _, r in table.iterrows()]
        return {**self._context(scenario, mode), "level": level, "rows": rows}

    def rank_upgrades(self, budget_naira: float, scenario: str = "base", mode: str = "mixed",
                      tier_costs_naira: dict | None = None) -> dict:
        from . import plan

        if mode not in self._inputs:
            self._inputs[mode] = plan.load_inputs(self.state, mode)
        res = plan.run_plan(self._inputs[mode], scenario, float(budget_naira), tier_costs=tier_costs_naira,
                            time_limit=min(10, settings()["planning"]["solver_time_limit_s"]))
        base_share = res["baseline"]["share_beyond_target"]
        after_share = res["after"]["share_beyond_target"]
        return {**self._context(scenario, mode), "run_id": res["run_id"],
                "budget_naira": res["budget_naira"], "budget": display.format_naira(res["budget_naira"]),
                "costs_are_placeholders": res["costs_are_placeholders"], "tier_costs_naira": res["tier_costs_naira"],
                "sites_chosen": len(res["plan"]["sites"]), "total_cost_naira": res["plan"]["cost_naira"],
                "total_cost": display.format_naira(res["plan"]["cost_naira"]),
                "women_newly_within_120_min": round(res["plan"]["women_newly_within_target"]),
                "share_beyond_120_min_pct_before": round(100 * base_share, 1) if base_share is not None else None,
                "share_beyond_120_min_pct_after": round(100 * after_share, 1) if after_share is not None else None,
                "greedy_women": round(res["greedy"]["women_newly_within_target"]),
                "straight_line_women": round(res["straight_line"]["women_newly_within_target"]),
                "sites": [{"facility_name": s["facility_name"], "lga": s["lga"], "tier": s["tier"],
                           "cost_naira": s["cost_naira"], "cost": display.format_naira(s["cost_naira"]),
                           "women_newly_within_120_min": round(s["women_newly_within_target"])} for s in res["plan"]["sites"]]}

    def compare_runs(self, run_a: str, run_b: str) -> dict:
        a, b = (read_json(self.store.plan(r)) for r in (run_a, run_b))
        sa = {s["facility_name"] for s in a["plan"]["sites"]}
        sb = {s["facility_name"] for s in b["plan"]["sites"]}
        return {"run_a": run_a, "run_b": run_b, "budget_a_naira": a["budget_naira"], "budget_b_naira": b["budget_naira"],
                "women_a": round(a["plan"]["women_newly_within_target"]), "women_b": round(b["plan"]["women_newly_within_target"]),
                "cost_a_naira": a["plan"]["cost_naira"], "cost_b_naira": b["plan"]["cost_naira"],
                "only_in_a": sorted(sa - sb), "only_in_b": sorted(sb - sa), "in_both": sorted(sa & sb)}

    def facility_lookup(self, query: str, limit: int = 5) -> dict:
        fac = pd.read_parquet(self.store.facilities)
        fac = fac[fac["in_state"]]
        hit = fac[fac["facility_name"].str.contains(query, case=False, regex=False)].head(limit)
        rows = []
        for _, r in hit.iterrows():
            rows.append({"facility_name": r["facility_name"], "type": r["facility_type"], "level": r["facility_level"],
                         "ownership": r["facility_ownership"], "functional_status": r["functional"] or "blank",
                         "lga": r["lga_standard"], "ward": r["ward_standard"],
                         "counts_as_capable_in": [s for s in scenarios.names() if r.get(f"cap_{s}")],
                         "minutes_to_nearest_capable_base_mixed": None if pd.isna(r.get("tt_base_mixed")) else round(float(r["tt_base_mixed"]))})
        return {"query": query, "matches": rows, "note": "Name search within this state only."}

    def build_brief(self, run_id: str | None = None) -> dict:
        from . import brief

        out = brief.build(self.state, run_id=run_id)
        return {"pdf": out["pdf"], "plan_run_id": out["plan"]}


# ----------------------------------------------------------------------------- agent

def _log(store: StateStore, record: dict) -> None:
    with open(store.assistant_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def answer(state: str, question: str, history: list[dict] | None = None, client: LLMClient | None = None) -> dict:
    """Answer one question. Returns answer text, verification, tool calls and any error."""
    client = client or LLMClient.from_settings()
    if client is None:
        return {"answer": None, "error": "The assistant is off: no API key is set (REACH120_LLM_API_KEY or GROQ_API_KEY).",
                "verified": False, "tool_calls": []}
    cfg = settings()["assistant"]
    box = ToolBox(state)
    messages = [{"role": "system", "content": SYSTEM.format(state=state)}, *(history or []), {"role": "user", "content": question}]
    sources, calls, retried, used = [question], [], False, {}
    t0 = time.time()
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "state": state, "question": question, "model": client.model}
    try:
        for _ in range(cfg["max_tool_calls"] + 3):
            budget_left = len(calls) < cfg["max_tool_calls"]
            resp = client.chat(messages, box.schemas, tool_choice="auto" if budget_left else "none",
                               temperature=cfg.get("temperature", 0))
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            if tool_calls and budget_left:
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "")
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments") or "{}")
                        result = box.call(name, args)
                    except ValueError:
                        args, result = None, {"error": "arguments were not valid JSON"}
                    content = json.dumps(result, ensure_ascii=False, default=str)
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": content})
                    sources.append(content)
                    calls.append({"tool": name, "arguments": args, "ok": "error" not in result})
                    if isinstance(result, dict) and result.get("scenario"):
                        used = {"scenario": result.get("scenario_label"), "mode": result.get("mode")}
                continue
            text = (msg.get("content") or "").strip()
            check = numbercheck.verify(text, sources)
            if check.ok:
                footer = []
                if used:
                    footer.append(f"Scenario: {used['scenario']}; transport: "
                                  f"{'vehicles on passable roads' if used['mode'] == 'mixed' else 'walking only'}.")
                if any(c["tool"] == "rank_upgrades" for c in calls) and load("costs").get("placeholder", True):
                    footer.append("Upgrade costs are placeholders.")
                out = {"answer": text + ("\n\n_" + " ".join(footer) + "_" if footer else ""), "verified": True,
                       "numbers_checked": check.checked, "tool_calls": calls, "error": None}
                break
            if not retried:
                retried = True
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "These numbers in your answer do not appear in any tool result: "
                                 + ", ".join(check.unverified) + ". Rewrite the answer using only numbers from the tool "
                                 "results, or call a tool to get them."})
                continue
            out = {"answer": "I could not verify that answer against Reach120's results, so I won't show it. "
                             "Try asking more specifically, or use the Map and Planner pages.",
                   "verified": False, "unverified": check.unverified, "tool_calls": calls, "error": None}
            break
        else:
            out = {"answer": None, "verified": False, "tool_calls": calls, "error": "the assistant did not finish its answer"}
    except LLMError as e:
        out = {"answer": None, "verified": False, "tool_calls": calls, "error": str(e)}
    out["seconds"] = round(time.time() - t0, 1)
    _log(box.store, {**record, **out})
    return out
