"""Check that every number in an answer appears in the tool results it was built from.

A number in the answer passes when some number in the tool results (or in the question)
rounds to it at the precision shown. "12.3%" passes against 0.1234 or 12.34; "1,250" against
1249.6; "₦2.00 billion" against 2000000000. Numbered-list markers are ignored.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
    r"(\s*(?:%|percent\b|per cent\b))?"
    r"(\s*(?:billion|bn|million|thousand)\b)?",
    re.IGNORECASE,
)
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "thousand": 1e3}


@dataclass
class Found:
    text: str
    value: float
    tolerance: float
    percent: bool


@dataclass
class Result:
    ok: bool
    checked: int
    unverified: list[str] = field(default_factory=list)


def extract(text: str) -> list[Found]:
    text = _LIST_MARKER.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(text):
        whole, frac, pct, scale = m.groups()
        value = float(whole.replace(",", "") + (frac or ""))
        decimals = len(frac) - 1 if frac else 0
        mult = _SCALE[scale.strip().lower()] if scale else 1.0
        out.append(Found(m.group(0).strip(), value * mult, 0.5 * 10 ** (-decimals) * mult + 1e-9, bool(pct)))
    return out


def allowed_numbers(sources) -> list[float]:
    vals: list[float] = []

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            vals.append(float(o))
        elif isinstance(o, str):
            vals.extend(f.value for f in extract(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    for src in sources:
        if isinstance(src, str):
            try:
                walk(json.loads(src))
                continue
            except ValueError:
                pass
        walk(src)
    extra = [v * 100 for v in vals if -1.0 <= v <= 1.0]       # shares quoted as percentages
    return vals + extra


def verify(answer: str, sources) -> Result:
    allowed = allowed_numbers(sources)
    found = extract(answer)
    bad = [f.text for f in found if not any(abs(a - f.value) <= f.tolerance for a in allowed)]
    return Result(ok=not bad, checked=len(found), unverified=bad)
