"""Command line: `reach120 <step> --state <name>`. Run `reach120 all --state Kano` for everything."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

from .config import PROJECT_ROOT


def _state(args) -> str:
    from . import inputs

    return str(inputs.state_record(args.state)["statename"])


def cmd_states(args):
    from . import inputs
    from .store import available_states

    done = set(available_states())
    for name in inputs.state_names():
        print(f"{'*' if name in done else ' '} {name}")
    print("\n* = results available")


def cmd_prepare(args):
    from . import prepare

    prepare.run(_state(args))


def cmd_travel_time(args):
    from . import access
    from .prepare import log

    log("travel times")
    access.run(_state(args), args.scenarios, args.modes)


def cmd_coverage(args):
    from . import coverage
    from .prepare import log

    log("coverage tables, maps and summary")
    s = coverage.run(_state(args))
    d = s["runs"][s["default_run"]]
    log(f"  {s['default_run']}: {100 * d['share_beyond_target']:.1f}% of {d['women']:,.0f} women are more than "
        f"{s['target_min']} minutes from care")


def cmd_reach(args):
    from . import planning

    planning.run(_state(args), verify=args.verify, modes=getattr(args, "plan_modes", None))


def cmd_plan(args):
    from . import display, plan
    from .prepare import log

    state = _state(args)
    if args.budget is None:
        log("budget curves and default plans")
        plan.run(state, modes=getattr(args, "plan_modes", None))
        return
    inp = plan.load_inputs(state, args.mode)
    res = plan.run_plan(inp, args.scenario, args.budget)
    p = res["plan"]
    print(f"{res['run_id']}: {len(p['sites'])} sites, {display.format_naira(p['cost_naira'])}, "
          f"{p['women_newly_within_target']:,.0f} more women within {res['target_min']} min")
    for i, s in enumerate(p["sites"], 1):
        print(f"  {i:2}. {s['facility_name']} ({s['lga']}, {s['tier']}) {display.format_naira(s['cost_naira'])}: "
              f"{s['women_newly_within_target']:,.0f} women")


def cmd_scorecard(args):
    from . import scorecard
    from .prepare import log

    log("scoring the capability stand-in against OnTIME's verified list")
    res = scorecard.run()
    for r in res["scenarios"]:
        f1 = f"{r['balance_f1']:.2f}" if r["balance_f1"] is not None else "n/a"
        print(f"  {r['label']:20} catch {100 * (r['catch_rate'] or 0):5.1f}%  not on list "
              f"{100 * (r['false_inclusion_rate'] or 0):5.1f}%  balance {f1}")
    print(f"  matched {res['verified_matched']} of {res['verified_total']}; recommended: {res['recommended']}")


def cmd_validate(args):
    from . import validate

    validate.run(_state(args), osrm=args.osrm)


def cmd_brief(args):
    from . import brief
    from .prepare import log

    out = brief.build(_state(args), args.scenario, args.run_id)
    log(f"  wrote {out['pdf']}")


def cmd_all(args):
    from .store import national

    t0 = time.time()
    args.budget, args.scenario, args.run_id = None, None, None
    steps = [cmd_prepare, cmd_travel_time, cmd_coverage, cmd_reach, cmd_plan, cmd_validate, cmd_brief]
    if not national("capability_scorecard.json").exists():
        steps.insert(0, cmd_scorecard)  # national and run once: it picks the default capability definition
    for step in steps:
        step(args)
    print(f"finished in {(time.time() - t0) / 60:.1f} minutes")


def cmd_ask(args):
    from . import assistant

    out = assistant.answer(_state(args), args.question)
    if out.get("error"):
        print("error:", out["error"])
    if out.get("answer"):
        print(out["answer"])
    for c in out.get("tool_calls", []):
        print(f"  [tool] {c['tool']} {c['arguments']}")


def cmd_bundle(args):
    """Copy the app, its code and one state's small outputs into a folder ready for Hugging Face Spaces."""
    from .config import outputs_root, slug

    state = _state(args)
    out = PROJECT_ROOT / "deploy" / "reach120-space" if args.out is None else args.out
    out = out.resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(PROJECT_ROOT / "app", out / "app")
    shutil.copytree(PROJECT_ROOT / "src" / "reach120", out / "src" / "reach120",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROJECT_ROOT / "config", out / "config")
    if (PROJECT_ROOT / ".streamlit").exists():
        shutil.copytree(PROJECT_ROOT / ".streamlit", out / ".streamlit")
    src_out = outputs_root()
    skip = {"friction_mixed.tif", "friction_walk.tif", "women_15_49.tif", "population_total.tif", "ward_ids.tif"}
    for d in [src_out / slug(state), src_out / "_national"]:
        if d.exists():
            shutil.copytree(d, out / "outputs" / d.name, ignore=lambda folder, names: [
                n for n in names if n in skip or n.startswith("tt30_") or n.startswith("ttpop_") or n.endswith(".tmp")])
    (out / "requirements.txt").write_text("\n".join([
        "streamlit>=1.40", "pandas>=2.1", "numpy>=1.26", "pyarrow>=14", "scipy>=1.11", "ortools>=9.8",
        "matplotlib>=3.8", "pyyaml>=6", "requests>=2.31", "pydeck>=0.9", "altair>=5"]) + "\n", encoding="utf-8")
    (out / "Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY . .\nENV REACH120_OUTPUTS=/app/outputs PYTHONPATH=/app/src\nEXPOSE 8501\n"
        'CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]\n', encoding="utf-8")
    (out / "README.md").write_text(
        "---\ntitle: Reach120\nemoji: 🏥\ncolorFrom: blue\ncolorTo: gray\nsdk: docker\napp_port: 8501\npinned: false\n"
        "license: cc-by-sa-4.0\n---\n\nReach120: two-hour access to emergency obstetric care, and hospital upgrades "
        f"ranked within a budget. Results for {state}. Set the `REACH120_LLM_API_KEY` secret to switch the assistant on.\n",
        encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"bundle written to {out} ({size / 1e6:.1f} MB)")


def cmd_app(args):
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app" / "streamlit_app.py")], check=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="reach120", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, state=True):
        sp = sub.add_parser(name, help=help_)
        if state:
            sp.add_argument("--state", required=True, help="state name, e.g. Kano")
        sp.set_defaults(fn=fn)
        return sp

    add("states", cmd_states, "list states and which have results", state=False)
    add("prepare", cmd_prepare, "cut the state's inputs from the national data")
    tt = add("travel-time", cmd_travel_time, "travel time to the nearest capable facility")
    tt.add_argument("--scenarios", nargs="*")
    tt.add_argument("--modes", nargs="*")
    add("coverage", cmd_coverage, "women within 30/60/120 minutes by ward, LGA and state")
    r = add("reach", cmd_reach, "demand points, candidate upgrades and what each would reach")
    r.add_argument("--verify", type=int, default=3, help="candidates to re-run without pruning as a check")
    r.add_argument("--modes", dest="plan_modes", nargs="*", help="transport modes to plan for (default: settings)")
    pl = add("plan", cmd_plan, "budget curves and plans")
    pl.add_argument("--scenario", default="base")
    pl.add_argument("--mode", default="mixed", choices=["mixed", "walk"])
    pl.add_argument("--budget", type=float, help="naira; omit for the sweep and default plans")
    pl.add_argument("--modes", dest="plan_modes", nargs="*", help="with no --budget: transport modes to sweep")
    add("scorecard", cmd_scorecard, "score capability scenarios against OnTIME's verified list", state=False)
    v = add("validate", cmd_validate, "validation report")
    v.add_argument("--osrm", action="store_true", help="also spot-check against the public OSRM routing server")
    b = add("brief", cmd_brief, "one-page PDF brief")
    b.add_argument("--scenario")
    b.add_argument("--run-id")
    a = add("all", cmd_all, "every step in order")
    a.add_argument("--osrm", action="store_true")
    a.add_argument("--verify", type=int, default=3)
    a.add_argument("--scenarios", nargs="*")
    a.add_argument("--modes", nargs="*")
    q = add("ask", cmd_ask, "ask the assistant a question")
    q.add_argument("question")
    bu = add("bundle", cmd_bundle, "folder ready for Hugging Face Spaces")
    bu.add_argument("--out", type=lambda s: __import__("pathlib").Path(s))
    add("app", cmd_app, "start the app", state=False)
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except FileNotFoundError as e:  # missing data or an earlier step not run: the message says what to do
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
