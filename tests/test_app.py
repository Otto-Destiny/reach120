"""Every app page runs without an exception on the results on disk (skipped when there are none)."""
from pathlib import Path

import pytest

from reach120.store import available_states

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["views/overview.py", "views/map.py", "views/planner.py", "views/ask.py", "views/method.py"]

pytestmark = pytest.mark.skipif(not available_states(), reason="no pipeline results on disk")


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    for var in ("REACH120_LLM_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if page != PAGES[0]:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.message for e in at.exception]
