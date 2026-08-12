import os

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "aligned_transactions.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH), reason="Run align_datasets.py first to generate the demo data"
)


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Fraud Detection" in t.value for t in at.title)


def test_app_shows_prompt_before_any_fetch():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Click the button above" in info.value for info in at.info)


def test_app_unbiased_fetch_runs_without_exception():
    """Uncheck the fraud bias so we sample the general (mostly legit)
    pool -- this exercises the fast path deterministically without
    necessarily invoking the (costly, slow) LLM slow path."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.checkbox[0].uncheck().run()
    at.button[0].click().run()
    assert not at.exception
    # Either the metric or the flagged/approved message should be present
    assert len(at.metric) > 0


def test_app_displays_risk_score_metric_after_fetch():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.checkbox[0].uncheck().run()
    at.button[0].click().run()
    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    assert "XGBoost Risk Score" in metric_labels
