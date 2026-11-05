from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_runs_selected_simulation():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception
    app.sidebar.slider[1].set_value(600)
    app.sidebar.slider[2].set_value(20)
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 9
    assert any("Results" in item.value for item in app.subheader)
    assert len(app.dataframe) == 3
