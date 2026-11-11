from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_starts_and_shows_the_uploader_without_errors():
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception
    assert "Speech Command Recognition" in app.title[0].value
