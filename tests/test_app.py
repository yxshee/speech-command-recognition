from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(APP), default_timeout=180).run()


def test_app_starts_without_errors(app):
    assert not app.exception
    assert app.title[0].value == "Speech Command Recognition"
    assert len(app.metric) <= 1          # only the sidebar accuracy, no prediction yet


def test_there_is_one_button_per_sample_clip(app):
    keys = sorted(button.key for button in app.button)
    assert keys == ["sample_left", "sample_stop", "sample_yes"]


@pytest.mark.parametrize("word", ["yes", "stop", "left"])
def test_clicking_a_sample_shows_that_word(word):
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    app.button(key=f"sample_{word}").click().run()
    assert not app.exception
    assert word.upper() in [metric.value for metric in app.metric]
