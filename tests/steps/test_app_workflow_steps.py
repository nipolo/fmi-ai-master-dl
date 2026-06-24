"""Step definitions binding the app_workflow.feature scenarios (BDD)."""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from objdetect.app import inference
from objdetect.app.inference import (
    detections_to_frame,
    load_model,
    select_detections,
    summarize_counts,
)
from objdetect.models.base import Detection

scenarios("../features/app_workflow.feature")


@pytest.fixture
def context() -> dict:
    return {}


@given(parsers.parse('a model "{display_name}" whose weights file is missing'))
def _register_missing_model(context, display_name, monkeypatch, tmp_path):
    missing = str(tmp_path / "not_installed.pt")
    models = dict(inference.AVAILABLE_MODELS)
    models[display_name] = ("yolo", {"weights": missing})
    monkeypatch.setattr(inference, "AVAILABLE_MODELS", models)


@when(parsers.parse('I load a model called "{display_name}"'))
def _load_model(context, display_name):
    try:
        context["model"] = load_model(display_name)
    except (ValueError, FileNotFoundError) as exc:
        context["error"] = exc


@then(parsers.parse('loading fails with an "{message}" error'))
@then(parsers.parse('loading fails with a "{message}" error'))
def _assert_load_error(context, message):
    assert "error" in context, "expected loading to raise, but it succeeded"
    assert message in str(context["error"])


@given("detections of a dog and a cat")
def _two_detections(context):
    context["detections"] = [
        Detection(label="dog", score=0.9, box=(1.0, 2.0, 3.0, 4.0)),
        Detection(label="cat", score=0.7, box=(5.0, 6.0, 7.0, 8.0)),
    ]


@when("I tabulate the detections")
def _tabulate(context):
    context["frame"] = detections_to_frame(context["detections"])


@then(parsers.parse("the table has {rows:d} rows"))
def _assert_rows(context, rows):
    assert len(context["frame"]) == rows


@then(parsers.parse('the table columns are "{columns}"'))
def _assert_columns(context, columns):
    expected = [c.strip() for c in columns.split(",")]
    assert list(context["frame"].columns) == expected


@given(parsers.parse("detections of {first}, {second}, and {third}"))
def _three_detections(context, first, second, third):
    labels = [first, second, third]
    context["detections"] = [
        Detection(label=label, score=0.9, box=(0.0, 0.0, 1.0, 1.0)) for label in labels
    ]


@when("I summarize the detections")
def _summarize(context):
    context["summary"] = summarize_counts(context["detections"])


@then(parsers.parse('the summary reports {count_a:d} "{label_a}" and {count_b:d} "{label_b}"'))
def _assert_summary(context, count_a, label_a, count_b, label_b):
    summary = context["summary"]
    assert summary.get(label_a) == count_a
    assert summary.get(label_b) == count_b


@when(parsers.re(r'I select the table rows "(?P<rows>[^"]*)"'))
def _select_rows(context, rows):
    selected_rows = [int(r.strip()) for r in rows.split(",") if r.strip()]
    context["shown"] = select_detections(context["detections"], selected_rows)


@then(parsers.parse('the shown detections are "{labels}"'))
def _assert_shown(context, labels):
    expected = [label.strip() for label in labels.split(",")]
    actual = [det.label for det in context["shown"]]
    assert actual == expected
