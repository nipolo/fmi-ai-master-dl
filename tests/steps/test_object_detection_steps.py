"""Step definitions binding the object_detection.feature scenarios (BDD).

Uses pytest-bdd. The Given/When/Then steps drive the same app-facing
inference helpers the Streamlit UI calls, with a FakeDetector standing in for
the neural network so the behaviour is verified without model weights.
"""

import numpy as np
import pytest
from PIL import Image
from pytest_bdd import given, parsers, scenarios, then, when

from objdetect.app.inference import run_detection
from objdetect.models.base import Detection
from tests.conftest import FakeDetector

# Bind every scenario in the feature file.
scenarios("../features/object_detection.feature")


@pytest.fixture
def context() -> dict:
    """Mutable bag carrying state between Given/When/Then steps."""
    return {}


def _blank_image() -> Image.Image:
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")


@given(
    parsers.parse(
        "an image with a dog at {dog_score:f} and a person at {person_score:f} confidence"
    )
)
def _image_with_two_objects(context, dog_score, person_score):
    context["image"] = _blank_image()
    context["detector"] = FakeDetector(
        [
            Detection(label="dog", score=dog_score, box=(10, 10, 50, 60)),
            Detection(label="person", score=person_score, box=(20, 5, 40, 60)),
        ]
    )


@when(parsers.parse("I run detection with a confidence threshold of {threshold:f}"))
def _run_detection(context, threshold):
    context["detections"] = run_detection(
        context["detector"], context["image"], score_threshold=threshold
    )


@then(parsers.parse("{count:d} objects are detected"))
@then(parsers.parse("{count:d} object is detected"))
def _assert_count(context, count):
    assert len(context["detections"]) == count


@then(parsers.parse('the most confident detection is a "{label}"'))
def _assert_top_label(context, label):
    assert context["detections"][0].label == label
