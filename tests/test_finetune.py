"""The transformer detector is imported lazily, so the rest of the benchmark and
CI work without torch. Training itself is exercised by the benchmark run."""

from src.detectors import is_zero_shot, make_detector


def test_transformer_is_trained_and_lazy():
    det = make_detector("transformer")
    assert det.name == "transformer" and not is_zero_shot("transformer")
    assert not hasattr(det, "model_")          # nothing loaded before fit


def test_cascade_is_trained_and_lazy():
    det = make_detector("cascade")
    assert det.fast is None and det.judge is None
