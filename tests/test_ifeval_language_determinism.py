import pytest


langdetect = pytest.importorskip("langdetect")


def test_ifeval_language_detection_is_deterministic():
    from lm_eval.tasks.ifeval.utils import process_results

    doc = {
        "key": 0,
        "prompt": "Respond in English.",
        "instruction_id_list": ["language:response_language"],
        "kwargs": [{"language": "en"}],
    }
    response = ["Hello world"]

    results = [process_results(doc, response) for _ in range(50)]

    assert langdetect.DetectorFactory.seed == 0
    assert all(result == results[0] for result in results[1:])


@pytest.mark.parametrize(
    "module_name",
    [
        "lm_eval.tasks.ifeval.instructions",
        "lm_eval.tasks.leaderboard.ifeval.instructions",
    ],
)
def test_ifeval_language_modules_pin_detector_seed(module_name):
    import importlib

    langdetect.DetectorFactory.seed = None
    importlib.reload(importlib.import_module(module_name))
    assert langdetect.DetectorFactory.seed == 0
