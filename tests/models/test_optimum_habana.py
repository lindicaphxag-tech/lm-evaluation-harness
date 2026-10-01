from unittest import mock

import pytest

from lm_eval.models.huggingface import HFLM
from lm_eval.models.optimum_habana import HabanaLM


def _bare_habana_model() -> HabanaLM:
    model = object.__new__(HabanaLM)
    model._max_length = None
    model.buckets = [16, 32, 128]
    return model


def test_generate_until_restores_max_length_on_success():
    model = _bare_habana_model()

    with mock.patch.object(HFLM, "max_length", new=property(lambda self: 2048)):
        with mock.patch.object(HFLM, "generate_until", return_value=["ok"]) as parent_generate:
            result = model.generate_until([], disable_tqdm=True)

    assert result == ["ok"]
    parent_generate.assert_called_once_with([], True)
    assert model.max_length == 128


def test_generate_until_restores_max_length_on_error():
    model = _bare_habana_model()

    with mock.patch.object(HFLM, "max_length", new=property(lambda self: 2048)):
        with mock.patch.object(
            HFLM, "generate_until", side_effect=RuntimeError("generation failed")
        ):
            with pytest.raises(RuntimeError, match="generation failed"):
                model.generate_until([])

    assert model.max_length == 128
