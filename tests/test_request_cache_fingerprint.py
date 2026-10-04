from lm_eval.api.task import _request_cache_config_hash
from lm_eval.config.task import TaskConfig


def _fewshot_prompt_a(doc):
    return f"A: {doc['question']}"


def _fewshot_prompt_b(doc):
    return f"B: {doc['question']}"


def test_request_cache_config_hash_is_order_stable():
    left = TaskConfig(
        task="cache_test",
        dataset_kwargs={"revision": "abc", "data_files": {"test": "a.json"}},
        doc_to_text="Question: {{question}}",
    )
    right = TaskConfig(
        task="cache_test",
        dataset_kwargs={"data_files": {"test": "a.json"}, "revision": "abc"},
        doc_to_text="Question: {{question}}",
    )

    assert _request_cache_config_hash(left) == _request_cache_config_hash(right)


def test_request_cache_config_hash_tracks_request_semantics():
    base = TaskConfig(
        task="cache_test",
        dataset_path="json",
        dataset_kwargs={"revision": "abc"},
        doc_to_text="Question: {{question}}",
    )
    changed_prompt = TaskConfig(
        task="cache_test",
        dataset_path="json",
        dataset_kwargs={"revision": "abc"},
        doc_to_text="Prompt: {{question}}",
    )
    changed_revision = TaskConfig(
        task="cache_test",
        dataset_path="json",
        dataset_kwargs={"revision": "def"},
        doc_to_text="Question: {{question}}",
    )

    assert _request_cache_config_hash(base) != _request_cache_config_hash(
        changed_prompt
    )
    assert _request_cache_config_hash(base) != _request_cache_config_hash(
        changed_revision
    )


def test_request_cache_config_hash_serializes_nested_fewshot_callables():
    first = TaskConfig(
        task="cache_test",
        fewshot_config={"doc_to_text": _fewshot_prompt_a},
    )
    same = TaskConfig(
        task="cache_test",
        fewshot_config={"doc_to_text": _fewshot_prompt_a},
    )
    changed = TaskConfig(
        task="cache_test",
        fewshot_config={"doc_to_text": _fewshot_prompt_b},
    )

    assert _request_cache_config_hash(first) == _request_cache_config_hash(same)
    assert _request_cache_config_hash(first) != _request_cache_config_hash(changed)
