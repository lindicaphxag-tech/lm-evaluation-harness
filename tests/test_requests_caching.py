import importlib
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import pytest
import torch

from lm_eval.api.task import _request_cache_config_hash
from lm_eval.caching.cache import PATH
from lm_eval.config.task import TaskConfig


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

# NOTE the script this loads uses simple evaluate
# TODO potentially test both the helper script and the normal script
sys.path.append(f"{MODULE_DIR}/../scripts")
model_loader = importlib.import_module("requests_caching")
run_model_for_task_caching = model_loader.run_model_for_task_caching

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
DEFAULT_TASKS = ["lambada_openai", "sciq"]



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

    assert _request_cache_config_hash(base) != _request_cache_config_hash(changed_prompt)
    assert _request_cache_config_hash(base) != _request_cache_config_hash(changed_revision)


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


@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup
    torch.use_deterministic_algorithms(False)
    clear_cache()
    # Yields control back to the test function
    yield
    # Cleanup here


def clear_cache():
    if os.path.exists(PATH):
        cache_files = os.listdir(PATH)
        for file in cache_files:
            file_path = f"{PATH}/{file}"
            os.unlink(file_path)


# leaving tasks here to allow for the option to select specific task files
def get_cache_files(tasks: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    cache_files = os.listdir(PATH)

    file_task_names = []

    for file in cache_files:
        file_without_prefix = file.split("-")[1]
        file_without_prefix_and_suffix = file_without_prefix.split(".")[0]
        file_task_names.extend([file_without_prefix_and_suffix])

    return cache_files, file_task_names


def assert_created(tasks: List[str], file_task_names: List[str]):
    tasks.sort()
    file_task_names.sort()

    assert tasks == file_task_names


@pytest.mark.parametrize("tasks", [DEFAULT_TASKS])
def requests_caching_true(tasks: List[str]):
    run_model_for_task_caching(tasks=tasks, cache_requests="true")

    cache_files, file_task_names = get_cache_files()
    print(file_task_names)
    assert_created(tasks=tasks, file_task_names=file_task_names)


@pytest.mark.parametrize("tasks", [DEFAULT_TASKS])
def requests_caching_refresh(tasks: List[str]):
    run_model_for_task_caching(tasks=tasks, cache_requests="true")

    timestamp_before_test = datetime.now().timestamp()

    run_model_for_task_caching(tasks=tasks, cache_requests="refresh")

    cache_files, file_task_names = get_cache_files()

    for file in cache_files:
        modification_time = os.path.getmtime(f"{PATH}/{file}")
        assert modification_time > timestamp_before_test

    tasks.sort()
    file_task_names.sort()

    assert tasks == file_task_names


@pytest.mark.parametrize("tasks", [DEFAULT_TASKS])
def requests_caching_delete(tasks: List[str]):
    # populate the data first, rerun this test within this test for additional confidence
    # test_requests_caching_true(tasks=tasks)

    run_model_for_task_caching(tasks=tasks, cache_requests="delete")

    cache_files, file_task_names = get_cache_files()

    assert len(cache_files) == 0


# useful for locally running tests through the debugger
if __name__ == "__main__":

    def run_tests():
        tests = [
            # test_requests_caching_true,
            # test_requests_caching_refresh,
            # test_requests_caching_delete,
        ]
        # Lookups of global names within a loop is inefficient, so copy to a local variable outside of the loop first
        default_tasks = DEFAULT_TASKS
        for test_func in tests:
            clear_cache()
            test_func(tasks=default_tasks)

        print("Tests pass")

    run_tests()
