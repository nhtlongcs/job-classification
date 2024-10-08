import pytest
from tools.search import DDGSWrapper


@pytest.fixture
def wrapper():
    return DDGSWrapper()


def test_search_with_valid_description(wrapper):
    job_description = "a job for old people"
    search_results = wrapper.search(job_description)
    assert search_results is not None
    assert isinstance(search_results, list)


def test_search_with_special_characters(wrapper):
    job_description = "@#$%^&*()!"
    search_results = wrapper.search(job_description)
    assert search_results is not None
    assert isinstance(search_results, list)


def test_search_with_long_description(wrapper):
    job_description = "a" * 100
    search_results = wrapper.search(job_description)
    assert search_results is not None
    assert isinstance(search_results, list)
