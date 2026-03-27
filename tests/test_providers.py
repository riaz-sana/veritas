"""Tests for LLM and search provider interfaces."""

import pytest

from veritas.providers.base import LLMProvider, SearchProvider, SearchResult
from veritas.providers.claude import ClaudeProvider
from veritas.providers.search import BraveSearchProvider, TavilySearchProvider


def test_search_result_creation():
    sr = SearchResult(
        title="iPhone Wikipedia",
        url="https://en.wikipedia.org/wiki/IPhone",
        snippet="The iPhone was released on June 29, 2007.",
    )
    assert sr.title == "iPhone Wikipedia"
    assert "2007" in sr.snippet


def test_claude_provider_instantiation():
    provider = ClaudeProvider(model="claude-sonnet-4-6", api_key="test-key")
    assert provider.model == "claude-sonnet-4-6"


def test_claude_provider_implements_protocol():
    provider = ClaudeProvider(model="claude-sonnet-4-6", api_key="test-key")
    assert isinstance(provider, LLMProvider)


def test_brave_search_provider_instantiation():
    provider = BraveSearchProvider(api_key="test-key")
    assert isinstance(provider, SearchProvider)


def test_tavily_search_provider_instantiation():
    provider = TavilySearchProvider(api_key="test-key")
    assert isinstance(provider, SearchProvider)


@pytest.mark.asyncio
async def test_claude_provider_generate_requires_real_key():
    provider = ClaudeProvider(model="claude-sonnet-4-6", api_key="fake-key")
    with pytest.raises(Exception):
        await provider.generate("Hello")


@pytest.mark.asyncio
async def test_brave_search_requires_real_key():
    provider = BraveSearchProvider(api_key="fake-key")
    with pytest.raises(Exception):
        await provider.search("test query")
