"""Web search providers for source verification."""

from __future__ import annotations

import httpx

from veritas.providers.base import SearchProvider, SearchResult


class BraveSearchProvider(SearchProvider):
    def __init__(self, api_key: str, num_results: int = 5):
        self.api_key = api_key
        self.default_num_results = num_results

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": num_results},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                },
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", ""),
                    )
                )
            return results


class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str, num_results: int = 5):
        self.api_key = api_key
        self.default_num_results = num_results

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": num_results,
                },
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("results", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                    )
                )
            return results
