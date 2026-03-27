"""Claude/Anthropic LLM provider."""

from __future__ import annotations

import anthropic

from veritas.providers.base import LLMProvider


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str, system: str = "") -> str:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system if system else "You are a verification agent.",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
