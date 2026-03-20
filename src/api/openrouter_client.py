"""Unified async OpenRouter client. All LLM calls route through this file."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from dotenv import load_dotenv

_RETRYABLE_ERRORS = (aiohttp.ClientError, OSError, ValueError, TimeoutError)

load_dotenv()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY not found in .env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMResponse:
    """Structured response from a single LLM API call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    success: bool
    error: Optional[str] = field(default=None)


class OpenRouterClient:
    """Async client for OpenRouter API with session reuse, retry, and exponential backoff."""

    def __init__(self, api_key: str = OPENROUTER_API_KEY, max_retries: int = 3, timeout: int = 60):
        if not api_key:
            raise ValueError("api_key must not be empty")
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return shared session with connection pool limits, creating one if needed."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        """Close the shared session. Call when done with all requests."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Send a single async chat completion request with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                async with session.post(
                    OPENROUTER_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status == 429:  # rate limit — backoff longer
                        body = await resp.text()
                        last_error = f"HTTP 429: {body[:200]}"
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(5 * (2 ** attempt))
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        raise ValueError(f"HTTP {resp.status}: {body[:200]}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return LLMResponse(
                        content=content,
                        model=model,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        success=True,
                    )
            except asyncio.CancelledError:
                raise  # never retry cancellation — propagate immediately
            except _RETRYABLE_ERRORS as exc:
                last_error = str(exc)
                # do NOT close shared session — would kill all concurrent in-flight requests
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # exponential backoff

        return LLMResponse(
            content="",
            model=model,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error=last_error,
        )
