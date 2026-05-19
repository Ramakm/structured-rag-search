from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import tiktoken

import config

_enc = tiktoken.get_encoding("cl100k_base")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    model: str


class OpenRouterClient:
    def __init__(self, api_key: str = config.OPENROUTER_API_KEY, model: str = config.LLM_MODEL):
        self.api_key = api_key
        self.model = model
        self._http = httpx.Client(timeout=120.0)

    def _count_tokens(self, text: str) -> int:
        return len(_enc.encode(text))

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * config.INPUT_COST_PER_1M_USD / 1_000_000
            + output_tokens * config.OUTPUT_COST_PER_1M_USD / 1_000_000
        )

    def chat(self, system: str, user: str) -> LLMResponse:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        input_tokens = sum(self._count_tokens(m["content"]) for m in messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Ramakm/structured-rag-search",
            "X-Title": "structured-rag-search benchmark",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
        }

        t0 = time.perf_counter()
        response = self._http.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Prefer API-reported token counts; fall back to tiktoken
        usage = data.get("usage", {})
        actual_input = usage.get("prompt_tokens", input_tokens)
        output_tokens = usage.get("completion_tokens", self._count_tokens(content))
        total_tokens = actual_input + output_tokens

        return LLMResponse(
            content=content,
            input_tokens=actual_input,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=self._compute_cost(actual_input, output_tokens),
            latency_ms=latency_ms,
            model=self.model,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
