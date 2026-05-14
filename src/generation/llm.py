"""LLM abstraction.

Providers:
  - stub: deterministic, offline, used in CI and smoke tests
  - ollama: local GPU runtime dependency
  - openai / anthropic: optional cloud providers

The MLOps pipeline deploys the FastAPI RAG app. Ollama is tracked as an
external runtime dependency, not as a trained/deployed artefact.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests

from src.core.config import settings
from src.generation.prompt import UNSUPPORTED_ANSWER
from src.retrieval.base import RetrievalResult


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        ...


class StubLLM(LLMClient):
    """Deterministic, extractive 'LLM' for testing."""

    def __init__(self, query: str, contexts: list[RetrievalResult]):
        self.query = query
        self.contexts = contexts

    def generate(self, system: str, user: str) -> str:
        from src.indexing.bm25 import tokenize

        if not self.contexts:
            return UNSUPPORTED_ANSWER
        q_tokens = set(tokenize(self.query))
        scored = []
        for i, r in enumerate(self.contexts):
            for sent in r.chunk.text.split(". "):
                s_tokens = set(tokenize(sent))
                overlap = len(q_tokens & s_tokens)
                if overlap > 0:
                    scored.append((overlap, i + 1, sent.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return UNSUPPORTED_ANSWER
        chosen = scored[:3]
        return " ".join(f"{s.rstrip('.')} [{i}]." for _, i, s in chosen)


class OllamaLLM(LLMClient):
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or settings.llm_model
        self.host = (host or settings.ollama_host).rstrip("/")

    def generate(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": settings.llm_temperature,
                    "top_p": settings.llm_top_p,
                    "num_predict": settings.llm_max_tokens,
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


class OpenAILLM(LLMClient):
    def __init__(self, model: str | None = None):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model or settings.llm_model

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=settings.llm_max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


class AnthropicLLM(LLMClient):
    def __init__(self, model: str | None = None):
        import anthropic  # type: ignore

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model or "claude-3-5-haiku-20241022"

    def generate(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=settings.llm_max_tokens,
        )
        return resp.content[0].text


def make_llm(query: str, contexts: list[RetrievalResult]) -> LLMClient:
    provider = settings.llm_provider
    if provider == "ollama":
        return OllamaLLM()
    if provider == "openai":
        return OpenAILLM()
    if provider == "anthropic":
        return AnthropicLLM()
    return StubLLM(query, contexts)
