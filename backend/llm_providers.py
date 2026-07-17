from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

from backend.token_meter import TokenMeter


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_usage: dict[str, Any]


PROVIDER_MODELS = {
    "groq": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "gemini": [
        "gemini-2.5-flash", 
        "gemini-2.5-pro", 
        "gemini-2.0-flash", 
        "gemini-3.5-flash",
        "gemini-3.1-pro-preview",
        "antigravity-preview-05-2026"
    ],
}


def configured(provider: str) -> bool:
    keys = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    variable = keys.get(provider.strip().lower())
    return bool(variable and os.getenv(variable, "").strip())


def _openai_compatible(
    provider: str,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    timeout_seconds: int,
) -> ProviderResponse:
    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "max_completion_tokens": max_completion_tokens,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" Response: {exc.response.text[:500]}"
        raise ProviderError(f"{provider} request failed: {exc}.{detail}") from exc

    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"{provider} returned an unexpected response shape.") from exc
    usage = payload.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or prompt + completion)
    return ProviderResponse(provider, model, str(text), prompt, completion, total, dict(usage))



def _xai_responses(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    timeout_seconds: int,
) -> ProviderResponse:
    try:
        response = requests.post(
            "https://api.x.ai/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": messages,
                "temperature": 0.2,
                "max_output_tokens": max_completion_tokens,
                "store": False,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" Response: {exc.response.text[:500]}"
        raise ProviderError(f"Grok/xAI request failed: {exc}.{detail}") from exc

    text_parts: list[str] = []
    for item in payload.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") in {"output_text", "text"} and part.get("text"):
                text_parts.append(str(part["text"]))
    if not text_parts and payload.get("output_text"):
        text_parts.append(str(payload["output_text"]))
    if not text_parts:
        raise ProviderError("Grok/xAI returned an unexpected response shape.")

    usage = payload.get("usage") or {}
    prompt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    completion = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or prompt + completion)
    return ProviderResponse("grok", model, "\n".join(text_parts), prompt, completion, total, dict(usage))

def _gemini(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    timeout_seconds: int,
) -> ProviderResponse:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        text = message.get("content", "")
        if role == "system":
            system_parts.append(text)
        else:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]})

    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_completion_tokens},
    }
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        response = requests.post(endpoint, params={"key": api_key}, json=body, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" Response: {exc.response.text[:500]}"
        raise ProviderError(f"Gemini request failed: {exc}.{detail}") from exc

    try:
        parts = payload["candidates"][0]["content"]["parts"]
        text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("Gemini returned an unexpected response shape.") from exc
    usage = payload.get("usageMetadata") or {}
    prompt = int(usage.get("promptTokenCount") or 0)
    completion = int(usage.get("candidatesTokenCount") or 0)
    total = int(usage.get("totalTokenCount") or prompt + completion)
    return ProviderResponse("gemini", model, text, prompt, completion, total, dict(usage))


def generate(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    meter: TokenMeter,
    max_completion_tokens: int = 700,
    timeout_seconds: int = 60,
) -> ProviderResponse:
    normalized = provider.strip().lower()
    if normalized == "grok":
        api_key = os.getenv("XAI_API_KEY", "").strip()
        if not api_key:
            raise ProviderError("XAI_API_KEY is not configured.")
        result = _xai_responses(
            api_key,
            model,
            messages,
            max_completion_tokens,
            timeout_seconds,
        )
    elif normalized == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ProviderError("GROQ_API_KEY is not configured.")
        result = _openai_compatible(
            "groq",
            "https://api.groq.com/openai/v1/chat/completions",
            api_key,
            model,
            messages,
            max_completion_tokens,
            timeout_seconds,
        )
    elif normalized == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not configured.")
        result = _gemini(api_key, model, messages, max_completion_tokens, timeout_seconds)
    else:
        raise ProviderError(f"Unsupported provider: {provider}")

    meter.add(result.provider, result.prompt_tokens, result.completion_tokens, result.total_tokens)
    return result
