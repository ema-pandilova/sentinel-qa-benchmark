from __future__ import annotations

import os
from typing import Literal, Any, Optional

from langchain_openai import ChatOpenAI

Provider = Literal["openai", "deepseek", "gemini", "anthropic"]


class _Resp:
    """Unified response wrapper matching LangChain's interface."""

    def __init__(self, text: str, response_metadata: Optional[dict] = None):
        self.content = text
        self.response_metadata = response_metadata


class _SDKChatModel:
    def __init__(self, provider: str, model_name: str, temperature: float = 0.1):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, messages: list[dict]) -> Any:
        prompt = "\n\n".join([m.get("content", "") for m in messages if m.get("content")])

        if self.provider == "gemini":
            return self._invoke_gemini(prompt)
        if self.provider == "anthropic":
            return self._invoke_anthropic(prompt)
        raise ValueError(f"Unsupported SDK provider: {self.provider}")

    def _invoke_gemini(self, prompt: str) -> _Resp:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError(
                "google-generativeai not installed. Install with: pip install -U google-generativeai"
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)

        resp = model.generate_content(
            prompt,
            generation_config={"temperature": float(self.temperature)},
        )

        usage_meta = getattr(resp, "usage_metadata", None)
        token_usage = None
        if usage_meta:
            token_usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", None),
                "completion_tokens": getattr(usage_meta, "candidates_token_count", None),
                "total_tokens": getattr(usage_meta, "total_token_count", None),
            }

        return _Resp(
            (resp.text or "").strip(),
            response_metadata={"token_usage": token_usage} if token_usage else None,
        )

    def _invoke_anthropic(self, prompt: str) -> _Resp:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic not installed. Install with: pip install -U anthropic"
            ) from e

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=float(self.temperature),
            messages=[{"role": "user", "content": prompt}],
        )

        text_chunks = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_chunks.append(block.text)

        usage = getattr(msg, "usage", None)
        token_usage = None
        if usage:
            token_usage = {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": (getattr(usage, "input_tokens", 0) or 0)
                + (getattr(usage, "output_tokens", 0) or 0),
            }

        return _Resp(
            "\n".join(text_chunks).strip(),
            response_metadata={"token_usage": token_usage} if token_usage else None,
        )


def make_chat_model(provider: Provider, model_name: str, temperature: float = 0.1) -> Any:
    provider = provider.lower()

    if provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if provider in ("gemini", "anthropic"):
        return _SDKChatModel(provider=provider, model_name=model_name, temperature=temperature)

    raise ValueError(f"Unsupported provider: {provider}")
