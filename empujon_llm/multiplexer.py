"""LLM Multiplexer — unified interface for OpenAI, Gemini, and Bedrock."""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Union

from empujon_llm.types import (
    LLMException, LLMMessage, LLMProvider, LLMRequest, LLMResponse,
    ModelNotSupportedException, ProviderNotAvailableException,
)
from empujon_llm.providers import LLMProviderBase

logger = logging.getLogger(__name__)


class LLMMultiplexer:
    """Main multiplexer class for unified LLM access.

    Usage::

        from empujon_llm import LLMMultiplexer

        llm = LLMMultiplexer()
        response = llm.chat("gemini-2.0-flash", system="...", user="...")
        print(response.content)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        bedrock_region: Optional[str] = None,
    ):
        self.providers: Dict[LLMProvider, LLMProviderBase] = {}

        # Initialize providers (fail silently if deps missing)
        try:
            from empujon_llm.providers.openai_provider import OpenAIProvider
            self.providers[LLMProvider.OPENAI] = OpenAIProvider(openai_api_key)
            logger.info("OpenAI provider initialized")
        except Exception as e:
            logger.warning("Failed to initialize OpenAI provider: %s", e)

        try:
            from empujon_llm.providers.gemini_provider import GeminiProvider
            self.providers[LLMProvider.GEMINI] = GeminiProvider(gemini_api_key)
            logger.info("Gemini provider initialized")
        except Exception as e:
            logger.warning("Failed to initialize Gemini provider: %s", e)

        try:
            from empujon_llm.providers.bedrock_provider import BedrockProvider
            self.providers[LLMProvider.BEDROCK] = BedrockProvider(bedrock_region)
            logger.info("Bedrock provider initialized")
        except Exception as e:
            logger.warning("Failed to initialize Bedrock provider: %s", e)

    # ── provider resolution ──

    def _detect_provider(self, model: str) -> LLMProvider:
        for provider_enum, provider in self.providers.items():
            if provider.supports_model(model):
                return provider_enum
        raise ModelNotSupportedException(
            f"Model '{model}' not supported by any available provider",
            "unknown", model,
        )

    def _get_provider(self, provider: LLMProvider, model: str) -> LLMProviderBase:
        if provider == LLMProvider.AUTO:
            provider = self._detect_provider(model)
        if provider not in self.providers:
            raise ProviderNotAvailableException(
                f"Provider '{provider.value}' not available",
                provider.value, model,
            )
        logger.info("Selected LLM provider '%s' for model '%s'", provider.value, model)
        return self.providers[provider]

    # ── message normalization ──

    @staticmethod
    def _normalize_messages(
        messages: Optional[Union[str, List[Union[Dict[str, str], LLMMessage]]]] = None,
        system: Optional[str] = None,
        user: Optional[str] = None,
    ) -> List[LLMMessage]:
        """Accept either messages list, or system+user shorthand, or both."""
        if messages is not None:
            if isinstance(messages, str):
                return [LLMMessage(role="user", content=messages)]
            out = []
            for msg in messages:
                if isinstance(msg, dict):
                    out.append(LLMMessage(role=msg["role"], content=msg["content"]))
                elif isinstance(msg, LLMMessage):
                    out.append(msg)
            return out

        # Shorthand: system + user
        result = []
        if system:
            result.append(LLMMessage(role="system", content=system))
        if user:
            result.append(LLMMessage(role="user", content=user))
        return result

    # ── async API ──

    async def chat_async(
        self,
        model: str,
        messages: Optional[Union[str, List[Union[Dict[str, str], LLMMessage]]]] = None,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        provider: LLMProvider = LLMProvider.AUTO,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> LLMResponse:
        """Generate chat completion asynchronously."""
        standardized = self._normalize_messages(messages, system, user)
        request = LLMRequest(
            model=model,
            messages=standardized,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            provider=provider,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        provider_instance = self._get_provider(provider, model)
        return await provider_instance.chat_async(request)

    async def chat_async_with_fallback(
        self,
        model: str,
        messages: Optional[Union[str, List[Union[Dict[str, str], LLMMessage]]]] = None,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        provider: LLMProvider = LLMProvider.AUTO,
        **kwargs,
    ) -> LLMResponse:
        """Chat with JSON validation when response_format is specified."""
        resp = await self.chat_async(
            model=model, messages=messages, system=system, user=user,
            provider=provider, **kwargs,
        )
        if kwargs.get("response_format"):
            content = (resp.content or "").strip()
            if not content:
                raise LLMException(f"LLM ({model}) returned empty response for structured request.", "multiplexer", model)
            if not (content.startswith("{") and content.endswith("}")):
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    resp.content = json_match.group(0)
                else:
                    raise LLMException(f"LLM ({model}) response is not valid JSON.", "multiplexer", model)
        return resp

    # ── sync API ──

    def chat(
        self,
        model: str,
        messages: Optional[Union[str, List[Union[Dict[str, str], LLMMessage]]]] = None,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        provider: LLMProvider = LLMProvider.AUTO,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> LLMResponse:
        """Generate chat completion synchronously."""
        standardized = self._normalize_messages(messages, system, user)
        request = LLMRequest(
            model=model,
            messages=standardized,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            provider=provider,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        provider_instance = self._get_provider(provider, model)
        return provider_instance.chat_sync(request)


def create_llm_multiplexer(**kwargs) -> LLMMultiplexer:
    """Create and return a configured LLM multiplexer instance."""
    return LLMMultiplexer(**kwargs)
