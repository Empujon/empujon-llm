"""LLM provider implementations."""

from abc import ABC, abstractmethod
from empujon_llm.types import LLMRequest, LLMResponse


class LLMProviderBase(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat_async(self, request: LLMRequest) -> LLMResponse:
        pass

    @abstractmethod
    def chat_sync(self, request: LLMRequest) -> LLMResponse:
        pass

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        pass
