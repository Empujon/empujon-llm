"""Shared types for the LLM multiplexer."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    BEDROCK = "bedrock"
    GEMINI = "gemini"
    AUTO = "auto"


@dataclass
class LLMMessage:
    """Standardized message format."""
    role: str  # "system", "user", "assistant", "developer"
    content: str


@dataclass
class LLMRequest:
    model: str
    messages: List[LLMMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    provider: LLMProvider = LLMProvider.AUTO
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"


@dataclass
class LLMResponse:
    """Standardized response format."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None


class LLMException(Exception):
    """Base exception for LLM operations."""
    def __init__(self, message: str, provider: str, model: str):
        self.message = message
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}:{model}] {message}")


class ModelNotSupportedException(LLMException):
    """Raised when model is not supported by any provider."""
    pass


class ProviderNotAvailableException(LLMException):
    """Raised when provider dependencies are not installed."""
    pass
