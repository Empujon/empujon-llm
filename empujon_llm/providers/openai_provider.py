"""OpenAI provider — Chat Completions + Responses API."""

import os
from typing import Any, Dict, List, Optional

from empujon_llm.types import (
    LLMException, LLMMessage, LLMRequest, LLMResponse,
    ProviderNotAvailableException,
)
from empujon_llm.providers import LLMProviderBase

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    AsyncOpenAI = OpenAI = None


class OpenAIProvider(LLMProviderBase):
    """OpenAI provider implementation compatible with Chat Completions & Responses API."""

    SUPPORTED_MODELS = [
        "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
        "o1", "o3", "gpt-5", "gpt-5-mini",
    ]

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        if not OpenAI:
            raise ProviderNotAvailableException(
                "OpenAI package not installed. Run: pip install openai",
                "openai", "unknown",
            )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMException("OPENAI_API_KEY not set", "openai", "unknown")
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=self.api_key, timeout=timeout)

    def supports_model(self, model: str) -> bool:
        m = (model or "").lower()
        return any(m.startswith(p) or p in m for p in self.SUPPORTED_MODELS)

    # ── helpers ──

    @staticmethod
    def _to_chat_messages(messages: List[LLMMessage]) -> List[Dict[str, str]]:
        out = []
        for msg in messages:
            role = msg.role if msg.role in ("system", "user", "assistant") else "user"
            out.append({"role": role, "content": msg.content})
        return out

    @staticmethod
    def _to_responses_input(messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        out = []
        for msg in messages:
            role = msg.role if msg.role in ("system", "user", "assistant", "developer") else "user"
            out.append({"role": role, "content": [{"type": "input_text", "text": msg.content}]})
        return out

    @staticmethod
    def _normalize_text_format_for_responses(fmt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not fmt or not isinstance(fmt, dict):
            return None
        if "format" in fmt and isinstance(fmt["format"], dict):
            f = fmt["format"]
            if f.get("type") == "json_schema":
                if "json_schema" in f:
                    js = f.get("json_schema") or {}
                    name = f.get("name") or js.get("name") or "schema"
                    schema = js.get("schema")
                    strict = js.get("strict")
                    out = {"type": "json_schema", "name": name}
                    if schema is not None: out["schema"] = schema
                    if strict is not None: out["strict"] = strict
                    return {"format": out}
                if "name" in f and "schema" in f:
                    return {"format": f}
            return {"format": f}
        if fmt.get("type") == "json_schema" and "json_schema" in fmt:
            js = fmt.get("json_schema") or {}
            name = fmt.get("name") or js.get("name") or "schema"
            schema = js.get("schema")
            strict = js.get("strict")
            out = {"type": "json_schema", "name": name}
            if schema is not None: out["schema"] = schema
            if strict is not None: out["strict"] = strict
            return {"format": out}
        if fmt.get("type") == "json_schema" and "schema" in fmt:
            out = {"type": "json_schema", "name": fmt.get("name") or "schema", "schema": fmt.get("schema")}
            if "strict" in fmt: out["strict"] = fmt["strict"]
            return {"format": out}
        if "schema" in fmt:
            out = {"type": "json_schema", "name": fmt.get("name") or "schema", "schema": fmt.get("schema")}
            if "strict" in fmt: out["strict"] = fmt["strict"]
            return {"format": out}
        return fmt

    @staticmethod
    def _usage_from_chat(resp: Any) -> Optional[Dict[str, Any]]:
        u = getattr(resp, "usage", None)
        if not u: return None
        return {"prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None), "total_tokens": getattr(u, "total_tokens", None)}

    @staticmethod
    def _usage_from_responses(resp: Any) -> Optional[Dict[str, Any]]:
        u = getattr(resp, "usage", None)
        if not u: return None
        return {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None), "total_tokens": getattr(u, "total_tokens", None)}

    def _is_responses_model(self, model: str) -> bool:
        return model.lower().startswith(("gpt-5", "o1", "o3"))

    # ── async ──

    async def chat_async(self, request: LLMRequest) -> LLMResponse:
        try:
            if self._is_responses_model(request.model):
                system_msg = next((m.content for m in request.messages if m.role in ("system", "developer")), None)
                other_messages = [m for m in request.messages if m.role not in ("system", "developer")]
                params: Dict[str, Any] = {"model": request.model, "input": self._to_responses_input(other_messages)}
                if system_msg: params["instructions"] = system_msg
                if request.max_tokens is not None: params["max_output_tokens"] = request.max_tokens
                if request.reasoning_effort: params["reasoning"] = {"effort": request.reasoning_effort}
                if request.response_format: params["text"] = self._normalize_text_format_for_responses(request.response_format)
                resp = await self.async_client.responses.create(**params)
                return LLMResponse(content=getattr(resp, "output_text", "") or "", model=getattr(resp, "model", request.model), provider="openai", usage=self._usage_from_responses(resp), raw_response=resp)
            else:
                params: Dict[str, Any] = {"model": request.model, "messages": self._to_chat_messages(request.messages)}
                if request.max_tokens is not None: params["max_tokens"] = request.max_tokens
                if request.temperature is not None: params["temperature"] = request.temperature
                if request.top_p is not None: params["top_p"] = request.top_p
                if request.response_format: params["response_format"] = request.response_format
                resp = await self.async_client.chat.completions.create(**params)
                return LLMResponse(content=(resp.choices[0].message.content if resp.choices else "") or "", model=getattr(resp, "model", request.model), provider="openai", usage=self._usage_from_chat(resp), raw_response=resp)
        except Exception as e:
            raise LLMException(f"OpenAI API error: {e}", "openai", request.model)

    # ── sync ──

    def chat_sync(self, request: LLMRequest) -> LLMResponse:
        try:
            if self._is_responses_model(request.model):
                params: Dict[str, Any] = {"model": request.model, "input": self._to_responses_input(request.messages)}
                if request.max_tokens is not None: params["max_output_tokens"] = request.max_tokens
                if request.reasoning_effort: params["reasoning"] = {"effort": request.reasoning_effort}
                if request.response_format: params["text"] = self._normalize_text_format_for_responses(request.response_format)
                resp = self.client.responses.create(**params)
                return LLMResponse(content=getattr(resp, "output_text", "") or "", model=getattr(resp, "model", request.model), provider="openai", usage=self._usage_from_responses(resp), raw_response=resp)
            else:
                params: Dict[str, Any] = {"model": request.model, "messages": self._to_chat_messages(request.messages)}
                if request.max_tokens is not None: params["max_tokens"] = request.max_tokens
                if request.temperature is not None: params["temperature"] = request.temperature
                if request.top_p is not None: params["top_p"] = request.top_p
                if request.response_format: params["response_format"] = request.response_format
                resp = self.client.chat.completions.create(**params)
                return LLMResponse(content=(resp.choices[0].message.content if resp.choices else "") or "", model=getattr(resp, "model", request.model), provider="openai", usage=self._usage_from_chat(resp), raw_response=resp)
        except Exception as e:
            raise LLMException(f"OpenAI API error: {e}", "openai", request.model)
