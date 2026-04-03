"""Google Gemini provider using google-genai SDK."""

import os
from typing import Any, Dict, List, Optional

from empujon_llm.types import (
    LLMException, LLMMessage, LLMRequest, LLMResponse,
    ProviderNotAvailableException,
)
from empujon_llm.providers import LLMProviderBase

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


class GeminiProvider(LLMProviderBase):
    """Google Gemini provider implementation using google-genai SDK."""

    SUPPORTED_MODELS = [
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-pro",
        "gemini-3-flash-preview",
    ]

    def __init__(self, api_key: Optional[str] = None, timeout_ms: int = 120_000):
        if not genai:
            raise ProviderNotAvailableException(
                "google-genai package not installed. Run: pip install google-genai",
                "gemini", "unknown",
            )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMException("GEMINI_API_KEY not set", "gemini", "unknown")
        self.client = genai.Client(api_key=self.api_key, http_options={"timeout": timeout_ms})

    def supports_model(self, model: str) -> bool:
        m = (model or "").lower()
        return any(p in m for p in self.SUPPORTED_MODELS)

    # ── schema helpers ──

    def _normalize_schema_for_gemini(self, fmt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        base_schema = None
        if not fmt:
            return None
        if "schema" in fmt and isinstance(fmt["schema"], dict) and "properties" in fmt["schema"]:
            base_schema = fmt["schema"]
        elif "json_schema" in fmt:
            base_schema = fmt["json_schema"].get("schema")
        elif fmt.get("type") == "json_schema" and "schema" in fmt:
            base_schema = fmt["schema"]
        elif "properties" in fmt or "type" in fmt:
            base_schema = fmt
        if base_schema:
            return self._clean_schema(base_schema)
        return None

    def _clean_schema(self, schema: Any) -> Any:
        if not isinstance(schema, dict):
            if isinstance(schema, list):
                return [self._clean_schema(i) for i in schema]
            return schema
        incompatible = {"additionalProperties", "strict", "property_ordering"}
        return {k: self._clean_schema(v) for k, v in schema.items() if k not in incompatible}

    def _to_gemini_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        out = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            out.append({"role": role, "parts": [{"text": msg.content}]})
        return out

    def _build_config(self, request: LLMRequest) -> tuple:
        """Returns (config_params dict, system_msg str|None, contents list)."""
        config_params: Dict[str, Any] = {}
        if request.max_tokens:
            config_params["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config_params["temperature"] = request.temperature
        if request.top_p is not None:
            config_params["top_p"] = request.top_p
        if request.response_format:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = self._normalize_schema_for_gemini(request.response_format)

        system_msg = next((m.content for m in request.messages if m.role in ("system", "developer")), None)
        user_messages = [m for m in request.messages if m.role not in ("system", "developer")]
        contents = self._to_gemini_messages(user_messages)

        if system_msg:
            config_params["system_instruction"] = system_msg

        return config_params, contents

    @staticmethod
    def _parse_usage(response: Any) -> Dict[str, Any]:
        um = response.usage_metadata
        return {
            "prompt_tokens": um.prompt_token_count,
            "completion_tokens": um.candidates_token_count,
            "total_tokens": um.total_token_count,
        }

    # ── async ──

    async def chat_async(self, request: LLMRequest) -> LLMResponse:
        try:
            config_params, contents = self._build_config(request)
            response = await self.client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config_params),
            )
            return LLMResponse(
                content=response.text,
                model=request.model,
                provider="gemini",
                usage=self._parse_usage(response),
                raw_response=response,
            )
        except Exception as e:
            raise LLMException(f"Gemini API error: {e}", "gemini", request.model)

    # ── sync ──

    def chat_sync(self, request: LLMRequest) -> LLMResponse:
        try:
            config_params, contents = self._build_config(request)
            response = self.client.models.generate_content(
                model=request.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config_params),
            )
            return LLMResponse(
                content=response.text,
                model=request.model,
                provider="gemini",
                usage=self._parse_usage(response),
                raw_response=response,
            )
        except Exception as e:
            raise LLMException(f"Gemini API error: {e}", "gemini", request.model)
