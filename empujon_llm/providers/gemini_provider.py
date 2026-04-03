"""Google Gemini provider — supports both google-genai (new) and google-generativeai (legacy)."""

import os
from typing import Any, Dict, List, Optional

from empujon_llm.types import (
    LLMException, LLMMessage, LLMRequest, LLMResponse,
    ProviderNotAvailableException,
)
from empujon_llm.providers import LLMProviderBase

# Try new SDK first, fall back to legacy
_USE_NEW_SDK = False
genai = None
genai_types = None
genai_legacy = None

try:
    from google import genai
    from google.genai import types as genai_types
    _USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
    except ImportError:
        pass


class GeminiProvider(LLMProviderBase):
    """Google Gemini provider implementation using google-genai SDK."""

    SUPPORTED_MODELS = [
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-pro",
        "gemini-3-flash-preview",
    ]

    def __init__(self, api_key: Optional[str] = None, timeout_ms: int = 120_000):
        if not genai and not genai_legacy:
            raise ProviderNotAvailableException(
                "Neither google-genai nor google-generativeai installed. Run: pip install google-genai",
                "gemini", "unknown",
            )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMException("GEMINI_API_KEY not set", "gemini", "unknown")
        self._use_new_sdk = _USE_NEW_SDK
        if self._use_new_sdk:
            self.client = genai.Client(api_key=self.api_key, http_options={"timeout": timeout_ms})
        else:
            genai_legacy.configure(api_key=self.api_key)

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

    # ── legacy SDK helpers ──

    def _build_legacy_config(self, request: LLMRequest) -> dict:
        config = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_tokens:
            config["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            config["top_p"] = request.top_p
        return config

    def _legacy_call(self, request: LLMRequest) -> LLMResponse:
        """Call using google-generativeai (legacy SDK)."""
        system_msg = next((m.content for m in request.messages if m.role in ("system", "developer")), None)
        user_msgs = [m.content for m in request.messages if m.role not in ("system", "developer")]
        prompt = "\n".join(user_msgs)

        model_kwargs = {}
        if system_msg:
            model_kwargs["system_instruction"] = system_msg

        model = genai_legacy.GenerativeModel(request.model, **model_kwargs)
        gen_config = self._build_legacy_config(request)
        response = model.generate_content(prompt, generation_config=gen_config)
        text = response.text

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", None),
                "completion_tokens": getattr(um, "candidates_token_count", None),
                "total_tokens": getattr(um, "total_token_count", None),
            }

        return LLMResponse(content=text, model=request.model, provider="gemini", usage=usage, raw_response=response)

    # ── async ──

    async def chat_async(self, request: LLMRequest) -> LLMResponse:
        try:
            if not self._use_new_sdk:
                # Legacy SDK has no native async — run in executor
                import asyncio
                return await asyncio.get_event_loop().run_in_executor(None, self._legacy_call, request)

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
            if not self._use_new_sdk:
                return self._legacy_call(request)

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
