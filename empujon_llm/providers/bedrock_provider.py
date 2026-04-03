"""AWS Bedrock provider — Claude (Messages API), Titan, Llama, Mistral, AI21."""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from empujon_llm.types import (
    LLMException, LLMMessage, LLMRequest, LLMResponse,
    ProviderNotAvailableException,
)
from empujon_llm.providers import LLMProviderBase

try:
    import boto3
except ImportError:
    boto3 = None


class BedrockProvider(LLMProviderBase):
    """AWS Bedrock provider — uses Messages API for Claude 3+."""

    SUPPORTED_MODELS = ["claude", "anthropic", "titan", "llama", "mistral", "ai21"]

    def __init__(self, region: Optional[str] = None):
        if not boto3:
            raise ProviderNotAvailableException(
                "boto3 package not installed. Run: pip install boto3",
                "bedrock", "unknown",
            )
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-2")
        self.session = boto3.Session(region_name=self.region)
        self.client = self.session.client("bedrock-runtime")

    def supports_model(self, model: str) -> bool:
        return any(s in model.lower() for s in self.SUPPORTED_MODELS)

    # ── Claude Messages API ──

    def _format_for_claude(self, request: LLMRequest) -> Dict[str, Any]:
        """Format for Claude Messages API (Claude 3+ on Bedrock)."""
        system_text = None
        messages = []

        for msg in request.messages:
            if msg.role in ("system", "developer"):
                system_text = msg.content
            else:
                role = "assistant" if msg.role == "assistant" else "user"
                messages.append({"role": role, "content": msg.content})

        # Ensure at least one user message
        if not messages:
            messages = [{"role": "user", "content": ""}]

        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }

        if system_text:
            body["system"] = system_text
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p

        return body

    def _parse_claude_response(self, response_body: Dict[str, Any]) -> tuple:
        """Parse Claude Messages API response → (content, usage)."""
        content_blocks = response_body.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = response_body.get("usage")
        usage_dict = None
        if usage:
            usage_dict = {
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
                "total_tokens": (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
            }

        return text, usage_dict

    # ── Titan ──

    def _format_for_titan(self, request: LLMRequest) -> Dict[str, Any]:
        prompt = "\n".join([msg.content for msg in request.messages if msg.role == "user"])
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": request.max_tokens or 500,
                "temperature": request.temperature or 0.7,
                "topP": request.top_p or 0.9,
            },
        }

    # ── Dispatch ──

    def _is_claude(self, model: str) -> bool:
        m = model.lower()
        return "claude" in m or "anthropic" in m

    async def chat_async(self, request: LLMRequest) -> LLMResponse:
        return await asyncio.get_event_loop().run_in_executor(None, self.chat_sync, request)

    def chat_sync(self, request: LLMRequest) -> LLMResponse:
        try:
            if self._is_claude(request.model):
                body = self._format_for_claude(request)
                response = self.client.invoke_model(
                    modelId=request.model,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                response_body = json.loads(response["body"].read())
                content, usage = self._parse_claude_response(response_body)

            elif "titan" in request.model.lower():
                body = self._format_for_titan(request)
                response = self.client.invoke_model(
                    modelId=request.model,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                response_body = json.loads(response["body"].read())
                content = response_body.get("results", [{}])[0].get("outputText", "")
                usage = None

            else:
                prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
                body = {"prompt": prompt, "max_tokens": request.max_tokens or 500, "temperature": request.temperature or 0.7}
                response = self.client.invoke_model(
                    modelId=request.model,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                response_body = json.loads(response["body"].read())
                content = str(response_body)
                usage = None

            return LLMResponse(
                content=content,
                model=request.model,
                provider="bedrock",
                usage=usage,
                raw_response=response_body,
            )

        except Exception as e:
            raise LLMException(f"Bedrock API error: {e}", "bedrock", request.model)
