"""AWS Bedrock provider — Claude, Titan, Llama, Mistral, AI21."""

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
    """AWS Bedrock provider implementation."""

    SUPPORTED_MODELS = ["claude", "anthropic", "titan", "llama", "mistral", "ai21"]

    def __init__(self, region: Optional[str] = None):
        if not boto3:
            raise ProviderNotAvailableException(
                "boto3 package not installed. Run: pip install boto3",
                "bedrock", "unknown",
            )
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.session = boto3.Session(region_name=self.region)
        self.client = self.session.client("bedrock-runtime")

    def supports_model(self, model: str) -> bool:
        return any(s in model.lower() for s in self.SUPPORTED_MODELS)

    def _format_for_claude(self, request: LLMRequest) -> Dict[str, Any]:
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"Human: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        prompt += "Assistant:"
        return {
            "prompt": prompt,
            "max_tokens_to_sample": request.max_tokens or 500,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9,
        }

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

    def _convert_request(self, request: LLMRequest) -> Dict[str, Any]:
        if "claude" in request.model.lower():
            return self._format_for_claude(request)
        elif "titan" in request.model.lower():
            return self._format_for_titan(request)
        else:
            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            return {"prompt": prompt, "max_tokens": request.max_tokens or 500, "temperature": request.temperature or 0.7}

    async def chat_async(self, request: LLMRequest) -> LLMResponse:
        return await asyncio.get_event_loop().run_in_executor(None, self.chat_sync, request)

    def chat_sync(self, request: LLMRequest) -> LLMResponse:
        try:
            body = self._convert_request(request)
            response = self.client.invoke_model(
                modelId=request.model,
                body=json.dumps(body),
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            if "claude" in request.model.lower():
                content = response_body.get("completion", "")
            elif "titan" in request.model.lower():
                content = response_body.get("results", [{}])[0].get("outputText", "")
            else:
                content = str(response_body)
            return LLMResponse(
                content=content,
                model=request.model,
                provider="bedrock",
                usage=response_body.get("usage"),
                raw_response=response_body,
            )
        except Exception as e:
            raise LLMException(f"Bedrock API error: {e}", "bedrock", request.model)
