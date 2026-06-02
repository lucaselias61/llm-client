import asyncio
import base64
import os
from typing import Any, Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from google.genai import Client as Gemini
from google.genai import types

from ._config import *
from ._types import ProviderConfig, Usage, ModelConfig
from ._usage import normalize_token_usage, get_cost


_OPENAI_JSON_FORMAT = {"type": "json_object"}


class LLMClient:
    temperature: float 
    model: ModelConfig
    config: ProviderConfig
    client: AsyncOpenAI | AsyncAnthropic | Gemini
    usage: Usage
    last_usage: Usage

    def __init__(
            self,  
            model_name: str = "gpt-4.1-mini", 
            temperature: float = 0.7, 
            max_concurrency: int = 5
    ) -> None:
        model = MODEL_CATALOG.get(model_name) 
        if model is None:
            raise ValueError(f"Unsupported model: {model_name}")
        config = PROVIDER_CATALOG.get(model.provider)
        if config is None:
            raise ValueError(f"Unsupported provider: {model.provider}")

        self.config: ProviderConfig = config
        self.model: ModelConfig = model
        self.temperature: float = temperature
        self.client = self._build_client(model.provider)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.usage = Usage()
        self.last_usage = Usage()

    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        image_b64: Optional[str] = None,
        image_media_type: Optional[str] = "image/jpeg",
        response_format: str | type | dict | None = None,
    ) -> str | list[float]:

        if response_format == "embedding" or response_format is list[float]:
            async with self._semaphore:
                return await self._generate_embedding(prompt)

        if response_format == "json":
            response_format = _OPENAI_JSON_FORMAT

        async with self._semaphore:
            response = await self._send_request(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature if temperature is not None else self.temperature,
                timeout=timeout,
                image_b64=image_b64,
                image_media_type=image_media_type,
                response_format=response_format,
            )
        return self._extract_text(response)
            

    async def _send_request(self, **kwargs: Any) -> Any:
        provider = self.model.provider

        if provider == "openai":
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": kwargs.get("prompt")},
            ]
            if kwargs.get("image_b64"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{kwargs.get('image_media_type')};base64,{kwargs.get('image_b64')}",
                    },
                })
            messages: list[dict[str, Any]] = []
            if kwargs.get("system_prompt"):
                messages.append({"role": "system", "content": kwargs.get("system_prompt")})
            messages.append({"role": "user", "content": user_content})

            request_kwargs: dict[str, Any] = {
                "model": self.model.model_name,
                "messages": messages,
                "timeout": kwargs.get("timeout"),
            }
            if kwargs.get("response_format") is not None:
                request_kwargs["response_format"] = kwargs.get("response_format")
            if not self.model.model_name.startswith(("gpt-5", "o1", "o3", "o4")):
                request_kwargs["temperature"] = kwargs.get("temperature")
            response = await self.client.chat.completions.create(**request_kwargs)

        elif provider == "anthropic":
            user_content = [
                {"type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": kwargs.get("image_media_type"),
                        "data": kwargs.get("image_b64"),
                    },
                } if kwargs.get("image_b64") else None,
                {"type": "text", "text": kwargs.get("prompt")},
            ]
            user_content = [item for item in user_content if item is not None]
            response = await self.client.messages.create(
                model=self.model.model_name,
                temperature=kwargs.get("temperature"),
                system=kwargs.get("system_prompt") or "",
                max_tokens=4096,
                messages=[{"role": "user", "content": user_content}],
                timeout=kwargs.get("timeout"),
            )

        elif provider == "gemini":
            config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature"),
                system_instruction=kwargs.get("system_prompt") if kwargs.get("system_prompt") else None,
                http_options=types.HttpOptions(timeout=kwargs.get("timeout")),
            )
            contents: list[Any] = []
            if kwargs.get("image_b64"):
                raw_bytes = base64.b64decode(kwargs.get("image_b64"))
                contents.append(types.Part(inline_data=types.Blob(
                    mime_type=kwargs.get("image_media_type"), data=raw_bytes,
                )))
            contents.append(kwargs.get("prompt"))
            response = await self.client.aio.models.generate_content(
                model=self.model.model_name,
                contents=contents,
                config=config,
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
        self._add_response_usage(response)
        return response
    
    
    async def _generate_embedding(
        self,
        input: str,
    ) -> list[float]:

        if self.config.embeddings_model is None:
            raise ValueError(f"Provider {self.model.provider} does not support embeddings")
        response = await self.client.embeddings.create(
            model=self.config.embeddings_model,
            input=input,
        )
        self._add_response_usage(response)
        return response.data[0].embedding

    def _extract_text(self, response) -> str:
        provider = self.model.provider
        if provider == "openai":
            choices = getattr(response, "choices", None) or []
            if not choices:
                return ""
            message = getattr(choices[0], "message", None)
            return (getattr(message, "content", "") or "") if message else ""

        elif provider == "anthropic":
            parts: list[str] = []
            for block in getattr(response, "content", []):
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "".join(parts)

        elif provider == "gemini":
            return getattr(response, "text", "") or ""

        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _build_client(provider: str):

        if provider == "openai":
            try:
                return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")

        if provider == "anthropic":
            try:
                return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")

        if provider == "gemini":
            try:
                return Gemini(api_key=os.getenv("GEMINI_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")

        raise ValueError(f"Unknown provider: {provider}")
    
    def _add_response_usage(self, response: Any) -> None:
        token_usage = normalize_token_usage(self.model.provider, response)
        cost = get_cost(self.model, token_usage)
        self.last_usage = Usage(tokens=token_usage, cost=cost)
        self.usage += self.last_usage