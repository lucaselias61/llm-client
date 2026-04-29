import base64
import os
from typing import Any, Optional 

from anthropic import Anthropic
from openai import OpenAI
from google.genai import Client as Gemini
from google.genai import types

from ._config import *
from ._types import LLMResult, ProviderConfig, UsageBreakdown, CostBreakdown, ModelConfig
from ._usage import normalize_token_usage, get_cost

class LLMClient:
    def __init__(self, provider: str, model: str | None = None, temperature: float | None = None) -> None:
        config = PROVIDER_CATALOG.get(provider)
        if config is None:
            raise ValueError(f"Unsupported provider: {provider}")
        
        model = MODEL_CATALOG.get(model) if model else None
        if model is None:
            raise ValueError(f"Unsupported model: {model}")

        self.provider: str = provider
        self.config: ProviderConfig = config
        self.model: ModelConfig = model 
        self.temperature: Optional[float] = temperature if temperature is not None else 0.7
        self.client = self._build_client(provider)

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        return_usage: bool = False,
        image_b64: Optional[str] = None,
        image_media_type: Optional[str] = "image/jpeg",
    ) -> LLMResult:
        
        response = self._send_request(
            prompt=prompt,
            system_prompt=system_prompt,
            image_b64=image_b64,
            image_media_type=image_media_type,
        )
        text = self._extract_text(response)

        if return_usage:

            usage: UsageBreakdown = normalize_token_usage(self.provider, response)
            cost: CostBreakdown = get_cost(self.model, usage)

            return LLMResult(
                provider=self.provider,
                model=self.model.model_name,
                text=text,
                usage=usage,
                cost=cost,
                raw_response_id=getattr(response, "id", None)
            )
        
        return LLMResult(
                provider=self.provider,
                model=self.model.model_name,
                text=text,
        )

    def _send_request(self, **kwargs: dict[str, Any]) -> Any:
        
        if self.provider == "openai":
            user_content = [
                {"type": "input_text", "text": kwargs.get("prompt")},
                {"type": "input_image", "image_url": f"data:{kwargs.get('image_media_type')};base64,{kwargs.get('image_b64')}"} if kwargs.get("image_b64") else None,
            ]
            user_content = [item for item in user_content if item is not None]
            input_payload: list[dict] = []
            if kwargs.get("system_prompt"):
                input_payload.append({"role": "system", "content": kwargs.get("system_prompt")})
            input_payload.append({"role": "user", "content": user_content})

            return self.client.responses.create(
                model=self.model.model_name,
                temperature=self.temperature,
                input=input_payload,
            )

        if self.provider == "anthropic":
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
            return self.client.messages.create(
                model=self.model.model_name,
                temperature=self.temperature,
                system=kwargs.get("system_prompt") or "",
                max_tokens=4096,
                messages=[{"role": "user", "content": user_content}],
            )

        if self.provider == "gemini":
            raw_bytes = base64.b64decode(kwargs.get("image_b64"))
            if kwargs.get("system_prompt"):
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    system_instruction=kwargs.get("system_prompt"),
                )
            else:
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                )
            return self.client.models.generate_content(
                model=self.model.model_name,
                contents=[
                    item for item in [
                        types.Part(inline_data=types.Blob(
                            mime_type=kwargs.get("image_media_type"), data=raw_bytes,
                        )) if kwargs.get("image_b64") else None,
                        kwargs.get("prompt"),
                    ] if item is not None
                ],
                config=config,
            )

        raise ValueError(f"Unsupported provider: {self.provider}")
    

    def _extract_text(self, response) -> str:
        if self.provider == "openai":
            return getattr(response, "output_text", "") or ""

        elif self.provider == "anthropic":
            parts: list[str] = []

            for block in getattr(response, "content", []):
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)

            return "".join(parts)
        
        elif self.provider == "gemini":
            return getattr(response, "text", "") or ""

        raise ValueError(f"Unsupported provider: {self.provider}")
    
    @staticmethod
    def _build_client(provider: str):

        if provider == "openai":
            try:
                return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")
                
        if provider == "anthropic":
            try:
                return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")
            
        if provider == "gemini":
            try:
                return Gemini(api_key=os.getenv("GEMINI_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")

        raise ValueError(f"Unknown provider: {provider}")




    