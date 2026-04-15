import os
from typing import Any, Optional 

from anthropic import Anthropic
from openai import OpenAI
from google.genai import Client as Gemini
from google.genai import types

from .config import *
from ._types import LLMResult, ProviderConfig, UsageBreakdown, CostBreakdown, ModelConfig
from ._usage import normalize_token_usage, get_cost
from .api_key import load_api_keys

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
    ) -> LLMResult:
        
        response = self._send_request(prompt=prompt, system_prompt=system_prompt)
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

    def _send_request(self, prompt: str, system_prompt: str | None = None):
    
        if self.provider == "openai":
            input_payload: str | list[dict[str, str]]

            if system_prompt:
                input_payload = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                input_payload = prompt

            return self.client.responses.create(
                model=self.model.model_name,
                temperature=self.temperature,
                input=input_payload,
            )

        if self.provider == "anthropic":
            return self.client.messages.create(
                model=self.model.model_name,
                temperature=self.temperature,
                system=system_prompt if system_prompt is not None else "",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

        if self.provider == "gemini":

            if system_prompt:
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    system_instruction=system_prompt,
                )
            else:
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                )

            return self.client.models.generate_content(
                model=self.model.model_name,
                contents=prompt,
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

        try:
            api_keys = load_api_keys()
        except Exception as e:
            print("Error loading API keys:", e)

        if provider == "openai":
            try:
                return OpenAI(api_key=api_keys.get("OPENAI_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")
                
        if provider == "anthropic":
            try:
                return Anthropic(api_key=api_keys.get("ANTHROPIC_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")
            
        if provider == "gemini":
            try:
                return Gemini(api_key=api_keys.get("GEMINI_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")

        raise ValueError(f"Unknown provider: {provider}")




    