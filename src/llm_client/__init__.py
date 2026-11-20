from typing import Any, Self, overload

from pydantic import BaseModel

from ._config import (
    MODEL_CATALOG,
    PROVIDER_CATALOG,
    get_default_model_for_provider,
    get_model_config,
    get_models,
    get_models_for_provider,
    get_providers,
)
from ._usage import CostBreakdown, Usage, UsageBreakdown
from .anthropic import AnthropicClient
from .claude import ClaudeClient
from .client import Client
from .codex import CodexClient
from .google import GoogleClient
from .openai import OpenAIClient
from .typesafeai import TypeSafeAIClient

__version__ = "3.0.0"
__author__ = "Lucas Passadore Elias"

__all__ = [
    "MODEL_CATALOG",
    "PROVIDER_CATALOG",
    "AnthropicClient",
    "ClaudeClient",
    "Client",
    "CodexClient",
    "CostBreakdown",
    "GoogleClient",
    "LLMClient",
    "OpenAIClient",
    "TypeSafeAIClient",
    "Usage",
    "UsageBreakdown",
    "get_default_model_for_provider",
    "get_models",
    "get_models_for_provider",
    "get_providers",
]


client_classes: dict[str, type[Client]] = {
    "anthropic": AnthropicClient,
    "gemini": GoogleClient,
    "openai": OpenAIClient,
    "typesafeai": TypeSafeAIClient,
}

# Clients for each provider's `subscription_cli`, used with `use_subscription=True`.
subscription_client_classes: dict[str, type[Client]] = {
    "claude": ClaudeClient,
    "codex": CodexClient,
}


def _subscription_client_class(provider: str) -> type[Client]:
    provider_config = PROVIDER_CATALOG.get(provider)
    cli = provider_config.subscription_cli if provider_config else None
    if cli is None:
        raise ValueError(f"Provider {provider} has no subscription CLI")
    if cli not in subscription_client_classes:
        raise ValueError(f"Unsupported subscription CLI: {cli}")
    return subscription_client_classes[cli]


class LLMClient:
    def __init__(
        self,
        model_name: str,
        max_concurrency: int = 1,
        *,
        temperature: float | None = None,
        use_subscription: bool = False,
    ) -> None:
        model_config = get_model_config(model_name)
        if model_config is None:
            raise ValueError(f"Unsupported model: {model_name}")
        provider = model_config.provider
        if use_subscription:
            client_class = _subscription_client_class(provider)
        elif provider in client_classes:
            client_class = client_classes[provider]
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        self.temperature = temperature
        self._client: Client = client_class(
            model_name=model_name, max_concurrency=max_concurrency
        )

    @overload
    async def query[ResponseT: BaseModel](
        self,
        *,
        text_format: type[ResponseT],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ResponseT: ...

    @overload
    async def query(
        self,
        *,
        text_format: None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any: ...

    async def query(
        self,
        *,
        text_format: type[BaseModel] | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        effective_temperature = self.temperature if temperature is None else temperature
        if effective_temperature is not None:
            kwargs["temperature"] = effective_temperature
        return await self._client.query(text_format=text_format, **kwargs)

    @property
    def usage(self) -> Usage:
        return self._client.usage

    @property
    def last_usage(self) -> Usage:
        return self._client.last_usage

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
