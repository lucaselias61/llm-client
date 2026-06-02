import os
from ._types import *

__all__ = ["PROVIDER_CATALOG", "MODEL_CATALOG", "get_providers", "get_models", "get_models_for_provider", "get_default_model_for_provider"]

PROVIDER_CATALOG: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
                provider="openai",
                default_model="gpt-4.1-mini",
                usage_paths=UsagePaths(
                    input_tokens="usage.prompt_tokens",
                    output_tokens="usage.completion_tokens",
                    total_tokens="usage.total_tokens",
                    cached_input_tokens="usage.prompt_tokens_details.cached_tokens",
                    reasoning_tokens="usage.completion_tokens_details.reasoning_tokens"
                ),
                embeddings_model="text-embedding-3-small",
    ),
    "anthropic": ProviderConfig(
                provider="anthropic",
                default_model="claude-sonnet",
                usage_paths=UsagePaths(
                    input_tokens="usage.input_tokens", 
                    output_tokens="usage.output_tokens", 
                    total_tokens="usage.total_tokens", 
                    cached_input_tokens="usage.input_tokens_details.cached_tokens", 
                    reasoning_tokens="usage.output_tokens_details.reasoning_tokens"
                ),
                embeddings_model=None,
    ),
    "gemini": ProviderConfig(
                provider="gemini",
                default_model="gemini-1.5-pro",
                usage_paths=UsagePaths(
                    input_tokens="usage.input_tokens", 
                    output_tokens="usage.output_tokens", 
                    total_tokens="usage.total_tokens", 
                    cached_input_tokens="usage.input_tokens_details.cached_tokens", 
                    reasoning_tokens="usage.output_tokens_details.reasoning_tokens"
                ),
                embeddings_model=None,
    )
}


MODEL_CATALOG: dict[str, ModelConfig] = {
    "gpt-4.1-mini": ModelConfig(
                        provider="openai",
                        model_name="gpt-4.1-mini",
                        pricing=ModelPricing(
                            input_per_1m=0.80, 
                            cached_input_per_1m=0.20, 
                            output_per_1m=3.20, 
                            training_per_1m=5.00
                        ),
                        supports_reasoning_tokens=True),
    "gpt-5-mini": ModelConfig(
                        provider="openai",
                        model_name="gpt-5-mini",
                        pricing=ModelPricing(
                            input_per_1m=0.25, 
                            cached_input_per_1m=0.025, 
                            output_per_1m=2.0, 
                            training_per_1m=None
                        ),
                        supports_reasoning_tokens=False),
    "claude-sonnet": ModelConfig(
                        provider="anthropic",
                        model_name="claude-sonnet",
                        pricing=ModelPricing(
                            input_per_1m=3.00, 
                            output_per_1m=15.00, 
                            cache_creation_input_per_1m=3.75, 
                            cache_read_input_per_1m=0.30
                        ),
                        supports_reasoning_tokens=False),
}

def get_providers() -> list[str]:
    return list(PROVIDER_CATALOG.keys())

def get_models() -> list[str]:
    return list(MODEL_CATALOG.keys())

def get_models_for_provider(provider: str) -> list[str]:
    return [model_name for model_name, cfg in MODEL_CATALOG.items() if cfg.provider == provider]

def get_default_model_for_provider(provider: str) -> str | None:
    config = PROVIDER_CATALOG.get(provider)
    if config is None:
        return None
    return config.default_model