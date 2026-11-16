from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "EMBEDDINGS_MODEL",
    "MODEL_CATALOG",
    "PROVIDER_CATALOG",
    "ModelConfig",
    "ModelPricing",
    "ProviderConfig",
    "get_default_model_for_provider",
    "get_models",
    "get_models_for_provider",
    "get_providers",
]


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    default_model: str


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_name: str
    pricing: ModelPricing
    # Reasoning models reject `temperature`, and their reasoning counts toward the output cap.
    reasoning_model: bool = False
    notes: str | None = None


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None
    training_per_1m: float | None = None
    cache_creation_input_per_1m: float | None = None
    cache_read_input_per_1m: float | None = None


# The only supported embedding model; embeddings always go through OpenAI.
EMBEDDINGS_MODEL = "text-embedding-3-small"

PROVIDER_CATALOG: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        provider="openai",
        default_model="gpt-6-luna",
    ),
    "anthropic": ProviderConfig(
        provider="anthropic",
        default_model="claude-sonnet-5-5",
    ),
    "gemini": ProviderConfig(
        provider="gemini",
        default_model="gemini-3.5-flash-lite",
    ),
    "typesafeai": ProviderConfig(provider="typesafeai", default_model="jev-latest"),
}

MODEL_CATALOG: dict[str, ModelConfig] = {
    "gemini-3.5-flash-lite": ModelConfig(
        provider="gemini",
        model_name="gemini-3.5-flash-lite",
        pricing=ModelPricing(
            input_per_1m=0.30, output_per_1m=2.50, cached_input_per_1m=0.03
        ),
    ),
    "gemini-2.5-flash": ModelConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        pricing=ModelPricing(
            input_per_1m=0.30, output_per_1m=2.50, cached_input_per_1m=0.03
        ),
        notes=(
            "Deprecated: shuts down 2026-12-01, after which requests route to "
            "gemini-3.5-flash-lite."
        ),
    ),
    "jev-latest": ModelConfig(
        provider="typesafeai",
        model_name="jev-latest",
        pricing=ModelPricing(input_per_1m=0.042, output_per_1m=0.0),
        reasoning_model=False,
    ),
    "jev-1.14.0": ModelConfig(
        provider="typesafeai",
        model_name="jev-1.14.0",
        pricing=ModelPricing(input_per_1m=0.042, output_per_1m=0.0),
        reasoning_model=False,
    ),
    "gpt-6-luna": ModelConfig(
        provider="openai",
        model_name="gpt-6-luna",
        # Rates for prompts up to 272K input tokens; longer prompts cost more.
        pricing=ModelPricing(
            input_per_1m=0.10,
            cached_input_per_1m=0.01,
            output_per_1m=0.50,
        ),
        reasoning_model=True,
    ),
    "gpt-6.1-terra": ModelConfig(
        provider="openai",
        model_name="gpt-6.1-terra",
        # Rates for prompts up to 272K input tokens; longer prompts cost more.
        pricing=ModelPricing(
            input_per_1m=1.00,
            cached_input_per_1m=0.05,
            output_per_1m=5.00,
        ),
        reasoning_model=True,
    ),
    "gpt-4.1-mini": ModelConfig(
        provider="openai",
        model_name="gpt-4.1-mini",
        pricing=ModelPricing(
            input_per_1m=0.40,
            cached_input_per_1m=0.10,
            output_per_1m=1.60,
        ),
        reasoning_model=False,
        notes="Deprecated: shuts down 2027-01-14. Replacement: gpt-6-luna.",
    ),
    "gpt-5-mini": ModelConfig(
        provider="openai",
        model_name="gpt-5-mini",
        pricing=ModelPricing(
            input_per_1m=0.25,
            cached_input_per_1m=0.025,
            output_per_1m=2.0,
            training_per_1m=None,
        ),
        reasoning_model=True,
        notes="Deprecated: shuts down 2026-12-11. Replacement: gpt-6.1-terra.",
    ),
    "claude-sonnet-5-5": ModelConfig(
        provider="anthropic",
        model_name="claude-sonnet-5-5",
        pricing=ModelPricing(
            input_per_1m=2.00,
            output_per_1m=10.00,
            # 5-minute cache writes; 1-hour writes cost $4.
            cache_creation_input_per_1m=2.50,
            cache_read_input_per_1m=0.10,
        ),
        # Thinking is on by default and non-default temperature is rejected.
        reasoning_model=True,
    ),
}


def get_provider_config(provider: str) -> ProviderConfig | None:
    return PROVIDER_CATALOG.get(provider)


def get_providers() -> list[str]:
    return list(PROVIDER_CATALOG.keys())


def get_model_config(model_name: str) -> ModelConfig | None:
    return MODEL_CATALOG.get(model_name)


def get_models() -> list[str]:
    return list(MODEL_CATALOG.keys())


def get_models_for_provider(provider: str) -> list[str]:
    return [
        model_name
        for model_name, cfg in MODEL_CATALOG.items()
        if cfg.provider == provider
    ]


def get_default_model_for_provider(provider: str) -> str | None:
    config = PROVIDER_CATALOG.get(provider)
    if config is None:
        return None
    return config.default_model
