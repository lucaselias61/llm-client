import os
from ._types import *

__all__ = ["PROVIDER_CATALOG", "MODEL_CATALOG"]

PROVIDER_CATALOG: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
                provider="openai",
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                usage_paths=UsagePaths(
                    input_tokens="usage.input_tokens", 
                    output_tokens="usage.output_tokens", 
                    total_tokens="usage.total_tokens", 
                    cached_input_tokens="usage.input_tokens_details.cached_tokens", 
                    reasoning_tokens="usage.output_tokens_details.reasoning_tokens"
                )),
    "anthropic": ProviderConfig(
                provider="anthropic",
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
                usage_paths=UsagePaths(
                    input_tokens="usage.input_tokens", 
                    output_tokens="usage.output_tokens", 
                    total_tokens="usage.total_tokens", 
                    cached_input_tokens="usage.input_tokens_details.cached_tokens", 
                    reasoning_tokens="usage.output_tokens_details.reasoning_tokens"
                )),
    "gemini": ProviderConfig(
                provider="gemini",
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
                usage_paths=UsagePaths(
                    input_tokens="usage.input_tokens", 
                    output_tokens="usage.output_tokens", 
                    total_tokens="usage.total_tokens", 
                    cached_input_tokens="usage.input_tokens_details.cached_tokens", 
                    reasoning_tokens="usage.output_tokens_details.reasoning_tokens"
                ))
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

