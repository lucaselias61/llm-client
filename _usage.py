from abc import ABC, abstractmethod
from decimal import Decimal

from ._config import *
from ._types import *

__all__ = ["get_cost", "normalize_token_usage", "get_nested_attr"]


def get_cost(model: ModelConfig, usage: UsageBreakdown) -> CostBreakdown:

    pricing: ModelPricing = model.pricing

    costs = CostBreakdown()

    if model.provider == "openai":
       
        non_cached_input_tokens = usage.input_tokens - usage.cached_input_tokens
        costs.input_cost_USD = (Decimal(non_cached_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.input_per_1m))
        costs.output_cost_USD = (Decimal(usage.output_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.output_per_1m))
        costs.cached_input_cost_USD = (Decimal(usage.cached_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.cached_input_per_1m))
        costs.total_cost_USD = costs.input_cost_USD + costs.output_cost_USD + costs.cached_input_cost_USD
    
    elif model.provider == "anthropic":
        costs.input_cost_USD = (Decimal(usage.input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.input_per_1m))
        costs.output_cost_USD = (Decimal(usage.output_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.output_per_1m))
        costs.cache_creation_cost_USD = (Decimal(usage.cache_creation_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.cache_creation_input_per_1m))
        costs.cache_read_cost_USD = (Decimal(usage.cache_read_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.cache_read_input_per_1m))
        costs.total_cost_USD = costs.input_cost_USD + costs.output_cost_USD + costs.cache_creation_cost_USD + costs.cache_read_cost_USD
    
    elif model.provider == "gemini":
        costs.input_cost_USD = (Decimal(usage.input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.input_per_1m))
        costs.output_cost_USD = (Decimal(usage.output_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.output_per_1m))
        costs.total_cost_USD = costs.input_cost_USD + costs.output_cost_USD

    if costs.total_cost_USD is None:
        raise ValueError("Total cost could not be calculated due to missing usage or pricing information")
    
    return costs

def normalize_token_usage(provider: str, response: object | None) -> UsageBreakdown:
    if response is None:
        return UsageBreakdown()
    
    provider_config: ProviderConfig = PROVIDER_CATALOG[provider]
    usage_paths: UsagePaths = provider_config.usage_paths

    def read(path: str | None) -> int:
        if path is None:
            return 0
        return get_nested_attr(response, path, 0) or 0

    return UsageBreakdown(
        input_tokens=read(usage_paths.input_tokens),
        output_tokens=read(usage_paths.output_tokens),
        cached_input_tokens=read(usage_paths.cached_input_tokens),
        cache_creation_input_tokens=read(usage_paths.cache_creation_input_tokens),
        cache_read_input_tokens=read(usage_paths.cache_read_input_tokens),
        reasoning_tokens=read(usage_paths.reasoning_tokens),
        total_tokens=read(usage_paths.total_tokens),
    )

def get_nested_attr(obj, path: str, default=None):
    value = obj
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return default
    return value
