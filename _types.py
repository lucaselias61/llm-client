from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from decimal import Decimal

__all_ = ["ModelConfig", "ProviderConfig", "UsagePaths", "UsageBreakdown", "CostBreakdown", "ModelPricing", "LLMResult", "asdict"]

@dataclass(frozen=True)
class UsagePaths:
    input_tokens: str | None = None
    output_tokens: str | None = None
    total_tokens: str | None = None
    cached_input_tokens: str | None = None
    reasoning_tokens: str | None = None
    cache_creation_input_tokens: str | None = None
    cache_read_input_tokens: str | None = None
    
@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    default_model: str
    temperature: float = 0.7
    usage_paths: UsagePaths = UsagePaths()

@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_name: str
    pricing: ModelPricing
    supports_reasoning_tokens: bool = False
    notes: str | None = None

@dataclass(frozen=False)
class UsageBreakdown:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    image_input_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: UsageBreakdown) -> UsageBreakdown:
        return UsageBreakdown(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def __str__(self) -> str:

        lines = ["Usage breakdown:"]
        if self.input_tokens:
            lines.append(f"  Input Tokens:          {self.input_tokens}")
        if self.output_tokens:
            lines.append(f"  Output Tokens:         {self.output_tokens}")
        if self.cached_input_tokens:
            lines.append(f"  Cached Input Tokens:   {self.cached_input_tokens}")
        if self.cache_creation_input_tokens:
            lines.append(f"  Cache Creation Tokens: {self.cache_creation_input_tokens}")
        if self.cache_read_input_tokens:
            lines.append(f"  Cache Read Tokens:     {self.cache_read_input_tokens}")
        if self.reasoning_tokens:
            lines.append(f"  Reasoning Tokens:      {self.reasoning_tokens}")
        if self.audio_input_tokens:
            lines.append(f"  Audio Input Tokens:    {self.audio_input_tokens}")
        if self.audio_output_tokens:
            lines.append(f"  Audio Output Tokens:   {self.audio_output_tokens}")
        if self.image_input_tokens:
            lines.append(f"  Image Input Tokens:    {self.image_input_tokens}")
        if self.total_tokens:
            lines.append(f"  Total Tokens:          {self.total_tokens}")
        return "\n".join(lines)

@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None
    training_per_1m: float | None = None
    cache_creation_input_per_1m: float | None = None
    cache_read_input_per_1m: float | None = None

@dataclass(frozen=False)
class CostBreakdown:
    input_cost_USD: Decimal = Decimal(0)
    output_cost_USD: Decimal = Decimal(0)
    cached_input_cost_USD: Decimal = Decimal(0)
    cache_creation_cost_USD: Decimal = Decimal(0)
    cache_read_cost_USD: Decimal = Decimal(0)
    reasoning_cost_USD: Decimal = Decimal(0)
    audio_input_cost_USD: Decimal = Decimal(0)
    audio_output_cost_USD: Decimal = Decimal(0)
    image_input_cost_USD: Decimal = Decimal(0)
    total_cost_USD: Decimal = Decimal(0)

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def __str__(self) -> str:
        def fmt(value: Decimal | None) -> str:
            return f"{round(value, 2)}"
        
        lines = ["Cost breakdown:"]
        if self.input_cost_USD:
            lines.append(f"  Input:          ${fmt(self.input_cost_USD)}")
        if self.output_cost_USD:
            lines.append(f"  Output:         ${fmt(self.output_cost_USD)}")
        if self.cached_input_cost_USD:
            lines.append(f"  Cached input:   ${fmt(self.cached_input_cost_USD)}")
        if self.cache_creation_cost_USD:
            lines.append(f"  Cache creation: ${fmt(self.cache_creation_cost_USD)}")
        if self.cache_read_cost_USD:
            lines.append(f"  Cache read:     ${fmt(self.cache_read_cost_USD)}")
        if self.reasoning_cost_USD:
            lines.append(f"  Reasoning:      ${fmt(self.reasoning_cost_USD)}")
        if self.audio_input_cost_USD:
            lines.append(f"  Audio input:    ${fmt(self.audio_input_cost_USD)}")
        if self.audio_output_cost_USD:
            lines.append(f"  Audio output:   ${fmt(self.audio_output_cost_USD)}")
        if self.image_input_cost_USD:
            lines.append(f"  Image input:    ${fmt(self.image_input_cost_USD)}")
        if self.total_cost_USD:
            lines.append(f"  Total:          ${fmt(self.total_cost_USD)}")
        return "\n".join(lines)

@dataclass(frozen=True)
class LLMResult:
    provider: str
    model: str
    text: str
    usage: UsageBreakdown
    cost: CostBreakdown
    raw_response_id: str | None = None