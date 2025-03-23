from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class LLMConfig:
    """Configuration for the LLM."""

    provider: str = "openai"
    api_key: str = None
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Retry settings
    retry_attempts: int = 3
    retry_delay: float = 1.0
