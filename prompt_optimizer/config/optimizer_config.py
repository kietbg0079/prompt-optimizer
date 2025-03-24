"""Configuration settings for the prompt optimizer."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class OptimizerConfig:
    """Configuration for the prompt optimizer."""
    # Optimization settings
    max_iterations: int = 1
    chunk_size: int = 2
