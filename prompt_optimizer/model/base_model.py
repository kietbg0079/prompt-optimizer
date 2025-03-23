"""Base class for LLM model implementations with synchronous and asynchronous support."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Awaitable, TypeVar
import time
import json
import asyncio
from functools import wraps

# Type variable for generic return type
T = TypeVar('T')

class BaseModel(ABC):
    """
    Abstract base class for LLM model implementations.
    
    This class defines the interface that all LLM model implementations
    must follow, providing a consistent API regardless of the underlying
    model provider (OpenAI, Anthropic, Google, etc.).
    
    Supports both synchronous and asynchronous operations.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        retry_attempts: int = 3,
        retry_delay: float = 0.5,
        **kwargs
    ):
        """
        Initialize the LLM model.
        
        Args:
            model_name: Name of the specific model to use
            api_key: API key for the model provider
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            retry_attempts: Number of retry attempts for API calls
            retry_delay: Delay between retry attempts in seconds
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.model_params = kwargs
        
        # Initialize the model client
        self._initialize_client()
        
    @abstractmethod
    def _initialize_client(self) -> None:
        pass
    
    @abstractmethod
    async def _initialize_async_client(self) -> None:
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass
    
    @abstractmethod
    async def generate_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass
    
    def with_retries(self, func: Callable[..., T], *args, **kwargs) -> T:
        last_error = None  
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt == self.retry_attempts - 1:
                    raise
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        raise last_error if last_error else RuntimeError("Unknown error during retries")
    
    async def with_retries_async(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise last_error if last_error else RuntimeError("Unknown error during async retries")
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with variables.
        
        Args:
            template: Prompt template with {variable} placeholders
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        return template.format(**kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "provider": self.get_provider_name(),
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **self.model_params
            }
        }
    
    def validate_api_key(self) -> bool:
        try:
            self.generate("Hello, this is a test prompt. Please respond with 'API key is valid'.")
            return True
        except Exception:
            return False
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model={self.model_name}, provider={self.get_provider_name()})"

# Helper for running async code from sync context
def run_async(async_func, *args, **kwargs):
    """
    Run an async function from a synchronous context.
    
    Args:
        async_func: Async function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the async function
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_func(*args, **kwargs))
