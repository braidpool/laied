"""
Base provider interface for LLM API integrations.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, retry: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retry = retry


@dataclass
class ModelResponse:
    """Response from a model completion request."""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    
    @property
    def choices(self):
        """Compatibility property to match LiteLLM response format."""
        from types import SimpleNamespace
        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = self.content
        choice.finish_reason = self.finish_reason
        return [choice]


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = kwargs
    
    @abstractmethod
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Union[ModelResponse, Iterator[str]]:
        """
        Generate a completion from the model.
        
        Args:
            model: The model identifier
            messages: List of message dicts with 'role' and 'content' keys
            stream: Whether to stream the response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ModelResponse for non-streaming, Iterator[str] for streaming
        """
        pass
    
    @abstractmethod
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate that the environment is properly configured for this provider.
        
        Returns:
            Dict with 'keys_in_environment' (bool) and 'missing_keys' (list)
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model: The model identifier
            
        Returns:
            Dict with model metadata (max_tokens, cost, etc.)
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str, model: str) -> List[int]:
        """
        Tokenize text using the provider's tokenizer.
        
        Args:
            text: Text to tokenize
            model: Model to use for tokenization
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """
        Count tokens in a message list.
        
        Args:
            messages: List of message dicts
            model: Model to use for counting
            
        Returns:
            Total token count
        """
        pass
    
    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for this provider's API.
        
        Default implementation returns messages unchanged.
        Override for provider-specific formatting.
        """
        return messages
    
    def handle_error(self, error: Exception, retry_count: int = 0) -> ProviderError:
        """
        Convert provider-specific errors to standardized ProviderError.
        
        Args:
            error: The original error
            retry_count: Current retry count
            
        Returns:
            ProviderError with appropriate retry flag
        """
        # Default error handling
        should_retry = retry_count < 3 and isinstance(error, (ConnectionError, TimeoutError))
        return ProviderError(str(error), retry=should_retry)
    
    def _extract_content_from_response(self, response_data: Dict[str, Any]) -> str:
        """
        Extract content from a completion response.
        
        Override for provider-specific response formats.
        """
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "")
            elif "text" in choice:
                return choice["text"]
        return ""
    
    def _extract_usage_from_response(self, response_data: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """
        Extract usage information from a completion response.
        
        Override for provider-specific response formats.
        """
        return response_data.get("usage")