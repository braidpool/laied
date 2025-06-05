"""
Provider manager for handling different LLM providers.
"""

import os
from typing import Dict, Any, List, Optional, Union, Iterator
from types import SimpleNamespace

from .base import BaseProvider, ModelResponse, ProviderError
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .ollama_provider import OllamaProvider


class ProviderManager:
    """Manages different LLM providers and routes requests appropriately."""
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._setup_default_providers()
    
    def _setup_default_providers(self):
        """Set up default provider instances."""
        self._providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "ollama": OllamaProvider(),
        }
    
    def get_provider_for_model(self, model: str) -> BaseProvider:
        """Get the appropriate provider for a model."""
        # Extract provider from model name
        if "/" in model:
            provider_name = model.split("/")[0]
            model_name = model.split("/", 1)[1]
        else:
            # Try to detect provider from model name patterns
            provider_name = self._detect_provider_from_model(model)
            model_name = model
        
        # Handle special provider mappings
        provider_mappings = {
            "gpt": "openai",
            "claude": "anthropic",
            "ollama_chat": "ollama",
            "whisper": "openai",  # Handle whisper models
        }
        provider_name = provider_mappings.get(provider_name, provider_name)
        
        if provider_name not in self._providers:
            raise ProviderError(f"Unknown provider: {provider_name}")
        
        return self._providers[provider_name]
    
    def _detect_provider_from_model(self, model: str) -> str:
        """Detect provider from model name patterns."""
        model_lower = model.lower()
        
        if any(pattern in model_lower for pattern in ["gpt", "o1", "chatgpt", "whisper"]):
            return "openai"
        elif any(pattern in model_lower for pattern in ["claude"]):
            return "anthropic"
        elif any(pattern in model_lower for pattern in ["gemini", "palm"]):
            return "gemini"
        elif any(pattern in model_lower for pattern in ["llama", "mixtral", "mistral", "qwen", "phi", "gemma"]):
            return "ollama"
        else:
            # Default to OpenAI for unknown models (most compatible)
            return "openai"
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Union[ModelResponse, Iterator]:
        """Generate a completion using the appropriate provider."""
        provider = self.get_provider_for_model(model)
        
        # Extract model name without provider prefix
        if "/" in model:
            model_name = model.split("/", 1)[1]
        else:
            model_name = model
        
        result = provider.completion(
            model=model_name,
            messages=messages,
            stream=stream,
            temperature=temperature,
            timeout=timeout,
            **kwargs
        )
        
        if stream:
            # Wrap string iterator in objects that match LiteLLM format
            return self._wrap_streaming_response(result)
        else:
            return result
    
    def _wrap_streaming_response(self, string_iterator: Iterator[str]):
        """Wrap streaming strings in objects that match LiteLLM format."""
        for content in string_iterator:
            # Create a chunk object that matches LiteLLM streaming format
            chunk = SimpleNamespace()
            chunk.choices = [SimpleNamespace()]
            chunk.choices[0].delta = SimpleNamespace()
            chunk.choices[0].delta.content = content
            chunk.choices[0].finish_reason = None
            yield chunk
        
        # Send final chunk with finish_reason
        final_chunk = SimpleNamespace()
        final_chunk.choices = [SimpleNamespace()]
        final_chunk.choices[0].delta = SimpleNamespace()
        final_chunk.choices[0].delta.content = ""
        final_chunk.choices[0].finish_reason = "stop"
        yield final_chunk
    
    def validate_environment(self, model: str) -> Dict[str, Any]:
        """Validate environment for a specific model's provider."""
        provider = self.get_provider_for_model(model)
        return provider.validate_environment()
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model information from the appropriate provider."""
        provider = self.get_provider_for_model(model)
        
        # Extract model name without provider prefix
        if "/" in model:
            model_name = model.split("/", 1)[1]
        else:
            model_name = model
        
        return provider.get_model_info(model_name)
    
    def tokenize(self, text: str, model: str) -> List[int]:
        """Tokenize text using the appropriate provider."""
        provider = self.get_provider_for_model(model)
        
        # Extract model name without provider prefix
        if "/" in model:
            model_name = model.split("/", 1)[1]
        else:
            model_name = model
        
        return provider.tokenize(text, model_name)
    
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count tokens using the appropriate provider."""
        provider = self.get_provider_for_model(model)
        
        # Extract model name without provider prefix
        if "/" in model:
            model_name = model.split("/", 1)[1]
        else:
            model_name = model
        
        return provider.count_tokens(messages, model_name)
    
    def add_provider(self, name: str, provider: BaseProvider):
        """Add a custom provider."""
        self._providers[name] = provider
    
    def configure_provider(self, name: str, **config):
        """Configure an existing provider."""
        if name in self._providers:
            provider_class = type(self._providers[name])
            self._providers[name] = provider_class(**config)
        else:
            raise ProviderError(f"Provider {name} not found")


# Global provider manager instance
provider_manager = ProviderManager()