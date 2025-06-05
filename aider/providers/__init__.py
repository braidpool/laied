"""
Provider system for direct LLM API integrations.

This module provides provider-specific implementations for communicating
with different LLM providers without LiteLLM as an intermediary.
"""

from .base import BaseProvider, ProviderError, ModelResponse
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider  
from .ollama_provider import OllamaProvider
from .llamacpp_provider import LLamaCPPProvider
from .manager import ProviderManager, provider_manager

__all__ = [
    "BaseProvider",
    "ProviderError", 
    "ModelResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LLamaCPPProvider",
    "ProviderManager",
    "provider_manager",
]