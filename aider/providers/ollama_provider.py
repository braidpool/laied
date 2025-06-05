"""
Ollama provider implementation.
"""

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests

from .base import BaseProvider, ModelResponse, ProviderError


class OllamaProvider(BaseProvider):
    """Ollama API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        
        # Ollama doesn't use API keys by default
        if not self.base_url:
            self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Ensure base URL doesn't end with /v1 (Ollama uses different endpoints)
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Union[ModelResponse, Iterator[str]]:
        """Generate a completion using Ollama API."""
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": self.format_messages(messages),
            "stream": stream
        }
        
        # Ollama options
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        
        # Handle num_ctx (context size)
        if "num_ctx" in kwargs:
            options["num_ctx"] = kwargs.pop("num_ctx")
        
        # Handle other Ollama-specific options
        for key in ["top_k", "top_p", "repeat_penalty", "seed"]:
            if key in kwargs:
                options[key] = kwargs.pop(key)
        
        if options:
            data["options"] = options
        
        # Handle format (for structured output)
        if "format" in kwargs:
            data["format"] = kwargs.pop("format")
        
        # Add any remaining kwargs
        data.update(kwargs)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers=headers,
                json=data,
                timeout=timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_non_streaming_response(response)
                
        except requests.exceptions.RequestException as e:
            raise self.handle_error(e)
    
    def _handle_streaming_response(self, response: requests.Response) -> Iterator[str]:
        """Handle streaming response from Ollama API."""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        content = data["message"]["content"]
                        if content:
                            yield content
                    
                    # Check if this is the final chunk
                    if data.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue
    
    def _handle_non_streaming_response(self, response: requests.Response) -> ModelResponse:
        """Handle non-streaming response from Ollama API."""
        response_data = response.json()
        
        content = ""
        if "message" in response_data and "content" in response_data["message"]:
            content = response_data["message"]["content"]
        
        # Ollama doesn't provide detailed usage stats by default
        usage = None
        if "prompt_eval_count" in response_data or "eval_count" in response_data:
            usage = {
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "completion_tokens": response_data.get("eval_count", 0),
                "total_tokens": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
            }
        
        return ModelResponse(
            content=content,
            finish_reason=response_data.get("done_reason"),
            usage=usage,
            model=response_data.get("model")
        )
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate Ollama environment configuration."""
        # Ollama doesn't require API keys, just check if server is reachable
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return {
                    "keys_in_environment": True,
                    "missing_keys": []
                }
        except requests.exceptions.RequestException:
            pass
        
        return {
            "keys_in_environment": False,
            "missing_keys": [f"Ollama server at {self.base_url}"]
        }
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get Ollama model information."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract model info from Ollama response
                info = {}
                
                # Try to get context size from model info
                if "model_info" in data:
                    model_info = data["model_info"]
                    # Look for context size in various possible fields
                    for key in ["context_length", "max_sequence_length", "n_ctx"]:
                        if key in model_info:
                            info["max_input_tokens"] = model_info[key]
                            info["max_output_tokens"] = model_info[key]
                            break
                
                # Default context size if not found
                if "max_input_tokens" not in info:
                    info["max_input_tokens"] = 4096
                    info["max_output_tokens"] = 4096
                
                # Ollama is typically free/self-hosted
                info["input_cost_per_token"] = 0.0
                info["output_cost_per_token"] = 0.0
                
                return info
                
        except requests.exceptions.RequestException:
            pass
        
        # Default fallback for Ollama models
        return {
            "max_input_tokens": 4096,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0
        }
    
    def tokenize(self, text: str, model: str) -> List[int]:
        """Tokenize text (Ollama doesn't expose tokenizer directly)."""
        # Rough estimation: ~4 chars per token
        return list(range(len(text) // 4))
    
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count tokens in messages (estimation)."""
        # Rough estimation for Ollama models
        total_chars = 0
        for message in messages:
            total_chars += len(message.get("content", ""))
            total_chars += 10  # Overhead for role and formatting
        
        # Rough estimation: ~4 chars per token
        return total_chars // 4
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except requests.exceptions.RequestException:
            pass
        return []
    
    def pull_model(self, model: str) -> bool:
        """Pull a model to Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=300  # Model pulls can take a while
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def handle_error(self, error: Exception, retry_count: int = 0) -> ProviderError:
        """Handle Ollama-specific errors."""
        if isinstance(error, requests.exceptions.HTTPError):
            status_code = error.response.status_code
            
            # Model not found
            if status_code == 404:
                return ProviderError(
                    f"Model not found: {error}. Try pulling the model first.",
                    status_code=status_code,
                    retry=False
                )
            
            # Server errors (potentially retryable)
            elif status_code >= 500:
                return ProviderError(
                    f"Server error: {error}",
                    status_code=status_code,
                    retry=retry_count < 3
                )
            
            # Client errors
            else:
                return ProviderError(
                    f"Client error: {error}",
                    status_code=status_code,
                    retry=False
                )
        
        # Connection errors (Ollama server might be down)
        elif isinstance(error, requests.exceptions.ConnectionError):
            return ProviderError(
                f"Cannot connect to Ollama server at {self.base_url}: {error}",
                retry=retry_count < 2
            )
        
        # Timeout errors (retryable)
        elif isinstance(error, requests.exceptions.Timeout):
            return ProviderError(
                f"Request timeout: {error}",
                retry=retry_count < 3
            )
        
        # Other errors
        else:
            return ProviderError(str(error), retry=False)