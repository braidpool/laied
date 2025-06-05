"""
Anthropic provider implementation.
"""

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests

from .base import BaseProvider, ModelResponse, ProviderError


class AnthropicProvider(BaseProvider):
    """Anthropic API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.base_url:
            self.base_url = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com")
        
        # Ensure base URL doesn't end with /v1 (Anthropic uses different versioning)
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
        """Generate a completion using Anthropic API."""
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Add beta header for prompt caching and PDFs
        beta_features = []
        if any("cache_control" in msg for msg in messages):
            beta_features.append("prompt-caching-2024-07-31")
        beta_features.append("pdfs-2024-09-25")
        if beta_features:
            headers["anthropic-beta"] = ",".join(beta_features)
        
        # Convert messages to Anthropic format
        formatted_messages = self.format_messages(messages)
        system_message = None
        
        # Extract system message if present
        if formatted_messages and formatted_messages[0]["role"] == "system":
            system_message = formatted_messages[0]["content"]
            formatted_messages = formatted_messages[1:]
        
        data = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "stream": stream
        }
        
        if system_message:
            data["system"] = system_message
        
        if temperature is not None:
            data["temperature"] = temperature
        
        # Handle thinking tokens for Claude
        if "thinking" in kwargs:
            thinking_config = kwargs.pop("thinking")
            if thinking_config.get("type") == "enabled":
                data["thinking"] = thinking_config
        
        # Handle extra_body parameters
        if "extra_body" in kwargs:
            data.update(kwargs.pop("extra_body"))
        
        # Add any remaining kwargs
        data.update(kwargs)
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/messages",
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
        """Handle streaming response from Anthropic API."""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                try:
                    data = json.loads(data_str)
                    
                    if data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                            
                except json.JSONDecodeError:
                    continue
    
    def _handle_non_streaming_response(self, response: requests.Response) -> ModelResponse:
        """Handle non-streaming response from Anthropic API."""
        response_data = response.json()
        
        content = ""
        if "content" in response_data:
            for block in response_data["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")
        
        usage = response_data.get("usage", {})
        usage_dict = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        }
        
        return ModelResponse(
            content=content,
            finish_reason=response_data.get("stop_reason"),
            usage=usage_dict,
            model=response_data.get("model")
        )
    
    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Anthropic API."""
        formatted = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Anthropic uses "user" and "assistant" roles
            if role == "system":
                # System messages are handled separately in Anthropic API
                formatted.append({"role": "system", "content": content})
            elif role in ["user", "assistant"]:
                formatted.append({"role": role, "content": content})
            else:
                # Convert other roles to user
                formatted.append({"role": "user", "content": content})
        
        return formatted
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate Anthropic environment configuration."""
        if not self.api_key:
            return {
                "keys_in_environment": False,
                "missing_keys": ["ANTHROPIC_API_KEY"]
            }
        
        return {
            "keys_in_environment": True,
            "missing_keys": []
        }
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get Anthropic model information."""
        # Model info mapping for Anthropic models
        model_info = {
            "claude-sonnet-4-20250514": {
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 3.0e-06,
                "output_cost_per_token": 1.5e-05
            },
            "claude-opus-4-20250514": {
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 1.5e-05,
                "output_cost_per_token": 7.5e-05
            },
            "claude-3-5-sonnet-20241022": {
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 3.0e-06,
                "output_cost_per_token": 1.5e-05
            },
            "claude-3-5-sonnet-20240620": {
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 3.0e-06,
                "output_cost_per_token": 1.5e-05
            },
            "claude-3-5-haiku-20241022": {
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 1.0e-06,
                "output_cost_per_token": 5.0e-06
            },
            "claude-3-opus-20240229": {
                "max_input_tokens": 200000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 1.5e-05,
                "output_cost_per_token": 7.5e-05
            },
            "claude-3-sonnet-20240229": {
                "max_input_tokens": 200000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 3.0e-06,
                "output_cost_per_token": 1.5e-05
            },
            "claude-3-haiku-20240307": {
                "max_input_tokens": 200000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 2.5e-07,
                "output_cost_per_token": 1.25e-06
            }
        }
        
        # Try exact match first
        if model in model_info:
            return model_info[model]
        
        # Try partial matches
        for base_model, info in model_info.items():
            if model.startswith(base_model.split("-")[0:3]):
                return info
        
        # Default fallback for Claude models
        return {
            "max_input_tokens": 200000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 3.0e-06,
            "output_cost_per_token": 1.5e-05
        }
    
    def tokenize(self, text: str, model: str) -> List[int]:
        """Tokenize text (Claude doesn't provide public tokenizer)."""
        # Anthropic doesn't provide a public tokenizer
        # Rough estimation: ~3.5 chars per token for English text
        return list(range(len(text) // 4))
    
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count tokens in messages (estimation)."""
        # Rough estimation for Claude models
        total_chars = 0
        for message in messages:
            total_chars += len(message.get("content", ""))
            total_chars += 10  # Overhead for role and formatting
        
        # Rough estimation: ~3.5 chars per token
        return total_chars // 4
    
    def handle_error(self, error: Exception, retry_count: int = 0) -> ProviderError:
        """Handle Anthropic-specific errors."""
        if isinstance(error, requests.exceptions.HTTPError):
            status_code = error.response.status_code
            
            # Rate limiting
            if status_code == 429:
                return ProviderError(
                    f"Rate limit exceeded: {error}", 
                    status_code=status_code, 
                    retry=retry_count < 3
                )
            
            # Server errors (potentially retryable)
            elif status_code >= 500:
                return ProviderError(
                    f"Server error: {error}",
                    status_code=status_code,
                    retry=retry_count < 3
                )
            
            # Client errors (not retryable)
            else:
                return ProviderError(
                    f"Client error: {error}",
                    status_code=status_code,
                    retry=False
                )
        
        # Network errors (retryable)
        elif isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return ProviderError(
                f"Network error: {error}",
                retry=retry_count < 3
            )
        
        # Other errors
        else:
            return ProviderError(str(error), retry=False)