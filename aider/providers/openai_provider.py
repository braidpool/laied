"""
OpenAI provider implementation.
"""

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests

from .base import BaseProvider, ModelResponse, ProviderError


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.base_url.endswith("/v1"):
            if self.base_url.endswith("/"):
                self.base_url = self.base_url + "v1"
            else:
                self.base_url = self.base_url + "/v1"
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Union[ModelResponse, Iterator[str]]:
        """Generate a completion using OpenAI API."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Handle extra headers (e.g., for GitHub Copilot)
        if "extra_headers" in kwargs:
            headers.update(kwargs.pop("extra_headers"))
        
        data = {
            "model": model,
            "messages": self.format_messages(messages),
            "stream": stream
        }
        
        if temperature is not None:
            data["temperature"] = temperature
        
        # Handle tools/functions
        if "tools" in kwargs:
            data["tools"] = kwargs.pop("tools")
        if "tool_choice" in kwargs:
            data["tool_choice"] = kwargs.pop("tool_choice")
        
        # Handle extra_body parameters
        if "extra_body" in kwargs:
            data.update(kwargs.pop("extra_body"))
        
        # Add any remaining kwargs
        data.update(kwargs)
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
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
        """Handle streaming response from OpenAI API."""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                except json.JSONDecodeError:
                    continue
    
    def _handle_non_streaming_response(self, response: requests.Response) -> ModelResponse:
        """Handle non-streaming response from OpenAI API."""
        response_data = response.json()
        
        content = self._extract_content_from_response(response_data)
        usage = self._extract_usage_from_response(response_data)
        finish_reason = None
        
        if "choices" in response_data and response_data["choices"]:
            finish_reason = response_data["choices"][0].get("finish_reason")
        
        return ModelResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            model=response_data.get("model")
        )
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate OpenAI environment configuration."""
        if not self.api_key:
            return {
                "keys_in_environment": False,
                "missing_keys": ["OPENAI_API_KEY"]
            }
        
        return {
            "keys_in_environment": True,
            "missing_keys": []
        }
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get OpenAI model information."""
        # Model info mapping for common OpenAI models
        model_info = {
            "gpt-4o": {
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
                "input_cost_per_token": 5.00e-06,
                "output_cost_per_token": 1.5e-05
            },
            "gpt-4": {
                "max_input_tokens": 8192,
                "max_output_tokens": 8192,
                "input_cost_per_token": 3.0e-05,
                "output_cost_per_token": 6.0e-05
            },
            "gpt-4-turbo": {
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 1.0e-05,
                "output_cost_per_token": 3.0e-05
            },
            "gpt-3.5-turbo": {
                "max_input_tokens": 16385,
                "max_output_tokens": 4096,
                "input_cost_per_token": 1.5e-06,
                "output_cost_per_token": 2.0e-06
            },
            "o1-preview": {
                "max_input_tokens": 128000,
                "max_output_tokens": 32768,
                "input_cost_per_token": 1.5e-05,
                "output_cost_per_token": 6.0e-05
            },
            "o1-mini": {
                "max_input_tokens": 128000,
                "max_output_tokens": 65536,
                "input_cost_per_token": 3.0e-06,
                "output_cost_per_token": 1.2e-05
            }
        }
        
        # Try exact match first
        if model in model_info:
            return model_info[model]
        
        # Try partial matches for versioned models
        for base_model, info in model_info.items():
            if model.startswith(base_model):
                return info
        
        # Default fallback
        return {
            "max_input_tokens": 8192,
            "max_output_tokens": 4096,
            "input_cost_per_token": 1.0e-05,
            "output_cost_per_token": 3.0e-05
        }
    
    def tokenize(self, text: str, model: str) -> List[int]:
        """Tokenize text using tiktoken (if available) or estimate."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model.split("/")[-1])
            return encoding.encode(text)
        except ImportError:
            # Fallback: rough estimation of ~4 chars per token
            return list(range(len(text) // 4))
        except Exception:
            # If tiktoken doesn't recognize the model, use cl100k_base
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                return encoding.encode(text)
            except ImportError:
                return list(range(len(text) // 4))
    
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count tokens in messages using tiktoken or estimation."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model.split("/")[-1])
            
            # Rough approximation of OpenAI's token counting
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # If there's a name, the role is omitted
                        num_tokens += -1  # Role is always required and always 1 token
            
            num_tokens += 2  # Every reply is primed with <im_start>assistant
            return num_tokens
            
        except ImportError:
            # Fallback: rough estimation
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            return total_chars // 4
        except Exception:
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                total_chars = sum(len(msg.get("content", "")) for msg in messages)
                return len(encoding.encode(str(total_chars)))
            except ImportError:
                total_chars = sum(len(msg.get("content", "")) for msg in messages)
                return total_chars // 4
    
    def handle_error(self, error: Exception, retry_count: int = 0) -> ProviderError:
        """Handle OpenAI-specific errors."""
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
    
    def transcription(self, file, model: str = "whisper-1", prompt: str = "", language: str = None) -> str:
        """Transcribe audio using OpenAI Whisper API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": model
        }
        
        if prompt:
            data["prompt"] = prompt
        if language:
            data["language"] = language
        
        files = {
            "file": file
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
                timeout=30
            )
            response.raise_for_status()
            
            response_data = response.json()
            return response_data.get("text", "")
            
        except requests.exceptions.RequestException as e:
            raise self.handle_error(e)