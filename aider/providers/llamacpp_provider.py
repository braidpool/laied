"""
LLamaCPP provider for direct API communication.
"""

import json
import requests
from typing import Dict, Any, List, Iterator, Optional

from .base import BaseProvider, ModelResponse, ProviderError


class LLamaCPPProvider(BaseProvider):
    """Provider for LLamaCPP server endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
    
    def _extract_base_url_from_model(self, model: str) -> str:
        """Extract base URL from model name if it contains URL patterns."""
        if any(pattern in model for pattern in ["localhost", "127.0.0.1", "://"]):
            # Handle cases like "localhost:8080/model" or "http://localhost:8080/model"
            if "://" in model:
                # Full URL provided
                parts = model.split("/")
                return f"{parts[0]}//{parts[1]}"
            elif ":" in model:
                # localhost:port format
                if "/" in model:
                    base_part = model.split("/")[0]
                else:
                    base_part = model
                return f"http://{base_part}"
        return self.base_url
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate LLamaCPP server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return {
                    "keys_in_environment": True,
                    "missing_keys": []
                }
        except requests.exceptions.RequestException:
            pass
        
        return {
            "keys_in_environment": False,
            "missing_keys": [f"LLamaCPP server at {self.base_url}"]
        }
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """Generate completion using LLamaCPP server."""
        try:
            # Convert messages to prompt format for LLamaCPP
            prompt = self._messages_to_prompt(messages)
            
            data = {
                "prompt": prompt,
                "stream": stream,
                "max_tokens": kwargs.get("max_tokens", 2048),
            }
            
            if temperature is not None:
                data["temperature"] = temperature
            
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=data,
                timeout=timeout or 60,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_non_streaming_response(response)
                
        except requests.exceptions.RequestException as e:
            raise self.handle_error(e)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def _handle_streaming_response(self, response: requests.Response) -> Iterator[str]:
        """Handle streaming response from LLamaCPP server."""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str.strip() == "[DONE]":
                    break
                    
                try:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        text = data["choices"][0].get("text", "")
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue
    
    def _handle_non_streaming_response(self, response: requests.Response) -> ModelResponse:
        """Handle non-streaming response from LLamaCPP server."""
        data = response.json()
        
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("text", "")
            finish_reason = data["choices"][0].get("finish_reason")
            
            usage = None
            if "usage" in data:
                usage = {
                    "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                    "completion_tokens": data["usage"].get("completion_tokens", 0),
                    "total_tokens": data["usage"].get("total_tokens", 0)
                }
            
            return ModelResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage
            )
        
        raise ProviderError("No valid response from LLamaCPP server")
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get LLamaCPP model information from /props endpoint."""
        # Extract base URL from model name if needed
        base_url = self._extract_base_url_from_model(model)
        
        try:
            # Try the /props endpoint first (recommended approach)
            response = requests.get(f"{base_url}/props", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract context size from default_generation_settings.n_ctx
                context_size = None
                if "default_generation_settings" in data:
                    gen_settings = data["default_generation_settings"]
                    context_size = gen_settings.get("n_ctx")
                
                if context_size:
                    info = {
                        "max_input_tokens": context_size,
                        "max_output_tokens": context_size,
                        "input_cost_per_token": 0.0,  # LLamaCPP is self-hosted
                        "output_cost_per_token": 0.0,
                        "_llamacpp_source": "props_endpoint",
                        "_llamacpp_raw_props": data
                    }
                    return info
            
        except requests.exceptions.RequestException:
            pass
        
        # Fallback to /v1/models endpoint 
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # LLamaCPP returns both 'models' and 'data' arrays
                # The 'data' array contains the detailed model info with n_ctx_train
                models_data = data.get("data", [])
                
                # Find the model by ID
                model_info = None
                for model_data in models_data:
                    if model_data.get("id") == model or model in model_data.get("id", ""):
                        model_info = model_data
                        break
                
                if not model_info and models_data:
                    # If no exact match, use the first available model
                    model_info = models_data[0]
                
                if model_info and "meta" in model_info:
                    meta = model_info["meta"]
                    
                    # Extract context size from n_ctx_train
                    context_size = meta.get("n_ctx_train")
                    if not context_size:
                        # Fallback to other possible fields
                        context_size = meta.get("n_ctx", 4096)
                    
                    # Extract other useful information
                    vocab_size = meta.get("n_vocab", 0)
                    param_count = meta.get("n_params", 0)
                    model_size = meta.get("size", 0)
                    
                    info = {
                        "max_input_tokens": context_size,
                        "max_output_tokens": context_size,
                        "input_cost_per_token": 0.0,  # LLamaCPP is self-hosted
                        "output_cost_per_token": 0.0,
                        # Additional metadata
                        "_llamacpp_source": "v1_models_endpoint",
                        "_llamacpp_vocab_size": vocab_size,
                        "_llamacpp_param_count": param_count,
                        "_llamacpp_model_size": model_size,
                        "_llamacpp_raw_data": model_info
                    }
                    
                    return info
                
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch LLamaCPP model info: {e}")
        
        # Default fallback
        return {
            "max_input_tokens": 4096,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "_llamacpp_source": "fallback"
        }
    
    def tokenize(self, text: str, model: str) -> List[int]:
        """Tokenize text using LLamaCPP server."""
        try:
            response = requests.post(
                f"{self.base_url}/v1/tokenize",
                json={"content": text},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("tokens", [])
        except requests.exceptions.RequestException:
            pass
        
        # Rough estimation fallback
        return [i for i in range(len(text) // 4)]
    
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count tokens for messages."""
        prompt = self._messages_to_prompt(messages)
        tokens = self.tokenize(prompt, model)
        return len(tokens)
    
    def handle_error(self, error: Exception) -> ProviderError:
        """Convert requests errors to ProviderError."""
        if isinstance(error, requests.exceptions.Timeout):
            return ProviderError(f"LLamaCPP server timeout: {error}", retry=True)
        elif isinstance(error, requests.exceptions.ConnectionError):
            return ProviderError(f"LLamaCPP server connection error: {error}", retry=True)
        else:
            return ProviderError(f"LLamaCPP server error: {error}", retry=False)