import difflib
import hashlib
import importlib.resources
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import json5
import yaml
from PIL import Image

from aider import __version__
from aider.dump import dump  # noqa: F401
from aider.providers import provider_manager
from aider.openrouter import OpenRouterModelManager
from aider.sendchat import ensure_alternating_roles, sanity_check_messages
from aider.utils import check_pip_install_extra

# Import after defining the config module to avoid circular imports
try:
    from aider.config import AiderConfig, ProviderConfig
except ImportError:
    # Fallback for when config system is not yet available
    AiderConfig = None
    ProviderConfig = None

RETRY_TIMEOUT = 60

request_timeout = 600

DEFAULT_MODEL_NAME = "gpt-4o"
ANTHROPIC_BETA_HEADER = "prompt-caching-2024-07-31,pdfs-2024-09-25"

OPENAI_MODELS = """
o1
o1-preview
o1-mini
o3-mini
gpt-4
gpt-4o
gpt-4o-2024-05-13
gpt-4-turbo-preview
gpt-4-0314
gpt-4-0613
gpt-4-32k
gpt-4-32k-0314
gpt-4-32k-0613
gpt-4-turbo
gpt-4-turbo-2024-04-09
gpt-4-1106-preview
gpt-4-0125-preview
gpt-4-vision-preview
gpt-4-1106-vision-preview
gpt-4o-mini
gpt-4o-mini-2024-07-18
gpt-3.5-turbo
gpt-3.5-turbo-0301
gpt-3.5-turbo-0613
gpt-3.5-turbo-1106
gpt-3.5-turbo-0125
gpt-3.5-turbo-16k
gpt-3.5-turbo-16k-0613
"""

OPENAI_MODELS = [ln.strip() for ln in OPENAI_MODELS.splitlines() if ln.strip()]

ANTHROPIC_MODELS = """
claude-2
claude-2.1
claude-3-haiku-20240307
claude-3-5-haiku-20241022
claude-3-opus-20240229
claude-3-sonnet-20240229
claude-3-5-sonnet-20240620
claude-3-5-sonnet-20241022
claude-sonnet-4-20250514
claude-opus-4-20250514
"""

ANTHROPIC_MODELS = [ln.strip() for ln in ANTHROPIC_MODELS.splitlines() if ln.strip()]

# Mapping of model aliases to their canonical names
MODEL_ALIASES = {
    # Claude models
    "sonnet": "anthropic/claude-sonnet-4-20250514",
    "haiku": "claude-3-5-haiku-20241022",
    "opus": "claude-opus-4-20250514",
    # GPT models
    "4": "gpt-4-0613",
    "4o": "gpt-4o",
    "4-turbo": "gpt-4-1106-preview",
    "35turbo": "gpt-3.5-turbo",
    "35-turbo": "gpt-3.5-turbo",
    "3": "gpt-3.5-turbo",
    # Other models
    "deepseek": "deepseek/deepseek-chat",
    "flash": "gemini/gemini-2.5-flash-preview-04-17",
    "quasar": "openrouter/openrouter/quasar-alpha",
    "r1": "deepseek/deepseek-reasoner",
    "gemini-2.5-pro": "gemini/gemini-2.5-pro-preview-05-06",
    "gemini": "gemini/gemini-2.5-pro-preview-05-06",
    "gemini-exp": "gemini/gemini-2.5-pro-exp-03-25",
    "grok3": "xai/grok-3-beta",
    "optimus": "openrouter/openrouter/optimus-alpha",
}
# Model metadata loaded from resources and user's files.


@dataclass
class ModelSettings:
    # Model class needs to have each of these as well
    name: str
    edit_format: str = "whole"
    weak_model_name: Optional[str] = None
    use_repo_map: bool = False
    send_undo_reply: bool = False
    lazy: bool = False
    overeager: bool = False
    reminder: str = "user"
    examples_as_sys_msg: bool = False
    extra_params: Optional[dict] = None
    cache_control: bool = False
    caches_by_default: bool = False
    use_system_prompt: bool = True
    use_temperature: Union[bool, float] = True
    streaming: bool = True
    editor_model_name: Optional[str] = None
    editor_edit_format: Optional[str] = None
    reasoning_tag: Optional[str] = None
    remove_reasoning: Optional[str] = None  # Deprecated alias for reasoning_tag
    system_prompt_prefix: Optional[str] = None
    accepts_settings: Optional[list] = None


# Load model settings from package resource
MODEL_SETTINGS = []
with importlib.resources.open_text("aider.resources", "model-settings.yml") as f:
    model_settings_list = yaml.safe_load(f)
    for model_settings_dict in model_settings_list:
        MODEL_SETTINGS.append(ModelSettings(**model_settings_dict))


class ModelInfoManager:
    MODEL_INFO_URL = (
        "https://raw.githubusercontent.com/BerriAI/litellm/main/"
        "model_prices_and_context_window.json"
    )
    CACHE_TTL = 60 * 60 * 24  # 24 hours

    def __init__(self):
        self.cache_dir = Path.home() / ".aider" / "caches"
        self.cache_file = self.cache_dir / "model_prices_and_context_window.json"
        self.content = None
        self.local_model_metadata = {}
        self.verify_ssl = True
        self._cache_loaded = False

        # Manager for the cached OpenRouter model database
        self.openrouter_manager = OpenRouterModelManager()

    def set_verify_ssl(self, verify_ssl):
        self.verify_ssl = verify_ssl
        if hasattr(self, "openrouter_manager"):
            self.openrouter_manager.set_verify_ssl(verify_ssl)

    def _load_cache(self):
        if self._cache_loaded:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.cache_file.exists():
                cache_age = time.time() - self.cache_file.stat().st_mtime
                if cache_age < self.CACHE_TTL:
                    try:
                        self.content = json.loads(self.cache_file.read_text())
                    except json.JSONDecodeError:
                        # If the cache file is corrupted, treat it as missing
                        self.content = None
        except OSError:
            pass

        self._cache_loaded = True

    def _update_cache(self):
        try:
            import requests

            # Respect the --no-verify-ssl switch
            response = requests.get(self.MODEL_INFO_URL, timeout=5, verify=self.verify_ssl)
            if response.status_code == 200:
                self.content = response.json()
                try:
                    self.cache_file.write_text(json.dumps(self.content, indent=4))
                except OSError:
                    pass
        except Exception as ex:
            print(str(ex))
            try:
                # Save empty dict to cache file on failure
                self.cache_file.write_text("{}")
            except OSError:
                pass

    def get_model_from_cached_json_db(self, model):
        data = self.local_model_metadata.get(model)
        if data:
            return data

        # Ensure cache is loaded before checking content
        self._load_cache()

        if not self.content:
            self._update_cache()

        if not self.content:
            return dict()

        info = self.content.get(model, dict())
        if info:
            return info

        pieces = model.split("/")
        if len(pieces) == 2:
            info = self.content.get(pieces[1])
            if info and info.get("litellm_provider") == pieces[0]:
                return info

        return dict()

    def get_model_info(self, model):
        # For dynamic/local models, always query the API first
        is_dynamic_model = self._is_dynamic_model(model)
        
        provider_info = None
        if is_dynamic_model:
            # For Ollama/LLamaCPP, always query live endpoint first
            try:
                provider_info = provider_manager.get_model_info(model)
                if provider_info:
                    return provider_info
            except Exception as ex:
                if "model_prices_and_context_window.json" not in str(ex):
                    print(f"Warning: Could not query dynamic model info for {model}: {ex}")
        
        # Fallback to static cache for non-dynamic models or if API query failed
        cached_info = self.get_model_from_cached_json_db(model)
        
        # For non-dynamic models, try provider API if no cached info
        if not cached_info and not is_dynamic_model:
            try:
                provider_info = provider_manager.get_model_info(model)
            except Exception as ex:
                if "model_prices_and_context_window.json" not in str(ex):
                    print(str(ex))

        if provider_info:
            return provider_info

        if not cached_info and model.startswith("openrouter/"):
            # First try using the locally cached OpenRouter model database
            openrouter_info = self.openrouter_manager.get_model_info(model)
            if openrouter_info:
                return openrouter_info

            # Fallback to legacy web-scraping if the API cache does not contain the model
            openrouter_info = self.fetch_openrouter_model_info(model)
            if openrouter_info:
                return openrouter_info

        return cached_info

    def _is_dynamic_model(self, model: str) -> bool:
        """Check if model should use dynamic API queries instead of static metadata."""
        model_lower = model.lower()
        
        # Ollama models
        if any(pattern in model_lower for pattern in ["ollama", "llama", "mixtral", "mistral", "qwen", "phi", "gemma"]):
            return True
        
        # LLamaCPP models (often served on localhost or custom ports)
        if any(pattern in model for pattern in ["localhost", "127.0.0.1", ":8080", ":8000"]):
            return True
        
        # Model names that typically indicate local/self-hosted models
        if any(pattern in model_lower for pattern in ["vicuna", "alpaca", "orca", "wizard", "nous-"]):
            return True
        
        return False

    def fetch_openrouter_model_info(self, model):
        """
        Fetch model info by scraping the openrouter model page.
        Expected URL: https://openrouter.ai/<model_route>
        Example: openrouter/qwen/qwen-2.5-72b-instruct:free
        Returns a dict with keys: max_tokens, max_input_tokens, max_output_tokens,
        input_cost_per_token, output_cost_per_token.
        """
        url_part = model[len("openrouter/") :]
        url = "https://openrouter.ai/" + url_part
        try:
            import requests

            response = requests.get(url, timeout=5, verify=self.verify_ssl)
            if response.status_code != 200:
                return {}
            html = response.text
            import re

            if re.search(
                rf"The model\s*.*{re.escape(url_part)}.* is not available", html, re.IGNORECASE
            ):
                print(f"\033[91mError: Model '{url_part}' is not available\033[0m")
                return {}
            text = re.sub(r"<[^>]+>", " ", html)
            context_match = re.search(r"([\d,]+)\s*context", text)
            if context_match:
                context_str = context_match.group(1).replace(",", "")
                context_size = int(context_str)
            else:
                context_size = None
            input_cost_match = re.search(r"\$\s*([\d.]+)\s*/M input tokens", text, re.IGNORECASE)
            output_cost_match = re.search(r"\$\s*([\d.]+)\s*/M output tokens", text, re.IGNORECASE)
            input_cost = float(input_cost_match.group(1)) / 1000000 if input_cost_match else None
            output_cost = float(output_cost_match.group(1)) / 1000000 if output_cost_match else None
            if context_size is None or input_cost is None or output_cost is None:
                return {}
            params = {
                "max_input_tokens": context_size,
                "max_tokens": context_size,
                "max_output_tokens": context_size,
                "input_cost_per_token": input_cost,
                "output_cost_per_token": output_cost,
            }
            return params
        except Exception as e:
            print("Error fetching openrouter info:", str(e))
            return {}


model_info_manager = ModelInfoManager()


class Model(ModelSettings):
    def __init__(
        self, model, weak_model=None, editor_model=None, editor_edit_format=None, verbose=False
    ):
        # Use endpoint-aware model resolution if config is available
        if endpoint_model_manager.config:
            resolved_model, provider_config = endpoint_model_manager.resolve_model_spec(model)
            if provider_config:
                # Set up environment variables for this endpoint
                endpoint_model_manager.setup_environment_for_model(model)
                # Use the provider type + model name for LiteLLM compatibility
                model = f"{provider_config.type}/{resolved_model}"
            else:
                model = resolved_model
        else:
            # Fallback to legacy alias resolution
            model = MODEL_ALIASES.get(model, model)

        self.name = model
        self.verbose = verbose

        self.max_chat_history_tokens = 1024
        self.weak_model = None
        self.editor_model = None

        # Find the extra settings
        self.extra_model_settings = next(
            (ms for ms in MODEL_SETTINGS if ms.name == "aider/extra_params"), None
        )

        self.info = self.get_model_info(model)

        # Are all needed keys/params available?
        res = self.validate_environment()
        self.missing_keys = res.get("missing_keys")
        self.keys_in_environment = res.get("keys_in_environment")

        max_input_tokens = self.info.get("max_input_tokens") or 0
        # Calculate max_chat_history_tokens as 1/16th of max_input_tokens,
        # with minimum 1k and maximum 8k
        self.max_chat_history_tokens = min(max(max_input_tokens / 16, 1024), 8192)

        self.configure_model_settings(model)
        
        # Apply any configuration overrides
        self._apply_config_overrides()
        
        # Display model parameters on startup if verbose
        if verbose:
            self._display_model_parameters()
        if weak_model is False:
            self.weak_model_name = None
        else:
            self.get_weak_model(weak_model)

        if editor_model is False:
            self.editor_model_name = None
        else:
            self.get_editor_model(editor_model, editor_edit_format)

    def get_model_info(self, model):
        return model_info_manager.get_model_info(model)

    def _copy_fields(self, source):
        """Helper to copy fields from a ModelSettings instance to self"""
        for field in fields(ModelSettings):
            val = getattr(source, field.name)
            setattr(self, field.name, val)

        # Handle backward compatibility: if remove_reasoning is set but reasoning_tag isn't,
        # use remove_reasoning's value for reasoning_tag
        if self.reasoning_tag is None and self.remove_reasoning is not None:
            self.reasoning_tag = self.remove_reasoning

    def configure_model_settings(self, model):
        # Look for exact model match
        exact_match = False
        for ms in MODEL_SETTINGS:
            # direct match, or match "provider/<model>"
            if model == ms.name:
                self._copy_fields(ms)
                exact_match = True
                break  # Continue to apply overrides

        # Initialize accepts_settings if it's None
        if self.accepts_settings is None:
            self.accepts_settings = []

        model = model.lower()

        # If no exact match, try generic settings
        if not exact_match:
            self.apply_generic_model_settings(model)

        # Apply override settings last if they exist
        if (
            self.extra_model_settings
            and self.extra_model_settings.extra_params
            and self.extra_model_settings.name == "aider/extra_params"
        ):
            # Initialize extra_params if it doesn't exist
            if not self.extra_params:
                self.extra_params = {}

            # Deep merge the extra_params dicts
            for key, value in self.extra_model_settings.extra_params.items():
                if isinstance(value, dict) and isinstance(self.extra_params.get(key), dict):
                    # For nested dicts, merge recursively
                    self.extra_params[key] = {**self.extra_params[key], **value}
                else:
                    # For non-dict values, simply update
                    self.extra_params[key] = value

        # Ensure OpenRouter models accept thinking_tokens and reasoning_effort
        if self.name.startswith("openrouter/"):
            if self.accepts_settings is None:
                self.accepts_settings = []
            if "thinking_tokens" not in self.accepts_settings:
                self.accepts_settings.append("thinking_tokens")
            if "reasoning_effort" not in self.accepts_settings:
                self.accepts_settings.append("reasoning_effort")

    def apply_generic_model_settings(self, model):
        if "/o3-mini" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.use_temperature = False
            self.system_prompt_prefix = "Formatting re-enabled. "
            self.system_prompt_prefix = "Formatting re-enabled. "
            if "reasoning_effort" not in self.accepts_settings:
                self.accepts_settings.append("reasoning_effort")
            return  # <--

        if "gpt-4.1-mini" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.reminder = "sys"
            self.examples_as_sys_msg = False
            return  # <--

        if "gpt-4.1" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.reminder = "sys"
            self.examples_as_sys_msg = False
            return  # <--

        if "/o1-mini" in model:
            self.use_repo_map = True
            self.use_temperature = False
            self.use_system_prompt = False
            return  # <--

        if "/o1-preview" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.use_temperature = False
            self.use_system_prompt = False
            return  # <--

        if "/o1" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.use_temperature = False
            self.streaming = False
            self.system_prompt_prefix = "Formatting re-enabled. "
            if "reasoning_effort" not in self.accepts_settings:
                self.accepts_settings.append("reasoning_effort")
            return  # <--

        if "deepseek" in model and "v3" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.reminder = "sys"
            self.examples_as_sys_msg = True
            return  # <--

        if "deepseek" in model and ("r1" in model or "reasoning" in model):
            self.edit_format = "diff"
            self.use_repo_map = True
            self.examples_as_sys_msg = True
            self.use_temperature = False
            self.reasoning_tag = "think"
            return  # <--

        if ("llama3" in model or "llama-3" in model) and "70b" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.send_undo_reply = True
            self.examples_as_sys_msg = True
            return  # <--

        if "gpt-4-turbo" in model or ("gpt-4-" in model and "-preview" in model):
            self.edit_format = "udiff"
            self.use_repo_map = True
            self.send_undo_reply = True
            return  # <--

        if "gpt-4" in model or "claude-3-opus" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.send_undo_reply = True
            return  # <--

        if "gpt-3.5" in model or "gpt-4" in model:
            self.reminder = "sys"
            return  # <--

        if "3-7-sonnet" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.examples_as_sys_msg = True
            self.reminder = "user"
            if "thinking_tokens" not in self.accepts_settings:
                self.accepts_settings.append("thinking_tokens")
            return  # <--

        if "3.5-sonnet" in model or "3-5-sonnet" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.examples_as_sys_msg = True
            self.reminder = "user"
            return  # <--

        if model.startswith("o1-") or "/o1-" in model:
            self.use_system_prompt = False
            self.use_temperature = False
            return  # <--

        if (
            "qwen" in model
            and "coder" in model
            and ("2.5" in model or "2-5" in model)
            and "32b" in model
        ):
            self.edit_format = "diff"
            self.editor_edit_format = "editor-diff"
            self.use_repo_map = True
            return  # <--

        if "qwq" in model and "32b" in model and "preview" not in model:
            self.edit_format = "diff"
            self.editor_edit_format = "editor-diff"
            self.use_repo_map = True
            self.reasoning_tag = "think"
            self.examples_as_sys_msg = True
            self.use_temperature = 0.6
            self.extra_params = dict(top_p=0.95)
            return  # <--

        if "qwen3" in model and "235b" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.system_prompt_prefix = "/no_think"
            self.use_temperature = 0.7
            self.extra_params = {"top_p": 0.8, "top_k": 20, "min_p": 0.0}
            return  # <--

        # use the defaults
        if self.edit_format == "diff":
            self.use_repo_map = True
            return  # <--

    def __str__(self):
        return self.name

    def get_weak_model(self, provided_weak_model_name):
        # If weak_model_name is provided, override the model settings
        if provided_weak_model_name:
            self.weak_model_name = provided_weak_model_name

        if not self.weak_model_name:
            self.weak_model = self
            return

        if self.weak_model_name == self.name:
            self.weak_model = self
            return

        self.weak_model = Model(
            self.weak_model_name,
            weak_model=False,
        )
        return self.weak_model

    def commit_message_models(self):
        return [self.weak_model, self]

    def get_editor_model(self, provided_editor_model_name, editor_edit_format):
        # If editor_model_name is provided, override the model settings
        if provided_editor_model_name:
            self.editor_model_name = provided_editor_model_name
        if editor_edit_format:
            self.editor_edit_format = editor_edit_format

        if not self.editor_model_name or self.editor_model_name == self.name:
            self.editor_model = self
        else:
            self.editor_model = Model(
                self.editor_model_name,
                editor_model=False,
            )

        if not self.editor_edit_format:
            self.editor_edit_format = self.editor_model.edit_format
            if self.editor_edit_format in ("diff", "whole", "diff-fenced"):
                self.editor_edit_format = "editor-" + self.editor_edit_format

        return self.editor_model

    def tokenizer(self, text):
        return provider_manager.tokenize(text=text, model=self.name)

    def token_count(self, messages):
        if type(messages) is list:
            try:
                return provider_manager.count_tokens(messages=messages, model=self.name)
            except Exception as err:
                print(f"Unable to count tokens: {err}")
                return 0

        if type(messages) is str:
            msgs = messages
        else:
            msgs = json.dumps(messages)

        try:
            tokens = self.tokenizer(msgs)
            return len(tokens)
        except Exception as err:
            print(f"Unable to count tokens: {err}")
            return 0

    def token_count_for_image(self, fname):
        """
        Calculate the token cost for an image assuming high detail.
        The token cost is determined by the size of the image.
        :param fname: The filename of the image.
        :return: The token cost for the image.
        """
        width, height = self.get_image_size(fname)

        # If the image is larger than 2048 in any dimension, scale it down to fit within 2048x2048
        max_dimension = max(width, height)
        if max_dimension > 2048:
            scale_factor = 2048 / max_dimension
            width = int(width * scale_factor)
            height = int(height * scale_factor)

        # Scale the image such that the shortest side is 768 pixels long
        min_dimension = min(width, height)
        scale_factor = 768 / min_dimension
        width = int(width * scale_factor)
        height = int(height * scale_factor)

        # Calculate the number of 512x512 tiles needed to cover the image
        tiles_width = math.ceil(width / 512)
        tiles_height = math.ceil(height / 512)
        num_tiles = tiles_width * tiles_height

        # Each tile costs 170 tokens, and there's an additional fixed cost of 85 tokens
        token_cost = num_tiles * 170 + 85
        return token_cost

    def get_image_size(self, fname):
        """
        Retrieve the size of an image.
        :param fname: The filename of the image.
        :return: A tuple (width, height) representing the image size in pixels.
        """
        with Image.open(fname) as img:
            return img.size

    def fast_validate_environment(self):
        """Fast path for common models. Avoids forcing litellm import."""

        model = self.name

        pieces = model.split("/")
        if len(pieces) > 1:
            provider = pieces[0]
        else:
            provider = None

        keymap = dict(
            openrouter="OPENROUTER_API_KEY",
            openai="OPENAI_API_KEY",
            deepseek="DEEPSEEK_API_KEY",
            gemini="GEMINI_API_KEY",
            anthropic="ANTHROPIC_API_KEY",
            groq="GROQ_API_KEY",
            fireworks_ai="FIREWORKS_API_KEY",
        )
        var = None
        if model in OPENAI_MODELS:
            var = "OPENAI_API_KEY"
        elif model in ANTHROPIC_MODELS:
            var = "ANTHROPIC_API_KEY"
        else:
            var = keymap.get(provider)

        if var and os.environ.get(var):
            return dict(keys_in_environment=[var], missing_keys=[])

    def validate_environment(self):
        res = self.fast_validate_environment()
        if res:
            return res

        # https://github.com/BerriAI/litellm/issues/3190

        model = self.name
        res = provider_manager.validate_environment(model)

        # If missing AWS credential keys but AWS_PROFILE is set, consider AWS credentials valid
        if res["missing_keys"] and any(
            key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] for key in res["missing_keys"]
        ):
            if model.startswith("bedrock/") or model.startswith("us.anthropic."):
                if os.environ.get("AWS_PROFILE"):
                    res["missing_keys"] = [
                        k
                        for k in res["missing_keys"]
                        if k not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
                    ]
                    if not res["missing_keys"]:
                        res["keys_in_environment"] = True

        if res["keys_in_environment"]:
            return res
        if res["missing_keys"]:
            return res

        provider = self.info.get("litellm_provider", "").lower()
        if provider == "cohere_chat":
            return validate_variables(["COHERE_API_KEY"])
        if provider == "gemini":
            return validate_variables(["GEMINI_API_KEY"])
        if provider == "groq":
            return validate_variables(["GROQ_API_KEY"])

        return res

    def get_repo_map_tokens(self):
        map_tokens = 1024
        max_inp_tokens = self.info.get("max_input_tokens")
        if max_inp_tokens:
            map_tokens = max_inp_tokens / 8
            map_tokens = min(map_tokens, 4096)
            map_tokens = max(map_tokens, 1024)
        return map_tokens

    def set_reasoning_effort(self, effort):
        """Set the reasoning effort parameter for models that support it"""
        if effort is not None:
            if self.name.startswith("openrouter/"):
                if not self.extra_params:
                    self.extra_params = {}
                if "extra_body" not in self.extra_params:
                    self.extra_params["extra_body"] = {}
                self.extra_params["extra_body"]["reasoning"] = {"effort": effort}
            else:
                if not self.extra_params:
                    self.extra_params = {}
                if "extra_body" not in self.extra_params:
                    self.extra_params["extra_body"] = {}
                self.extra_params["extra_body"]["reasoning_effort"] = effort

    def parse_token_value(self, value):
        """
        Parse a token value string into an integer.
        Accepts formats: 8096, "8k", "10.5k", "0.5M", "10K", etc.

        Args:
            value: String or int token value

        Returns:
            Integer token value
        """
        if isinstance(value, int):
            return value

        if not isinstance(value, str):
            return int(value)  # Try to convert to int

        value = value.strip().upper()

        if value.endswith("K"):
            multiplier = 1024
            value = value[:-1]
        elif value.endswith("M"):
            multiplier = 1024 * 1024
            value = value[:-1]
        else:
            multiplier = 1

        # Convert to float first to handle decimal values like "10.5k"
        return int(float(value) * multiplier)

    def set_thinking_tokens(self, value):
        """
        Set the thinking token budget for models that support it.
        Accepts formats: 8096, "8k", "10.5k", "0.5M", "10K", etc.
        """
        if value is not None:
            num_tokens = self.parse_token_value(value)
            self.use_temperature = False
            if not self.extra_params:
                self.extra_params = {}

            # OpenRouter models use 'reasoning' instead of 'thinking'
            if self.name.startswith("openrouter/"):
                if "extra_body" not in self.extra_params:
                    self.extra_params["extra_body"] = {}
                self.extra_params["extra_body"]["reasoning"] = {"max_tokens": num_tokens}
            else:
                self.extra_params["thinking"] = {"type": "enabled", "budget_tokens": num_tokens}

    def get_raw_thinking_tokens(self):
        """Get formatted thinking token budget if available"""
        budget = None

        if self.extra_params:
            # Check for OpenRouter reasoning format
            if self.name.startswith("openrouter/"):
                if (
                    "extra_body" in self.extra_params
                    and "reasoning" in self.extra_params["extra_body"]
                    and "max_tokens" in self.extra_params["extra_body"]["reasoning"]
                ):
                    budget = self.extra_params["extra_body"]["reasoning"]["max_tokens"]
            # Check for standard thinking format
            elif (
                "thinking" in self.extra_params and "budget_tokens" in self.extra_params["thinking"]
            ):
                budget = self.extra_params["thinking"]["budget_tokens"]

        return budget

    def get_thinking_tokens(self):
        budget = self.get_raw_thinking_tokens()

        if budget is not None:
            # Format as xx.yK for thousands, xx.yM for millions
            if budget >= 1024 * 1024:
                value = budget / (1024 * 1024)
                if value == int(value):
                    return f"{int(value)}M"
                else:
                    return f"{value:.1f}M"
            else:
                value = budget / 1024
                if value == int(value):
                    return f"{int(value)}k"
                else:
                    return f"{value:.1f}k"
        return None

    def get_reasoning_effort(self):
        """Get reasoning effort value if available"""
        if self.extra_params:
            # Check for OpenRouter reasoning format
            if self.name.startswith("openrouter/"):
                if (
                    "extra_body" in self.extra_params
                    and "reasoning" in self.extra_params["extra_body"]
                    and "effort" in self.extra_params["extra_body"]["reasoning"]
                ):
                    return self.extra_params["extra_body"]["reasoning"]["effort"]
            # Check for standard reasoning_effort format (e.g. in extra_body)
            elif (
                "extra_body" in self.extra_params
                and "reasoning_effort" in self.extra_params["extra_body"]
            ):
                return self.extra_params["extra_body"]["reasoning_effort"]
        return None

    def is_deepseek_r1(self):
        name = self.name.lower()
        if "deepseek" not in name:
            return
        return "r1" in name or "reasoner" in name

    def is_ollama(self):
        return self.name.startswith("ollama/") or self.name.startswith("ollama_chat/")

    def github_copilot_token_to_open_ai_key(self, extra_headers):
        # check to see if there's an openai api key
        # If so, check to see if it's expire
        openai_api_key = "OPENAI_API_KEY"

        if openai_api_key not in os.environ or (
            int(dict(x.split("=") for x in os.environ[openai_api_key].split(";"))["exp"])
            < int(datetime.now().timestamp())
        ):
            import requests

            class GitHubCopilotTokenError(Exception):
                """Custom exception for GitHub Copilot token-related errors."""

                pass

            # Validate GitHub Copilot token exists
            if "GITHUB_COPILOT_TOKEN" not in os.environ:
                raise KeyError("GITHUB_COPILOT_TOKEN environment variable not found")

            github_token = os.environ["GITHUB_COPILOT_TOKEN"]
            if not github_token.strip():
                raise KeyError("GITHUB_COPILOT_TOKEN environment variable is empty")

            headers = {
                "Authorization": f"Bearer {os.environ['GITHUB_COPILOT_TOKEN']}",
                "Editor-Version": extra_headers["Editor-Version"],
                "Copilot-Integration-Id": extra_headers["Copilot-Integration-Id"],
                "Content-Type": "application/json",
            }

            url = "https://api.github.com/copilot_internal/v2/token"
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                safe_headers = {k: v for k, v in headers.items() if k != "Authorization"}
                token_preview = github_token[:5] + "..." if len(github_token) >= 5 else github_token
                safe_headers["Authorization"] = f"Bearer {token_preview}"
                raise GitHubCopilotTokenError(
                    f"GitHub Copilot API request failed (Status: {res.status_code})\n"
                    f"URL: {url}\n"
                    f"Headers: {json.dumps(safe_headers, indent=2)}\n"
                    f"JSON: {res.text}"
                )

            response_data = res.json()
            token = response_data.get("token")
            if not token:
                raise GitHubCopilotTokenError("Response missing 'token' field")

            os.environ[openai_api_key] = token

    def send_completion(self, messages, functions, stream, temperature=None):
        if os.environ.get("AIDER_SANITY_CHECK_TURNS"):
            sanity_check_messages(messages)

        if self.is_deepseek_r1():
            messages = ensure_alternating_roles(messages)

        kwargs = dict(
            model=self.name,
            stream=stream,
        )

        if self.use_temperature is not False:
            if temperature is None:
                if isinstance(self.use_temperature, bool):
                    temperature = 0
                else:
                    temperature = float(self.use_temperature)

            kwargs["temperature"] = temperature

        if functions is not None:
            function = functions[0]
            kwargs["tools"] = [dict(type="function", function=function)]
            kwargs["tool_choice"] = {"type": "function", "function": {"name": function["name"]}}
        if self.extra_params:
            kwargs.update(self.extra_params)
        if self.is_ollama() and "num_ctx" not in kwargs:
            num_ctx = int(self.token_count(messages) * 1.25) + 8192
            kwargs["num_ctx"] = num_ctx
        key = json.dumps(kwargs, sort_keys=True).encode()

        # dump(kwargs)

        hash_object = hashlib.sha1(key)
        if "timeout" not in kwargs:
            kwargs["timeout"] = request_timeout
        if self.verbose:
            dump(kwargs)
        kwargs["messages"] = messages

        # Are we using github copilot?
        if "GITHUB_COPILOT_TOKEN" in os.environ:
            if "extra_headers" not in kwargs:
                kwargs["extra_headers"] = {
                    "Editor-Version": f"aider/{__version__}",
                    "Copilot-Integration-Id": "vscode-chat",
                }

            self.github_copilot_token_to_open_ai_key(kwargs["extra_headers"])

        res = provider_manager.completion(**kwargs)
        return hash_object, res

    def simple_send_with_retries(self, messages):
        from aider.providers import ProviderError

        if "deepseek-reasoner" in self.name:
            messages = ensure_alternating_roles(messages)
        retry_delay = 0.125

        if self.verbose:
            dump(messages)

        retry_count = 0
        while True:
            try:
                kwargs = {
                    "messages": messages,
                    "functions": None,
                    "stream": False,
                }

                _hash, response = self.send_completion(**kwargs)
                if not response or not hasattr(response, "choices") or not response.choices:
                    return None
                res = response.choices[0].message.content
                from aider.reasoning_tags import remove_reasoning_content

                return remove_reasoning_content(res, self.reasoning_tag)

            except ProviderError as err:
                print(str(err))
                should_retry = err.retry and retry_count < 3
                if should_retry:
                    retry_delay *= 2
                    if retry_delay > RETRY_TIMEOUT:
                        should_retry = False
                if not should_retry:
                    return None
                print(f"Retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                retry_count += 1
                continue
            except AttributeError:
                return None

    def _display_model_parameters(self):
        """Display key model parameters on startup."""
        print(f"\n=== Model Parameters for {self.name} ===")
        
        # Core capacity parameters
        max_input = self.info.get("max_input_tokens", "unknown")
        max_output = self.info.get("max_output_tokens", "unknown")
        print(f"Max input tokens: {max_input:,}" if isinstance(max_input, int) else f"Max input tokens: {max_input}")
        print(f"Max output tokens: {max_output:,}" if isinstance(max_output, int) else f"Max output tokens: {max_output}")
        print(f"Max chat history tokens: {self.max_chat_history_tokens:,}")
        
        # Configuration parameters
        if hasattr(self, 'use_temperature'):
            if self.use_temperature is False:
                print("Temperature: disabled")
            elif isinstance(self.use_temperature, (int, float)):
                print(f"Temperature: {self.use_temperature}")
            else:
                print("Temperature: default (model-dependent)")
        
        print(f"Streaming: {'enabled' if self.streaming else 'disabled'}")
        print(f"System prompts: {'supported' if self.use_system_prompt else 'not supported'}")
        print(f"Prompt caching: {'default' if self.caches_by_default else 'disabled'}")
        
        # Cost parameters
        input_cost = self.info.get("input_cost_per_token", 0)
        output_cost = self.info.get("output_cost_per_token", 0)
        if input_cost > 0 or output_cost > 0:
            print(f"Input cost: ${input_cost:.6f} per token")
            print(f"Output cost: ${output_cost:.6f} per token")
        else:
            print("Cost: free/self-hosted")
        
        # Ollama-specific debugging info
        if "_detected_context_size" in self.info:
            detected_size = self.info["_detected_context_size"]
            print(f"Detected context size: {detected_size:,}" if detected_size else "Detected context size: not found")
        
        print("=" * 50)

    def _apply_config_overrides(self):
        """Apply configuration-based parameter overrides."""
        # This will be enhanced when config integration is completed
        # For now, it's a placeholder for future config-based overrides
        pass


class EndpointAwareModelManager:
    """
    Manages endpoint-aware model resolution and environment setup.

    This class bridges the new unified configuration system with the existing
    Model class, providing endpoint-specific model resolution and environment
    management for multi-endpoint support.
    """

    def __init__(self, config: Optional["AiderConfig"] = None):
        self.config = config
        self._model_cache = {}

    def set_config(self, config: "AiderConfig"):
        """Set the configuration and clear model cache."""
        self.config = config
        self._model_cache.clear()

    def resolve_model_spec(self, model_spec: str) -> tuple[str, Optional["ProviderConfig"]]:
        """
        Resolve a model specification to (model_name, provider_config).

        Args:
            model_spec: Model specification (alias, endpoint/model, or model name)

        Returns:
            Tuple of (resolved_model_name, provider_config) or (model_spec, None) if not found
        """
        if not self.config:
            # Fallback to legacy behavior when no config is available
            return MODEL_ALIASES.get(model_spec, model_spec), None

        resolved = self.config.resolve_model_name(model_spec)
        if not resolved:
            # Fallback to legacy aliases if config resolution fails
            legacy_resolved = MODEL_ALIASES.get(model_spec, model_spec)
            return legacy_resolved, None

        # Extract endpoint and model name
        if "/" in resolved:
            endpoint_name, model_name = resolved.split("/", 1)
            provider_config = self.config.providers.get(endpoint_name)
            return model_name, provider_config
        else:
            return resolved, None

    def setup_environment_for_model(self, model_spec: str) -> dict:
        """
        Set up environment variables for a specific model endpoint.

        Returns a dictionary of environment variables that were set.
        """
        if not self.config:
            return {}

        model_name, provider_config = self.resolve_model_spec(model_spec)

        if not provider_config:
            return {}

        env_vars = {}

        # Set provider-specific environment variables
        provider_type = provider_config.type.upper()

        if provider_config.api_key:
            env_key = f"{provider_type}_API_KEY"
            os.environ[env_key] = provider_config.api_key
            env_vars[env_key] = provider_config.api_key

        if provider_config.base_url:
            # Map provider types to their base URL environment variables
            base_url_map = {
                "OPENAI": "OPENAI_API_BASE",
                "OLLAMA": "OLLAMA_HOST",
                "ANTHROPIC": "ANTHROPIC_API_BASE",
                "GROQ": "GROQ_API_BASE",
                "DEEPSEEK": "DEEPSEEK_API_BASE",
            }

            env_key = base_url_map.get(provider_type, f"{provider_type}_API_BASE")
            os.environ[env_key] = provider_config.base_url
            env_vars[env_key] = provider_config.base_url

        # Set any extra parameters as environment variables
        for key, value in provider_config.extra_params.items():
            env_key = f"{provider_type}_{key.upper()}"
            os.environ[env_key] = str(value)
            env_vars[env_key] = str(value)

        return env_vars

    def create_model(
        self,
        model_spec: str,
        weak_model: Optional[str] = None,
        editor_model: Optional[str] = None,
        **kwargs
    ) -> "Model":
        """
        Create a Model instance with endpoint-aware configuration.

        This method resolves the model specification, sets up the appropriate
        environment, and returns a configured Model instance.
        """
        # Resolve the model specification
        resolved_model, provider_config = self.resolve_model_spec(model_spec)

        # Set up environment for this specific endpoint
        env_vars = self.setup_environment_for_model(model_spec)

        # Apply model-specific settings from config
        model_settings = {}
        if self.config:
            model_settings = self.config.get_model_settings(model_spec)

        try:
            # Create the model instance with the resolved name
            # The Model class will use the environment we just set up
            model = Model(
                model=resolved_model,
                weak_model=weak_model,
                editor_model=editor_model,
                **kwargs
            )

            # Apply any model-specific settings from the config
            if model_settings:
                for key, value in model_settings.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                    elif hasattr(model, 'extra_params'):
                        if not model.extra_params:
                            model.extra_params = {}
                        model.extra_params[key] = value

            # Store endpoint information on the model for reference
            model._endpoint_name = model_spec.split("/")[0] if "/" in model_spec else None
            model._provider_config = provider_config
            model._resolved_spec = model_spec

            return model

        except Exception as e:
            # Clean up environment variables if model creation fails
            for env_key in env_vars:
                if env_key in os.environ:
                    del os.environ[env_key]
            raise e

    def get_available_models(self) -> dict:
        """
        Get all available models organized by endpoint.

        Returns:
            Dictionary mapping endpoint names to lists of available models
        """
        if not self.config:
            return {}

        available = {}
        for endpoint_name, provider_config in self.config.providers.items():
            if provider_config.models:
                available[endpoint_name] = provider_config.models.copy()
            else:
                # If no models specified, return empty list (allowing any model)
                available[endpoint_name] = []

        return available

    def validate_model_spec(self, model_spec: str) -> tuple[bool, str]:
        """
        Validate a model specification.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.config:
            # Without config, accept any model that exists in legacy aliases
            if model_spec in MODEL_ALIASES or "/" in model_spec:
                return True, ""
            return False, f"Model '{model_spec}' not found in aliases"

        resolved = self.config.resolve_model_name(model_spec)
        if not resolved:
            return False, f"Model '{model_spec}' not found in configuration"

        # Check if the endpoint exists
        if "/" in resolved:
            endpoint_name = resolved.split("/", 1)[0]
            if endpoint_name not in self.config.providers:
                return False, f"Endpoint '{endpoint_name}' not configured"

        return True, ""


# Global endpoint-aware model manager
endpoint_model_manager = EndpointAwareModelManager()


def register_models(model_settings_fnames):
    files_loaded = []
    for model_settings_fname in model_settings_fnames:
        if not os.path.exists(model_settings_fname):
            continue

        if not Path(model_settings_fname).read_text().strip():
            continue

        try:
            with open(model_settings_fname, "r") as model_settings_file:
                model_settings_list = yaml.safe_load(model_settings_file)

            for model_settings_dict in model_settings_list:
                model_settings = ModelSettings(**model_settings_dict)
                existing_model_settings = next(
                    (ms for ms in MODEL_SETTINGS if ms.name == model_settings.name), None
                )

                if existing_model_settings:
                    MODEL_SETTINGS.remove(existing_model_settings)
                MODEL_SETTINGS.append(model_settings)
        except Exception as e:
            raise Exception(f"Error loading model settings from {model_settings_fname}: {e}")
        files_loaded.append(model_settings_fname)

    return files_loaded


def register_litellm_models(model_fnames):
    files_loaded = []
    for model_fname in model_fnames:
        if not os.path.exists(model_fname):
            continue

        try:
            data = Path(model_fname).read_text()
            if not data.strip():
                continue
            model_def = json5.loads(data)
            if not model_def:
                continue

            # Defer registration with litellm to faster path.
            model_info_manager.local_model_metadata.update(model_def)
        except Exception as e:
            raise Exception(f"Error loading model definition from {model_fname}: {e}")

        files_loaded.append(model_fname)

    return files_loaded


def validate_variables(vars):
    missing = []
    for var in vars:
        if var not in os.environ:
            missing.append(var)
    if missing:
        return dict(keys_in_environment=False, missing_keys=missing)
    return dict(keys_in_environment=True, missing_keys=missing)


def sanity_check_models(io, main_model):
    problem_main = sanity_check_model(io, main_model)

    problem_weak = None
    if main_model.weak_model and main_model.weak_model is not main_model:
        problem_weak = sanity_check_model(io, main_model.weak_model)

    problem_editor = None
    if (
        main_model.editor_model
        and main_model.editor_model is not main_model
        and main_model.editor_model is not main_model.weak_model
    ):
        problem_editor = sanity_check_model(io, main_model.editor_model)

    return problem_main or problem_weak or problem_editor


def sanity_check_model(io, model):
    show = False

    if model.missing_keys:
        show = True
        io.tool_warning(f"Warning: {model} expects these environment variables")
        for key in model.missing_keys:
            value = os.environ.get(key, "")
            status = "Set" if value else "Not set"
            io.tool_output(f"- {key}: {status}")

        if platform.system() == "Windows":
            io.tool_output(
                "Note: You may need to restart your terminal or command prompt for `setx` to take"
                " effect."
            )

    elif not model.keys_in_environment:
        show = True
        io.tool_warning(f"Warning for {model}: Unknown which environment variables are required.")

    # Check for model-specific dependencies
    check_for_dependencies(io, model.name)

    if not model.info:
        show = True
        io.tool_warning(
            f"Warning for {model}: Unknown context window size and costs, using sane defaults."
        )

        possible_matches = fuzzy_match_models(model.name)
        if possible_matches:
            io.tool_output("Did you mean one of these?")
            for match in possible_matches:
                io.tool_output(f"- {match}")

    return show


def check_for_dependencies(io, model_name):
    """
    Check for model-specific dependencies and install them if needed.

    Args:
        io: The IO object for user interaction
        model_name: The name of the model to check dependencies for
    """
    # Check if this is a Bedrock model and ensure boto3 is installed
    if model_name.startswith("bedrock/"):
        check_pip_install_extra(
            io, "boto3", "AWS Bedrock models require the boto3 package.", ["boto3"]
        )

    # Check if this is a Vertex AI model and ensure google-cloud-aiplatform is installed
    elif model_name.startswith("vertex_ai/"):
        check_pip_install_extra(
            io,
            "google.cloud.aiplatform",
            "Google Vertex AI models require the google-cloud-aiplatform package.",
            ["google-cloud-aiplatform"],
        )


def fuzzy_match_models(name):
    """
    Find models that match the search term, using only configured and actually available models.

    Search order:
    1. Models from .laied.conf.yml configuration
    2. Actually available models (by querying APIs)

    No fallback to LiteLLM database to avoid unusable models.
    """
    name = name.lower()
    chat_models = set()

    # First priority: Get models from configuration
    try:
        from aider.config import ConfigManager
        config_manager = ConfigManager()
        config_path = config_manager.find_config_file()

        if config_path:
            config = config_manager.load_config(config_path)

            # Add configured models from each provider
            for endpoint_name, provider in config.providers.items():
                if provider.models:
                    for model in provider.models:
                        # Add fully qualified model name using endpoint name
                        fq_model = f"{endpoint_name}/{model}"
                        chat_models.add(fq_model)
                        # Add short model name
                        chat_models.add(model)

            # Add model aliases
            for alias, target in config.model_aliases.items():
                chat_models.add(alias)
                chat_models.add(target)

    except Exception:
        # If config system fails, we'll fall back to discovery
        pass

    # Second priority: Query actual endpoints for available models
    try:
        discovered_models = _discover_available_models()
        chat_models.update(discovered_models)
    except Exception:
        # Discovery failed, continue with configured models only
        pass

    # No LiteLLM fallback - only use configured and discovered models

    chat_models = sorted(chat_models)

    # Check for model names containing the search term
    matching_models = [m for m in chat_models if name in m.lower()]
    if matching_models:
        return sorted(set(matching_models))

    # Check for slight misspellings
    models = set(chat_models)
    matching_models = difflib.get_close_matches(name, models, n=3, cutoff=0.8)

    return sorted(set(matching_models))


def _discover_available_models():
    """Discover actually available models by querying provider APIs."""
    available_models = set()

    try:
        from aider.config import ConfigManager
        config_manager = ConfigManager()
        config_path = config_manager.find_config_file()

        if config_path:
            config = config_manager.load_config(config_path)

            for endpoint_name, provider in config.providers.items():
                try:
                    if provider.type == 'ollama':
                        models = _query_ollama_available_models(provider.base_url or 'http://localhost:11434')
                        for model in models:
                            available_models.add(f"{endpoint_name}/{model}")
                            available_models.add(model)

                    # Could add other providers here (OpenAI, Anthropic, etc.)
                    # but they don't typically have "installed" vs "available" distinction

                except Exception:
                    # If individual provider query fails, continue with others
                    continue

    except Exception:
        # If config system fails entirely, return empty set
        pass

    return available_models


def _query_ollama_available_models(base_url):
    """Query Ollama API for actually installed models."""
    import requests

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
    except Exception:
        pass

    return []


def print_matching_models(io, search):
    """Print a nicely formatted table of models with metadata and deduplication."""
    matches = fuzzy_match_models(search)
    if not matches:
        io.tool_output(f'No models match "{search}".')
        return

    # Load config to get aliases and provider info
    try:
        from aider.config import ConfigManager
        config_manager = ConfigManager()
        config_path = config_manager.find_config_file()
        config = None
        if config_path:
            config = config_manager.load_config(config_path)
    except Exception:
        config = None

    # Organize models and aliases
    aliases = {}
    provider_models = {}
    reverse_aliases = {}  # model -> [aliases]

    if config:
        # Get aliases
        aliases = config.model_aliases or {}
        # Create reverse mapping for alias lookup
        for alias, target in aliases.items():
            if target not in reverse_aliases:
                reverse_aliases[target] = []
            reverse_aliases[target].append(alias)

        # Organize by provider
        for endpoint_name, provider in config.providers.items():
            if provider.models:
                provider_models[endpoint_name] = {
                    'type': provider.type,
                    'models': set(provider.models),
                    'endpoint': endpoint_name
                }

    if search:
        io.tool_output(f'Models which match "{search}":')
    io.tool_output()

    # Show models grouped by provider in table format
    if provider_models:
        for endpoint_name, provider_info in sorted(provider_models.items()):
            provider_matches = []
            for model in provider_info['models']:
                fq_model = f"{endpoint_name}/{model}"
                if (fq_model in matches or model in matches):
                    if search.lower() in model.lower() or search.lower() in fq_model.lower():
                        provider_matches.append(model)

            if provider_matches:
                provider_type = provider_info['type'].title()
                endpoint_desc = f" ({endpoint_name})" if endpoint_name != provider_info['type'] else ""
                io.tool_output(f"**{provider_type}{endpoint_desc}:**")

                # Create table
                _print_models_table(io, provider_matches, provider_info['type'], reverse_aliases, endpoint_name, config_manager)
                io.tool_output()

def _print_models_table(io, models, provider_type, reverse_aliases, endpoint_name, config_manager=None):
    """Print a formatted table of models."""
    # Prepare table data
    table_data = []

    for model in sorted(models):
        fq_model = f"{endpoint_name}/{model}"

        # Get description (try to get from API metadata first)
        description = _get_model_description(model, provider_type, config_manager)

        # Get aliases for this model
        model_aliases = []
        if fq_model in reverse_aliases:
            model_aliases.extend(reverse_aliases[fq_model])
        if model in reverse_aliases:
            model_aliases.extend(reverse_aliases[model])

        alias_str = ", ".join(sorted(set(model_aliases))) if model_aliases else ""

        table_data.append({
            'name': model,
            'description': description,
            'aliases': alias_str
        })

    if not table_data:
        return

    # Calculate column widths
    max_name = max(len(row['name']) for row in table_data)
    max_desc = max(len(row['description']) for row in table_data)
    max_aliases = max(len(row['aliases']) for row in table_data)

    # Ensure minimum widths and reasonable maximums
    name_width = max(12, min(max_name + 2, 35))
    desc_width = max(15, min(max_desc + 2, 50))
    alias_width = max(8, min(max_aliases + 2, 25))

    # Print table header
    header = f"│ {'Name':<{name_width}} │ {'Description':<{desc_width}} │ {'Aliases':<{alias_width}} │"
    separator = f"├─{'─' * name_width}─┼─{'─' * desc_width}─┼─{'─' * alias_width}─┤"
    top_border = f"┌─{'─' * name_width}─┬─{'─' * desc_width}─┬─{'─' * alias_width}─┐"
    bottom_border = f"└─{'─' * name_width}─┴─{'─' * desc_width}─┴─{'─' * alias_width}─┘"

    io.tool_output(top_border)
    io.tool_output(header)
    io.tool_output(separator)

    # Print table rows
    for i, row in enumerate(table_data):
        # Truncate if too long
        name = row['name'][:name_width].ljust(name_width)
        desc = row['description'][:desc_width].ljust(desc_width)
        aliases = row['aliases'][:alias_width].ljust(alias_width)

        io.tool_output(f"│ {name} │ {desc} │ {aliases} │")

    io.tool_output(bottom_border)

def _get_model_description(model_name: str, provider_type: str, config_manager=None) -> str:
    """Get a human-readable description for a model."""
    # Try to get live Anthropic display name
    if provider_type == 'anthropic':
        anthropic_display_name = _get_anthropic_display_name(model_name, config_manager)
        if anthropic_display_name:
            return anthropic_display_name

    descriptions = {
        # OpenAI models
        'gpt-4o': 'GPT-4 Omni - latest multimodal model',
        'gpt-4': 'GPT-4 - advanced reasoning',
        'gpt-3.5-turbo': 'GPT-3.5 Turbo - fast and efficient',
        'o1-preview': 'O1 Preview - advanced reasoning',
        'o1-mini': 'O1 Mini - lightweight reasoning',

        # Anthropic models
        'claude-sonnet-4-20250514': 'Claude 4 Sonnet - latest and most capable',
        'claude-opus-4-20250514': 'Claude 4 Opus - maximum intelligence',
        'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet - balanced performance',
        'claude-3-5-haiku-20241022': 'Claude 3.5 Haiku - fast and efficient',

        # Groq models
        'llama-3.1-8b-instant': 'Llama 3.1 8B - fast inference',
        'llama-3.3-70b-versatile': 'Llama 3.3 70B - versatile large model',

        # Deepseek models
        'deepseek-chat': 'Deepseek Chat - general conversation',
        'deepseek-reasoner': 'Deepseek Reasoner - advanced reasoning',

        # Gemini models
        'gemini-2.5-pro': 'Gemini 2.5 Pro - most advanced',
        'gemini-1.5-pro': 'Gemini 1.5 Pro - multimodal',
        'gemini-1.5-flash': 'Gemini 1.5 Flash - fast responses',
    }

    # Try exact match first
    if model_name in descriptions:
        return descriptions[model_name]

    # Try pattern matching for versioned models
    base_name = model_name.split('-')[0] + '-' + model_name.split('-')[1] if '-' in model_name else model_name
    if base_name in descriptions:
        return descriptions[base_name]

    # Provider-specific patterns
    if provider_type == 'ollama':
        if 'qwen' in model_name.lower():
            return 'Qwen model for local inference'
        elif 'gemma' in model_name.lower():
            return 'Google Gemma model'
        elif 'phi' in model_name.lower():
            return 'Microsoft Phi model'
        elif 'llama' in model_name.lower():
            return 'Meta Llama model'

    return ""

def _get_anthropic_display_name(model_name: str, config_manager=None) -> str:
    """Get live display name from Anthropic API."""
    if not config_manager:
        return ""

    try:
        # Get Anthropic API key from config
        config = config_manager.load_config()
        if not config or 'anthropic' not in config.providers:
            return ""

        anthropic_provider = config.providers['anthropic']
        api_key = anthropic_provider.api_key
        if not api_key:
            return ""

        import requests
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }

        # Query Anthropic API (with caching to avoid repeated calls)
        if not hasattr(_get_anthropic_display_name, '_cache'):
            response = requests.get('https://api.anthropic.com/v1/models', headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                _get_anthropic_display_name._cache = {
                    model['id']: model.get('display_name', '')
                    for model in data.get('data', [])
                }
            else:
                _get_anthropic_display_name._cache = {}

        return _get_anthropic_display_name._cache.get(model_name, "")

    except Exception:
        return ""


def get_model_settings_as_yaml():
    from dataclasses import fields

    import yaml

    model_settings_list = []
    # Add default settings first with all field values
    defaults = {}
    for field in fields(ModelSettings):
        defaults[field.name] = field.default
    defaults["name"] = "(default values)"
    model_settings_list.append(defaults)

    # Sort model settings by name
    for ms in sorted(MODEL_SETTINGS, key=lambda x: x.name):
        # Create dict with explicit field order
        model_settings_dict = {}
        for field in fields(ModelSettings):
            value = getattr(ms, field.name)
            if value != field.default:
                model_settings_dict[field.name] = value
        model_settings_list.append(model_settings_dict)
        # Add blank line between entries
        model_settings_list.append(None)

    # Filter out None values before dumping
    yaml_str = yaml.dump(
        [ms for ms in model_settings_list if ms is not None],
        default_flow_style=False,
        sort_keys=False,  # Preserve field order from dataclass
    )
    # Add actual blank lines between entries
    return yaml_str.replace("\n- ", "\n\n- ")


def main():
    if len(sys.argv) < 2:
        print("Usage: python models.py <model_name> or python models.py --yaml")
        sys.exit(1)

    if sys.argv[1] == "--yaml":
        yaml_string = get_model_settings_as_yaml()
        print(yaml_string)
    else:
        model_name = sys.argv[1]
        matching_models = fuzzy_match_models(model_name)

        if matching_models:
            print(f"Matching models for '{model_name}':")
            for model in matching_models:
                print(model)
        else:
            print(f"No matching models found for '{model_name}'.")


if __name__ == "__main__":
    main()
