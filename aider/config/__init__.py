import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import yaml

from .dataclasses import AiderConfig  # noqa: F401
from .dataclasses import AnalyticsConfig  # noqa: F401
from .dataclasses import CacheConfig  # noqa: F401
from .dataclasses import GitConfig  # noqa: F401
from .dataclasses import ModelParameterOverrides  # noqa: F401
from .dataclasses import OutputConfig  # noqa: F401
from .dataclasses import ProviderConfig  # noqa: F401
from .dataclasses import RepomapConfig  # noqa: F401
from .dataclasses import VoiceConfig  # noqa: F401; noqa: F401
from .discovery import check_openai_compatible_api  # noqa: F401
from .discovery import check_port_open  # noqa: F401
from .discovery import discover_local_llm_servers  # noqa: F401
from .discovery import scan_running_llm_processes  # noqa: F401
from .discovery import update_config_with_discoveries  # noqa: F401; noqa: F401


class ConfigManager:
    """Manages loading and saving of aider configuration."""

    DEFAULT_CONFIG_NAMES = [".laied.conf.yml", ".laied.conf.yaml"]

    def __init__(self):
        self.config_path = None
        self.config = None

    def find_config_file(self, start_path: Optional[Path] = None) -> Optional[Path]:
        """
        Find configuration file by searching up the directory tree.

        Search order:
        1. Current working directory
        2. Git repository root (if in a git repo)
        3. Home directory
        """
        if start_path is None:
            start_path = Path.cwd()

        search_paths = [start_path]

        # Add git repository root if we're in one
        try:
            import git

            repo = git.Repo(start_path, search_parent_directories=True)
            repo_root = Path(repo.working_tree_dir)
            if repo_root != start_path:
                search_paths.append(repo_root)
        except (ImportError, git.exc.InvalidGitRepositoryError):
            pass

        # Add home directory
        search_paths.append(Path.home())

        # Search for config files
        for path in search_paths:
            for config_name in self.DEFAULT_CONFIG_NAMES:
                config_file = path / config_name
                if config_file.exists():
                    return config_file

        return None

    def check_and_migrate_old_config(self) -> Optional[Path]:
        """
        Check for old .aider.yml files and offer to migrate them.

        Returns:
            Path to migrated config file if migration occurred, None otherwise
        """
        old_config_names = [".aider.yml", ".aider.yaml"]

        # Check same locations as find_config_file
        search_paths = [Path.cwd()]

        # Add git repository root if we're in one
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                git_root = Path(result.stdout.strip())
                if git_root != Path.cwd():
                    search_paths.append(git_root)
        except Exception:
            pass

        search_paths.append(Path.home())

        # Look for old config files
        for path in search_paths:
            for old_name in old_config_names:
                old_file = path / old_name
                if old_file.exists():
                    new_file = path / ".laied.conf.yml"
                    if not new_file.exists():
                        print(f"\n⚠️  Found old configuration file: {old_file}")
                        print(
                            f"   Would you like to migrate it to {new_file}? (y/n): ",
                            end="",
                            flush=True,
                        )
                        try:
                            response = input().strip().lower()
                            if response == "y":
                                import shutil

                                shutil.copy2(old_file, new_file)
                                print(f"✅ Migrated configuration to {new_file}")
                                print(f"   You can now delete the old {old_file} file.\n")
                                return new_file
                        except (KeyboardInterrupt, EOFError):
                            print("\n   Migration cancelled.")
                    else:
                        print(
                            f"\n⚠️  Found old config {old_file} but new config already exists at"
                            f" {new_file}"
                        )
                        print("   Consider removing the old file to avoid confusion.\n")

        return None

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> AiderConfig:
        """
        Load configuration from file or create default configuration.

        Args:
            config_path: Explicit path to config file, or None to auto-discover

        Returns:
            AiderConfig instance
        """
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        else:
            config_path = self.find_config_file()

            # If no new config found, check for old config files to migrate
            if not config_path:
                migrated_path = self.check_and_migrate_old_config()
                if migrated_path:
                    config_path = migrated_path

        if config_path:
            self.config_path = config_path
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

            # Convert provider configurations
            providers = {}
            if "providers" in config_data:
                for name, provider_data in config_data["providers"].items():
                    providers[name] = ProviderConfig(**provider_data)
            config_data["providers"] = providers

            # Convert sub-configurations
            if "git" in config_data:
                config_data["git"] = GitConfig(**config_data["git"])
            if "output" in config_data:
                config_data["output"] = OutputConfig(**config_data["output"])
            if "cache" in config_data:
                config_data["cache"] = CacheConfig(**config_data["cache"])
            if "repomap" in config_data:
                config_data["repomap"] = RepomapConfig(**config_data["repomap"])
            if "voice" in config_data:
                config_data["voice"] = VoiceConfig(**config_data["voice"])
            if "analytics" in config_data:
                config_data["analytics"] = AnalyticsConfig(**config_data["analytics"])

            self.config = AiderConfig(**config_data)
        else:
            # Create default configuration
            self.config = AiderConfig()

        return self.config

    def save_config(self, config: AiderConfig, config_path: Optional[Union[str, Path]] = None):
        """Save configuration to file."""
        if config_path:
            config_path = Path(config_path)
        elif self.config_path:
            config_path = self.config_path
        else:
            config_path = Path.cwd() / ".laied.conf.yml"

        # Convert to dictionary
        config_dict = self._config_to_dict(config)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self.config_path = config_path

    def _config_to_dict(self, config: AiderConfig) -> Dict[str, Any]:
        """Convert AiderConfig to dictionary for YAML serialization."""
        result = {}

        # Providers
        if config.providers:
            result["providers"] = {}
            for name, provider in config.providers.items():
                provider_dict = {
                    "type": provider.type,
                }
                if provider.api_key:
                    provider_dict["api_key"] = provider.api_key
                if provider.base_url:
                    provider_dict["base_url"] = provider.base_url
                if provider.models:
                    provider_dict["models"] = provider.models
                if provider.extra_params:
                    provider_dict.update(provider.extra_params)
                result["providers"][name] = provider_dict

        # Model configuration
        if config.model_aliases:
            result["model_aliases"] = config.model_aliases

        result["model"] = config.model
        if config.weak_model:
            result["weak_model"] = config.weak_model
        if config.editor_model:
            result["editor_model"] = config.editor_model

        if config.model_settings:
            result["model_settings"] = config.model_settings

        # Sub-configurations
        result["git"] = self._dataclass_to_dict(config.git)
        result["output"] = self._dataclass_to_dict(config.output)
        result["cache"] = self._dataclass_to_dict(config.cache)
        result["repomap"] = self._dataclass_to_dict(config.repomap)
        result["voice"] = self._dataclass_to_dict(config.voice)
        result["analytics"] = self._dataclass_to_dict(config.analytics)

        # Other settings
        result["edit_format"] = config.edit_format
        result["encoding"] = config.encoding
        result["vim"] = config.vim
        result["check_update"] = config.check_update

        return result

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary, excluding None values."""
        result = {}
        for key, value in obj.__dict__.items():
            if value is not None:
                result[key] = value
        return result

    def create_sample_config(
        self,
        path: Optional[Union[str, Path]] = None,
        include_env_vars: bool = True,
    ) -> Path:
        """Create a sample configuration file.

        Optionally include detected environment variables.
        """
        if path:
            config_path = Path(path)
        else:
            config_path = Path.cwd() / ".laied.conf.yml"

        if include_env_vars:
            # Create config with detected environment variables
            self._create_config_with_env_vars(config_path)
        else:
            # Find the sample file in the aider package
            import aider

            aider_dir = Path(aider.__file__).parent.parent
            sample_file = aider_dir / "laied.conf.yml.sample"

            if sample_file.exists():
                # Copy the sample file
                import shutil

                shutil.copy2(sample_file, config_path)
            else:
                # Fallback to creating a basic config
                self._create_basic_config_file(config_path)

        return config_path

    def update_existing_config_with_models(
        self, config_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Update an existing configuration file with newly discovered models."""
        # Find existing config file
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        else:
            config_path = self.find_config_file()
            if not config_path:
                raise FileNotFoundError(
                    "No configuration file found. Use --init-config to create one."
                )

        print(f"Updating configuration file: {config_path}")

        # Load existing configuration
        existing_config = self.load_config(config_path)

        # Convert providers to dict format for interrogation (preserve endpoint names)
        provider_configs = {}
        for name, provider in existing_config.providers.items():
            provider_configs[name] = {
                "type": provider.type,
                "api_key": provider.api_key,
                "base_url": provider.base_url or (
                    "https://api.openai.com/v1" if provider.type == "openai" else None
                ),
            }

        # Interrogate endpoints for new models
        if provider_configs:
            print("Discovering available models from API endpoints...")
            updated_providers = self.interrogate_endpoints_for_models(provider_configs)

            # Update the existing config with new models
            for name, provider in existing_config.providers.items():
                if name in updated_providers and "models" in updated_providers[name]:
                    new_models = updated_providers[name]["models"]
                    old_models = provider.models or []
                    provider.models = new_models
                    print(f"  Updated {name}: {len(old_models)} -> {len(new_models)} models")

        # Save the updated configuration
        self.save_config(existing_config, config_path)
        print(f"Configuration file updated: {config_path}")

        return config_path

    def interrogate_endpoints_for_models(
        self, detected_providers: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Query API endpoints to discover available models."""
        import time

        # Initialize metadata storage
        self._model_metadata = {}

        for endpoint_name, config in detected_providers.items():
            provider_type = config.get("type", endpoint_name)

            # Skip providers that require API keys but don't have them
            if provider_type not in ["ollama"] and "api_key" not in config:
                continue

            print(f"Discovering models for {endpoint_name}...")

            try:
                models = self._query_provider_models(provider_type, config)
                if models:
                    config["models"] = models
                    print(f"  Found {len(models)} models")
                else:
                    print(f"  No models discovered for {endpoint_name} (API may be unavailable)")
                    config["models"] = []
            except Exception as e:
                print(f"  Model discovery failed for {endpoint_name}: {e}")
                config["models"] = []

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        return detected_providers

    def _filter_chat_models(self, models: List[str]) -> List[str]:
        """Filter models to only include those suitable for chat completion."""
        # Patterns for non-chat models that should be excluded
        exclude_patterns = [
            # Audio/TTS models
            "tts",
            "whisper",
            "speech",
            "audio",
            # Image generation models
            "dall-e",
            "image",
            "vision-only",
            "-image-",
            # Embeddings models
            "embedding",
            "embed",
            # Safety/guard models
            "guard",
            "safety",
            "moderation",
            "prompt-guard",
            # Translation models
            "translate",
            # Fine-tuning models
            "tuning",
            # Specific unusable/experimental models
            "playai-tts",
            "distil-whisper",
            "allam-2-7b",
            # Preview/experimental image generation
            "preview-image-generation",
            "exp-image-generation",
        ]

        filtered_models = []
        for model in models:
            model_lower = model.lower()
            # Skip if model name contains any exclude pattern
            if any(pattern in model_lower for pattern in exclude_patterns):
                continue
            # Skip if model name ends with non-chat suffixes
            if any(
                model_lower.endswith(suffix) for suffix in ["-tts", "-stt", "-guard", "-embedding"]
            ):
                continue
            filtered_models.append(model)

        return filtered_models

    def _query_provider_models(self, provider_type: str, config: Dict[str, str]) -> List[str]:
        """Query a specific provider for available models."""

        if provider_type == "openai":
            models = self._query_openai_models(config)
        elif provider_type == "anthropic":
            models = self._query_anthropic_models(config)
        elif provider_type == "groq":
            models = self._query_groq_models(config)
        elif provider_type == "ollama":
            models = self._query_ollama_models(config)
        elif provider_type == "deepseek":
            models = self._query_deepseek_models(config)
        elif provider_type == "gemini":
            models = self._query_gemini_models(config)
        else:
            models = []

        # Filter out non-chat models
        return self._filter_chat_models(models)

    def _query_openai_models(self, config: Dict[str, str]) -> List[str]:
        """Query OpenAI API for available models."""

        base_url = config.get("base_url", "https://api.openai.com/v1")
        api_key = config.get("api_key")

        if not api_key:
            return []

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [
                    model["id"]
                    for model in data.get("data", [])
                    if model["id"].startswith(("gpt-", "o1-", "chatgpt-"))
                ]
                return sorted(models)
        except Exception:
            pass

        return []

    def _query_anthropic_models(self, config: Dict[str, str]) -> List[str]:
        """Query Anthropic API for available models with display names."""

        api_key = config.get("api_key")
        if not api_key:
            return []

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        try:
            response = requests.get(
                "https://api.anthropic.com/v1/models", headers=headers, timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("data", []):
                    model_id = model["id"]
                    models.append(model_id)
                    # Store display name for later use
                    if hasattr(self, "_model_metadata"):
                        self._model_metadata[model_id] = {
                            "display_name": model.get("display_name"),
                            "created_at": model.get("created_at"),
                        }
                return models
        except Exception:
            pass

        # Fallback to known models if API fails
        return [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    def _query_groq_models(self, config: Dict[str, str]) -> List[str]:
        """Query Groq API for available models."""

        api_key = config.get("api_key")
        if not api_key:
            return []

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models", headers=headers, timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data.get("data", [])]
                return sorted(models)
        except Exception:
            pass

        return []

    def _query_ollama_models(self, config: Dict[str, str]) -> List[str]:
        """Query Ollama API for available models."""

        base_url = config.get("base_url", "http://localhost:11434")

        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return sorted(models)
        except Exception:
            pass

        return []

    def _query_deepseek_models(self, config: Dict[str, str]) -> List[str]:
        """Query Deepseek API for available models."""

        api_key = config.get("api_key")
        if not api_key:
            return []

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            # Try Deepseek's OpenAI-compatible models endpoint
            response = requests.get(
                "https://api.deepseek.com/v1/models", headers=headers, timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data.get("data", [])]
                return sorted(models)
        except Exception:
            pass

        return []

    def _query_gemini_models(self, config: Dict[str, str]) -> List[str]:
        """Query Gemini API for available models."""

        api_key = config.get("api_key")
        if not api_key:
            return []

        try:
            # Try Google AI Studio models endpoint
            response = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                # Extract model names that support generateContent
                models = []
                for model in data.get("models", []):
                    model_name = model.get("name", "").replace("models/", "")
                    supported_methods = model.get("supportedGenerationMethods", [])
                    if "generateContent" in supported_methods and model_name.startswith("gemini"):
                        models.append(model_name)
                return sorted(models)
        except Exception:
            pass

        return []

    def _get_default_models(self, provider_type: str) -> List[str]:
        """Get default models for a provider when discovery fails."""
        defaults = {
            "openai": ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": [
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
            ],
            "groq": ["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            "ollama": ["llama3:8b", "codellama:7b", "mistral:7b"],
            "deepseek": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
            "gemini": ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
            "xai": ["grok-1"],
            "cohere": ["command-r", "command-r-plus"],
        }
        return defaults.get(provider_type, [])

    def _create_config_with_env_vars(self, config_path: Path):
        """Create a configuration file with detected environment variables."""
        detected_providers = {}

        # Detect provider configurations from environment
        env_vars_to_check = {
            "OPENAI_API_KEY": ("openai", "api_key"),
            "OPENAI_API_BASE": ("openai", "base_url"),
            "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
            "ANTHROPIC_API_BASE": ("anthropic", "base_url"),
            "GROQ_API_KEY": ("groq", "api_key"),
            "GROQ_API_BASE": ("groq", "base_url"),
            "DEEPSEEK_API_KEY": ("deepseek", "api_key"),
            "DEEPSEEK_API_BASE": ("deepseek", "base_url"),
            "GEMINI_API_KEY": ("gemini", "api_key"),
            "XAI_API_KEY": ("xai", "api_key"),
            "COHERE_API_KEY": ("cohere", "api_key"),
            "OLLAMA_HOST": ("ollama", "base_url"),
        }

        for env_var, (provider_type, setting) in env_vars_to_check.items():
            value = os.getenv(env_var)
            if value:
                if provider_type not in detected_providers:
                    detected_providers[provider_type] = {"type": provider_type}
                detected_providers[provider_type][setting] = value

        # Always include ollama endpoints
        # First try OLLAMA_HOST if set
        ollama_host = os.getenv("OLLAMA_HOST")
        if ollama_host and "ollama" not in detected_providers:
            detected_providers["ollama"] = {"type": "ollama", "base_url": ollama_host}

        # Always try localhost:11434 as well (as separate endpoint if different)
        localhost_url = "http://localhost:11434"
        if (
            "ollama" not in detected_providers
            or detected_providers.get("ollama", {}).get("base_url") != localhost_url
        ):
            # Add localhost as primary or secondary ollama endpoint
            endpoint_name = "ollama" if "ollama" not in detected_providers else "ollama_local"
            detected_providers[endpoint_name] = {
                "type": "ollama",
                "base_url": localhost_url,
            }

        # Discover additional local LLM servers (non-interactive for automatic config creation)
        print("Scanning for local LLM servers...")
        discovered_local = discover_local_llm_servers(interactive=False)
        if discovered_local:
            print(f"Found {len(discovered_local)} local LLM server(s):")
            for name, info in discovered_local.items():
                models_count = len(info.get("models", []))
                process_note = " (from running process)" if info.get("process_discovered") else ""
                print(
                    f"  - {info['description']} at"
                    f" {info['base_url']} ({models_count} models){process_note}"
                )
                # Add to detected providers if not already present
                if name not in detected_providers:
                    detected_providers[name] = {
                        "type": info["type"],
                        "base_url": info["base_url"],
                        "models": info["models"],
                    }
            print()
        else:
            print("No additional local LLM servers found.")
            print()

        # Interrogate endpoints for models
        if detected_providers:
            print("Discovering available models from API endpoints...")
            detected_providers = self.interrogate_endpoints_for_models(detected_providers)

        # Create config content with detected values
        config_content = self._generate_config_content_with_env_vars(detected_providers)

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

    def _generate_config_content_with_env_vars(
        self, detected_providers: Dict[str, Dict[str, str]]
    ) -> str:
        """Generate configuration file content with detected environment variables."""
        content = """# Aider Configuration File
# Generated with detected environment variables
# For full documentation: https://aider.chat/docs/config/

"""

        if detected_providers:
            content += "providers:\n"

            for endpoint_name, config in detected_providers.items():
                provider_type = config.get("type", endpoint_name)
                content += f"  {endpoint_name}:\n"
                content += f"    type: {provider_type}\n"

                if "api_key" in config:
                    content += f"    api_key: \"{config['api_key']}\"\n"

                if "base_url" in config:
                    content += f"    base_url: \"{config['base_url']}\"\n"

                # Add discovered or default models
                if "models" in config and config["models"]:
                    models_str = ", ".join([f'"{model}"' for model in config["models"]])
                    content += f"    models: [{models_str}]\n"

                content += "\n"
        else:
            # No environment variables detected, create example config
            content += """providers:
  openai:
    type: openai
    api_key: "sk-your-openai-key-here"
    base_url: "https://api.openai.com/v1"
    models: ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]

  anthropic:
    type: anthropic
    api_key: "sk-ant-your-key-here"
    models: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]

  ollama:
    type: ollama
    base_url: "http://localhost:11434"
    models: ["llama3:8b", "codellama:7b", "mistral:7b"]

"""

        # Generate model aliases and default model based on discovered providers
        aliases_content = self._generate_model_aliases(detected_providers)
        default_model = self._select_default_model(detected_providers)

        if aliases_content:
            content += f"\n# Model aliases\nmodel_aliases:\n{aliases_content}\n"

        content += f"""

# Default model to use
model: "{default_model}"

# Git settings
git:
  auto_commits: true
  commit_prefix: "aider: "
  attribute_author: true

# Output settings
output:
  user_input_color: "#00cc00"
  pretty: true
  stream: true
  show_diffs: false

# Other settings
edit_format: "diff"
vim: false
encoding: "utf-8"
"""

        return content

    def _create_basic_config_file(self, config_path: Path):
        """Create a basic configuration file if sample is not available."""
        basic_config = """# Aider Configuration File
# Copy this to .laied.conf.yml and customize as needed

providers:
  openai:
    type: openai
    api_key: "sk-your-openai-key-here"
    base_url: "https://api.openai.com/v1"
    models: ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]

  anthropic:
    type: anthropic
    api_key: "sk-ant-your-key-here"
    models: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]

  ollama:
    type: ollama
    base_url: "http://localhost:11434"
    models: ["llama3:8b", "codellama:7b"]

model_aliases:
  sonnet: "anthropic/claude-3-5-sonnet-20241022"
  gpt4: "openai/gpt-4"
  gpt4o: "openai/gpt-4o"

model: "sonnet"
weak_model: "openai/gpt-3.5-turbo"

git:
  auto_commits: true
  commit_prefix: "aider: "

output:
  user_input_color: "#00cc00"
  pretty: true
  stream: true

edit_format: "diff"
vim: false
"""
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(basic_config)

    def _generate_model_aliases(self, detected_providers: Dict[str, Dict[str, str]]) -> str:
        """Generate model aliases from discovered providers.

        Only create aliases for common or useful models.
        """
        aliases = []

        for endpoint_name, config in detected_providers.items():
            provider_type = config.get("type", endpoint_name)
            models = config.get("models", [])

            # Only create aliases for very common, well-known models
            if provider_type == "openai" and models:
                if "gpt-4o" in models:
                    aliases.append(f'  gpt4o: "{endpoint_name}/gpt-4o"')
                if "gpt-4" in models:
                    aliases.append(f'  gpt4: "{endpoint_name}/gpt-4"')

            elif provider_type == "anthropic" and models:
                # Find the latest sonnet model
                sonnet_models = [
                    m for m in models if "sonnet" in m.lower() and "claude-3-5" in m.lower()
                ]
                if sonnet_models:
                    # Use the latest one (they're typically sorted)
                    latest_sonnet = sonnet_models[-1]
                    aliases.append(f'  sonnet: "{endpoint_name}/{latest_sonnet}"')

            elif provider_type == "ollama" and models:
                # For ollama, create aliases for very common local models
                if any("llama" in m.lower() for m in models):
                    llama_models = [m for m in models if "llama" in m.lower()]
                    if llama_models:
                        aliases.append(f'  llama: "{endpoint_name}/{llama_models[0]}"')

        return "\n".join(aliases) if aliases else ""

    def _select_default_model(self, detected_providers: Dict[str, Dict[str, str]]) -> str:
        """Select a sensible default model from discovered providers."""

        # Priority order for selecting default model
        # 1. Anthropic Claude (if available)
        # 2. OpenAI GPT-4o or GPT-4 (if available)
        # 3. Local Ollama models (if available)
        # 4. Any other available model
        # 5. Fallback to a reasonable default

        for endpoint_name, config in detected_providers.items():
            provider_type = config.get("type", endpoint_name)
            models = config.get("models", [])

            if not models:
                continue

            # Prefer Anthropic Claude models
            if provider_type == "anthropic":
                sonnet_models = [
                    m for m in models if "sonnet" in m.lower() and "claude-3-5" in m.lower()
                ]
                if sonnet_models:
                    return f"{endpoint_name}/{sonnet_models[-1]}"  # Latest sonnet
                # Fallback to any claude model
                claude_models = [m for m in models if m.startswith("claude-")]
                if claude_models:
                    return f"{endpoint_name}/{claude_models[0]}"

        # Second choice: OpenAI models
        for endpoint_name, config in detected_providers.items():
            provider_type = config.get("type", endpoint_name)
            models = config.get("models", [])

            if provider_type == "openai" and models:
                if "gpt-4o" in models:
                    return f"{endpoint_name}/gpt-4o"
                elif "gpt-4" in models:
                    return f"{endpoint_name}/gpt-4"
                elif any("gpt-4" in m for m in models):
                    gpt4_models = [m for m in models if "gpt-4" in m]
                    return f"{endpoint_name}/{gpt4_models[0]}"

        # Third choice: Local Ollama models (prefer smaller, efficient ones)
        for endpoint_name, config in detected_providers.items():
            provider_type = config.get("type", endpoint_name)
            models = config.get("models", [])

            if provider_type == "ollama" and models:
                # Prefer lightweight models for local usage
                preferred_local = ["qwen3:0.6b", "qwen3:1.7b", "gemma3:1b", "phi4-mini"]
                for preferred in preferred_local:
                    if any(preferred in m for m in models):
                        matching = [m for m in models if preferred in m][0]
                        return f"{endpoint_name}/{matching}"

                # Fallback to first available local model
                return f"{endpoint_name}/{models[0]}"

        # Fourth choice: Any available model from any provider
        for endpoint_name, config in detected_providers.items():
            models = config.get("models", [])
            if models:
                return f"{endpoint_name}/{models[0]}"

        # Final fallback if no models found
        return "claude-3-5-sonnet-20241022"


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_path: Optional[Union[str, Path]] = None) -> AiderConfig:
    """Load aider configuration."""
    return config_manager.load_config(config_path)


def save_config(config: AiderConfig, config_path: Optional[Union[str, Path]] = None):
    """Save aider configuration."""
    config_manager.save_config(config, config_path)
