"""
Unified configuration system for aider.

This module provides a single, consistent way to configure aider through
configuration files, replacing the complex multi-source system of CLI args,
environment variables, and multiple config file formats.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ProviderConfig:
    """Configuration for a single provider endpoint."""
    type: str  # openai, ollama, anthropic, etc.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate provider configuration."""
        if not self.type:
            raise ValueError("Provider type is required")
        
        # Set default base URLs for known providers
        if self.type == "ollama" and not self.base_url:
            self.base_url = "http://localhost:11434"


@dataclass
class GitConfig:
    """Git operation settings."""
    auto_commits: bool = True
    commit_prefix: str = "aider: "
    auto_add: bool = True
    dirty_commits: bool = True
    attribute_author: bool = True
    attribute_committer: bool = True
    attribute_commit_message_author: bool = False
    attribute_commit_message_committer: bool = False
    dry_run: bool = False


@dataclass
class OutputConfig:
    """UI and display settings."""
    user_input_color: str = "#00cc00"
    tool_output_color: str = None
    tool_error_color: str = "#FF2222"
    tool_warning_color: str = "#FFA500"
    assistant_output_color: str = "#0088ff"
    completion_menu_color: str = "#ffffff"
    completion_menu_bg_color: str = "#444444"
    completion_menu_current_color: str = "#ffffff"
    completion_menu_current_bg_color: str = "#666666"
    code_theme: str = "default"
    stream: bool = True
    pretty: bool = True
    show_diffs: bool = False
    show_repo_map: bool = False
    show_model_warnings: bool = True
    verbose: bool = False
    no_pretty: bool = False


@dataclass
class CacheConfig:
    """Caching configuration."""
    cache_prompts: bool = False
    cache_keepalive_pings: int = 60


@dataclass
class RepomapConfig:
    """Repository mapping settings."""
    map_tokens: int = 1024
    map_refresh: str = "auto"
    map_multiplier_no_files: float = 2.0
    repo_map: bool = True


@dataclass
class VoiceConfig:
    """Voice input settings."""
    voice_format: str = "wav"
    voice_language: str = "en"


@dataclass
class AnalyticsConfig:
    """Analytics and logging settings."""
    analytics: bool = True
    analytics_log: Optional[str] = None


@dataclass
class AiderConfig:
    """Main aider configuration."""
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    model_aliases: Dict[str, str] = field(default_factory=dict)
    model: str = "claude-3-5-sonnet-20241022"
    weak_model: Optional[str] = None
    editor_model: Optional[str] = None
    model_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Sub-configurations
    git: GitConfig = field(default_factory=GitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    repomap: RepomapConfig = field(default_factory=RepomapConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    
    # Additional settings
    edit_format: str = "diff"
    encoding: str = "utf-8"
    vim: bool = False
    check_update: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_default_providers()

    def _validate_config(self):
        """Validate configuration consistency."""
        # Validate model references
        resolved_model = self.resolve_model_name(self.model)
        if not resolved_model:
            raise ValueError(f"Invalid model: {self.model}")
        
        if self.weak_model:
            resolved_weak = self.resolve_model_name(self.weak_model)
            if not resolved_weak:
                raise ValueError(f"Invalid weak_model: {self.weak_model}")

    def _setup_default_providers(self):
        """Set up default providers if none are configured."""
        if not self.providers:
            # Create default providers from environment variables
            self._create_default_providers()

    def _create_default_providers(self):
        """Create default provider configurations from environment."""
        # OpenAI provider
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        if openai_key:
            self.providers["openai_default"] = ProviderConfig(
                type="openai",
                api_key=openai_key,
                base_url=openai_base,
                models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
            )

        # Anthropic provider
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.providers["anthropic_default"] = ProviderConfig(
                type="anthropic",
                api_key=anthropic_key,
                models=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
            )

        # Ollama provider
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.providers["ollama_default"] = ProviderConfig(
            type="ollama",
            base_url=ollama_host,
            models=["llama3:8b", "codellama:7b"]
        )

        # Add other providers based on environment variables
        provider_env_map = {
            "groq": ("GROQ_API_KEY", None),
            "deepseek": ("DEEPSEEK_API_KEY", None),
            "gemini": ("GEMINI_API_KEY", None),
            "xai": ("XAI_API_KEY", None),
            "cohere": ("COHERE_API_KEY", None),
        }
        
        for provider_type, (key_env, base_env) in provider_env_map.items():
            api_key = os.getenv(key_env)
            if api_key:
                base_url = os.getenv(base_env) if base_env else None
                self.providers[f"{provider_type}_default"] = ProviderConfig(
                    type=provider_type,
                    api_key=api_key,
                    base_url=base_url
                )

    def detect_environment_conflicts(self):
        """
        Detect conflicts between config file settings and environment variables.
        Returns a list of conflict descriptions.
        """
        conflicts = []
        
        for endpoint_name, provider in self.providers.items():
            provider_type = provider.type.upper()
            
            # Check API key conflicts
            env_key_var = f"{provider_type}_API_KEY"
            env_api_key = os.getenv(env_key_var)
            if env_api_key and provider.api_key and env_api_key != provider.api_key:
                conflicts.append(
                    f"API key mismatch: {env_key_var}={env_api_key[:8]}... differs from "
                    f"config file {endpoint_name}.api_key={provider.api_key[:8]}..."
                )
            
            # Check base URL conflicts  
            base_url_map = {
                "OPENAI": "OPENAI_API_BASE",
                "OLLAMA": "OLLAMA_HOST",
                "ANTHROPIC": "ANTHROPIC_API_BASE",
                "GROQ": "GROQ_API_BASE",
                "DEEPSEEK": "DEEPSEEK_API_BASE",
            }
            
            env_base_var = base_url_map.get(provider_type, f"{provider_type}_API_BASE")
            env_base_url = os.getenv(env_base_var)
            if env_base_url and provider.base_url and env_base_url != provider.base_url:
                conflicts.append(
                    f"Base URL mismatch: {env_base_var}={env_base_url} differs from "
                    f"config file {endpoint_name}.base_url={provider.base_url}"
                )
        
        return conflicts

    def resolve_model_name(self, model_spec: str) -> Optional[str]:
        """
        Resolve a model specification to endpoint_name/model_name format.
        
        Args:
            model_spec: Model specification (alias, endpoint/model, or just model)
            
        Returns:
            Resolved model name in endpoint_name/model_name format
        """
        # Check if it's an alias
        if model_spec in self.model_aliases:
            return self.model_aliases[model_spec]
        
        # If already in endpoint/model format, validate it exists
        if "/" in model_spec:
            endpoint_name, model_name = model_spec.split("/", 1)
            if endpoint_name in self.providers:
                provider = self.providers[endpoint_name]
                if not provider.models or model_name in provider.models:
                    return model_spec
            return None
        
        # Try to find a provider that supports this model
        for endpoint_name, provider in self.providers.items():
            if not provider.models or model_spec in provider.models:
                return f"{endpoint_name}/{model_spec}"
        
        return None

    def get_provider_for_model(self, model_spec: str) -> Optional[ProviderConfig]:
        """Get the provider configuration for a given model."""
        resolved = self.resolve_model_name(model_spec)
        if not resolved or "/" not in resolved:
            return None
        
        endpoint_name = resolved.split("/", 1)[0]
        return self.providers.get(endpoint_name)

    def get_model_name(self, model_spec: str) -> Optional[str]:
        """Extract just the model name from a model specification."""
        resolved = self.resolve_model_name(model_spec)
        if not resolved or "/" not in resolved:
            return model_spec
        
        return resolved.split("/", 1)[1]

    def get_model_settings(self, model_spec: str) -> Dict[str, Any]:
        """Get model-specific settings."""
        resolved = self.resolve_model_name(model_spec)
        if not resolved:
            return {}
        
        settings = {}
        
        # Check for exact match
        if resolved in self.model_settings:
            settings.update(self.model_settings[resolved])
        
        # Check for provider wildcard match
        endpoint_name = resolved.split("/", 1)[0]
        wildcard_key = f"{endpoint_name}/*"
        if wildcard_key in self.model_settings:
            wildcard_settings = self.model_settings[wildcard_key].copy()
            wildcard_settings.update(settings)  # Exact matches override wildcards
            settings = wildcard_settings
        
        return settings


class ConfigManager:
    """Manages loading and saving of aider configuration."""
    
    DEFAULT_CONFIG_NAMES = [".aider.yml", ".aider.yaml"]
    
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
        
        if config_path:
            self.config_path = config_path
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Convert provider configurations
            providers = {}
            if 'providers' in config_data:
                for name, provider_data in config_data['providers'].items():
                    providers[name] = ProviderConfig(**provider_data)
            config_data['providers'] = providers
            
            # Convert sub-configurations
            if 'git' in config_data:
                config_data['git'] = GitConfig(**config_data['git'])
            if 'output' in config_data:
                config_data['output'] = OutputConfig(**config_data['output'])
            if 'cache' in config_data:
                config_data['cache'] = CacheConfig(**config_data['cache'])
            if 'repomap' in config_data:
                config_data['repomap'] = RepomapConfig(**config_data['repomap'])
            if 'voice' in config_data:
                config_data['voice'] = VoiceConfig(**config_data['voice'])
            if 'analytics' in config_data:
                config_data['analytics'] = AnalyticsConfig(**config_data['analytics'])
            
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
            config_path = Path.cwd() / ".aider.yml"
        
        # Convert to dictionary
        config_dict = self._config_to_dict(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        self.config_path = config_path

    def _config_to_dict(self, config: AiderConfig) -> Dict[str, Any]:
        """Convert AiderConfig to dictionary for YAML serialization."""
        result = {}
        
        # Providers
        if config.providers:
            result['providers'] = {}
            for name, provider in config.providers.items():
                provider_dict = {
                    'type': provider.type,
                }
                if provider.api_key:
                    provider_dict['api_key'] = provider.api_key
                if provider.base_url:
                    provider_dict['base_url'] = provider.base_url
                if provider.models:
                    provider_dict['models'] = provider.models
                if provider.extra_params:
                    provider_dict.update(provider.extra_params)
                result['providers'][name] = provider_dict
        
        # Model configuration
        if config.model_aliases:
            result['model_aliases'] = config.model_aliases
        
        result['model'] = config.model
        if config.weak_model:
            result['weak_model'] = config.weak_model
        if config.editor_model:
            result['editor_model'] = config.editor_model
        
        if config.model_settings:
            result['model_settings'] = config.model_settings
        
        # Sub-configurations
        result['git'] = self._dataclass_to_dict(config.git)
        result['output'] = self._dataclass_to_dict(config.output)
        result['cache'] = self._dataclass_to_dict(config.cache)
        result['repomap'] = self._dataclass_to_dict(config.repomap)
        result['voice'] = self._dataclass_to_dict(config.voice)
        result['analytics'] = self._dataclass_to_dict(config.analytics)
        
        # Other settings
        result['edit_format'] = config.edit_format
        result['encoding'] = config.encoding
        result['vim'] = config.vim
        result['check_update'] = config.check_update
        
        return result

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary, excluding None values."""
        result = {}
        for key, value in obj.__dict__.items():
            if value is not None:
                result[key] = value
        return result

    def create_sample_config(self, path: Optional[Union[str, Path]] = None, include_env_vars: bool = True) -> Path:
        """Create a sample configuration file, optionally including detected environment variables."""
        if path:
            config_path = Path(path)
        else:
            config_path = Path.cwd() / ".aider.yml"
        
        if include_env_vars:
            # Create config with detected environment variables
            self._create_config_with_env_vars(config_path)
        else:
            # Find the sample file in the aider package
            import aider
            aider_dir = Path(aider.__file__).parent.parent
            sample_file = aider_dir / "aider.yml.sample"
            
            if sample_file.exists():
                # Copy the sample file
                import shutil
                shutil.copy2(sample_file, config_path)
            else:
                # Fallback to creating a basic config
                self._create_basic_config_file(config_path)
        
        return config_path

    def update_existing_config_with_models(self, config_path: Optional[Union[str, Path]] = None) -> Path:
        """Update an existing configuration file with newly discovered models."""
        # Find existing config file
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        else:
            config_path = self.find_config_file()
            if not config_path:
                raise FileNotFoundError("No configuration file found. Use --init-config to create one.")
        
        print(f"Updating configuration file: {config_path}")
        
        # Load existing configuration
        existing_config = self.load_config(config_path)
        
        # Convert providers to dict format for interrogation
        provider_configs = {}
        for name, provider in existing_config.providers.items():
            provider_configs[provider.type] = {
                'type': provider.type,
                'api_key': provider.api_key,
                'base_url': provider.base_url
            }
        
        # Interrogate endpoints for new models
        if provider_configs:
            print("Discovering available models from API endpoints...")
            updated_providers = self.interrogate_endpoints_for_models(provider_configs)
            
            # Update the existing config with new models
            for name, provider in existing_config.providers.items():
                provider_type = provider.type
                if provider_type in updated_providers and 'models' in updated_providers[provider_type]:
                    new_models = updated_providers[provider_type]['models']
                    old_models = provider.models or []
                    provider.models = new_models
                    print(f"  Updated {name}: {len(old_models)} -> {len(new_models)} models")
        
        # Save the updated configuration
        self.save_config(existing_config, config_path)
        print(f"Configuration file updated: {config_path}")
        
        return config_path

    def interrogate_endpoints_for_models(self, detected_providers: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """Query API endpoints to discover available models."""
        import time
        import json
        
        for provider_type, config in detected_providers.items():
            if 'api_key' not in config:
                continue  # Skip providers without API keys
                
            print(f"Discovering models for {provider_type}...")
            
            try:
                models = self._query_provider_models(provider_type, config)
                if models:
                    config['models'] = models
                    print(f"  Found {len(models)} models")
                else:
                    # Use default models if discovery fails
                    config['models'] = self._get_default_models(provider_type)
                    print(f"  Using default models (discovery failed)")
            except Exception as e:
                print(f"  Model discovery failed for {provider_type}: {e}")
                config['models'] = self._get_default_models(provider_type)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return detected_providers

    def _query_provider_models(self, provider_type: str, config: Dict[str, str]) -> List[str]:
        """Query a specific provider for available models."""
        import requests
        
        if provider_type == 'openai':
            return self._query_openai_models(config)
        elif provider_type == 'anthropic':
            return self._query_anthropic_models(config)
        elif provider_type == 'groq':
            return self._query_groq_models(config)
        elif provider_type == 'ollama':
            return self._query_ollama_models(config)
        # Add more providers as needed
        
        return []

    def _query_openai_models(self, config: Dict[str, str]) -> List[str]:
        """Query OpenAI API for available models."""
        import requests
        
        base_url = config.get('base_url', 'https://api.openai.com/v1')
        api_key = config.get('api_key')
        
        if not api_key:
            return []
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', []) 
                         if model['id'].startswith(('gpt-', 'o1-', 'chatgpt-'))]
                return sorted(models)
        except Exception:
            pass
        
        return []

    def _query_anthropic_models(self, config: Dict[str, str]) -> List[str]:
        """Query Anthropic API for available models (Claude models are well-known)."""
        # Anthropic doesn't have a public models endpoint, so we use known models
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

    def _query_groq_models(self, config: Dict[str, str]) -> List[str]:
        """Query Groq API for available models."""
        import requests
        
        api_key = config.get('api_key')
        if not api_key:
            return []
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', [])]
                return sorted(models)
        except Exception:
            pass
        
        return []

    def _query_ollama_models(self, config: Dict[str, str]) -> List[str]:
        """Query Ollama API for available models."""
        import requests
        
        base_url = config.get('base_url', 'http://localhost:11434')
        
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return sorted(models)
        except Exception:
            pass
        
        return []

    def _get_default_models(self, provider_type: str) -> List[str]:
        """Get default models for a provider when discovery fails."""
        defaults = {
            'openai': ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            'anthropic': ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
            'groq': ["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            'ollama': ["llama3:8b", "codellama:7b", "mistral:7b"],
            'deepseek': ["deepseek-chat", "deepseek-coder"],
            'gemini': ["gemini-1.5-pro", "gemini-1.5-flash"],
            'xai': ["grok-1"],
            'cohere': ["command-r", "command-r-plus"]
        }
        return defaults.get(provider_type, [])

    def _create_config_with_env_vars(self, config_path: Path):
        """Create a configuration file with detected environment variables."""
        detected_providers = {}
        detected_settings = {}
        
        # Detect provider configurations from environment
        env_vars_to_check = {
            'OPENAI_API_KEY': ('openai', 'api_key'),
            'OPENAI_API_BASE': ('openai', 'base_url'),
            'ANTHROPIC_API_KEY': ('anthropic', 'api_key'),
            'ANTHROPIC_API_BASE': ('anthropic', 'base_url'),
            'GROQ_API_KEY': ('groq', 'api_key'),
            'GROQ_API_BASE': ('groq', 'base_url'),
            'DEEPSEEK_API_KEY': ('deepseek', 'api_key'),
            'DEEPSEEK_API_BASE': ('deepseek', 'base_url'),
            'GEMINI_API_KEY': ('gemini', 'api_key'),
            'XAI_API_KEY': ('xai', 'api_key'),
            'COHERE_API_KEY': ('cohere', 'api_key'),
            'OLLAMA_HOST': ('ollama', 'base_url'),
        }
        
        for env_var, (provider_type, setting) in env_vars_to_check.items():
            value = os.getenv(env_var)
            if value:
                if provider_type not in detected_providers:
                    detected_providers[provider_type] = {'type': provider_type}
                detected_providers[provider_type][setting] = value
        
        # Interrogate endpoints for models
        if detected_providers:
            print("Discovering available models from API endpoints...")
            detected_providers = self.interrogate_endpoints_for_models(detected_providers)
        
        # Create config content with detected values
        config_content = self._generate_config_content_with_env_vars(detected_providers)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

    def _generate_config_content_with_env_vars(self, detected_providers: Dict[str, Dict[str, str]]) -> str:
        """Generate configuration file content with detected environment variables."""
        content = """# Aider Configuration File
# Generated with detected environment variables
# For full documentation: https://aider.chat/docs/config/

"""
        
        if detected_providers:
            content += "providers:\n"
            
            for provider_type, config in detected_providers.items():
                endpoint_name = f"{provider_type}_main"
                content += f"  {endpoint_name}:\n"
                content += f"    type: {provider_type}\n"
                
                if 'api_key' in config:
                    # Show truncated key for reference
                    truncated_key = config['api_key'][:8] + "..." if len(config['api_key']) > 8 else config['api_key']
                    content += f"    api_key: \"{config['api_key']}\"\n"
                
                if 'base_url' in config:
                    content += f"    base_url: \"{config['base_url']}\"\n"
                
                # Add discovered or default models
                if 'models' in config and config['models']:
                    models_str = ", ".join([f'"{model}"' for model in config['models']])
                    content += f"    models: [{models_str}]\n"
                
                content += "\n"
        else:
            # No environment variables detected, create example config
            content += """providers:
  openai_main:
    type: openai
    api_key: "sk-your-openai-key-here"
    base_url: "https://api.openai.com/v1"
    models: ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    
  anthropic_main:
    type: anthropic
    api_key: "sk-ant-your-key-here"
    models: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
    
  local_ollama:
    type: ollama
    base_url: "http://localhost:11434"
    models: ["llama3:8b", "codellama:7b", "mistral:7b"]

"""
        
        content += """# Model aliases for convenience
model_aliases:
  sonnet: "anthropic_main/claude-3-5-sonnet-20241022"
  gpt4o: "openai_main/gpt-4o"
  gpt4: "openai_main/gpt-4"

# Default model to use
model: "sonnet"

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
# Copy this to .aider.yml and customize as needed

providers:
  openai_main:
    type: openai
    api_key: "sk-your-openai-key-here"
    base_url: "https://api.openai.com/v1"
    models: ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]
    
  anthropic_main:
    type: anthropic
    api_key: "sk-ant-your-key-here"
    models: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
    
  local_ollama:
    type: ollama
    base_url: "http://localhost:11434"
    models: ["llama3:8b", "codellama:7b"]

model_aliases:
  sonnet: "anthropic_main/claude-3-5-sonnet-20241022"
  gpt4: "openai_main/gpt-4"
  gpt4o: "openai_main/gpt-4o"

model: "sonnet"
weak_model: "openai_main/gpt-3.5-turbo"

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
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(basic_config)


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_path: Optional[Union[str, Path]] = None) -> AiderConfig:
    """Load aider configuration."""
    return config_manager.load_config(config_path)


def save_config(config: AiderConfig, config_path: Optional[Union[str, Path]] = None):
    """Save aider configuration."""
    config_manager.save_config(config, config_path)


def create_sample_config(path: Optional[Union[str, Path]] = None) -> Path:
    """Create a sample configuration file."""
    return config_manager.create_sample_config(path)