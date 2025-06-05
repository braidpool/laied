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

    def create_sample_config(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Create a sample configuration file by copying the sample template."""
        if path:
            config_path = Path(path)
        else:
            config_path = Path.cwd() / ".aider.yml"
        
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