"""
Unified configuration system for aider.

This module provides a single, consistent way to configure aider through
configuration files, replacing the complex multi-source system of CLI args,
environment variables, and multiple config file formats.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelParameterOverrides:
    """Model parameter overrides for fine-tuning behavior."""

    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    streaming: Optional[bool] = None
    use_system_prompt: Optional[bool] = None
    cache_prompts: Optional[bool] = None


@dataclass
class ProviderConfig:
    """Configuration for a single provider endpoint."""

    type: str  # openai, ollama, anthropic, etc.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    model_overrides: Dict[str, ModelParameterOverrides] = field(default_factory=dict)

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
            self.providers["openai"] = ProviderConfig(
                type="openai",
                api_key=openai_key,
                base_url=openai_base,
                models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            )

        # Anthropic provider
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.providers["anthropic"] = ProviderConfig(
                type="anthropic",
                api_key=anthropic_key,
                models=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
            )

        # Ollama provider
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.providers["ollama"] = ProviderConfig(
            type="ollama", base_url=ollama_host, models=["llama3:8b", "codellama:7b"]
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
                self.providers[provider_type] = ProviderConfig(
                    type=provider_type, api_key=api_key, base_url=base_url
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
        # First, prefer providers that explicitly have this model in their discovered models
        for endpoint_name, provider in self.providers.items():
            if provider.models and model_spec in provider.models:
                return f"{endpoint_name}/{model_spec}"

        # Fallback: try providers with no explicit model list (legacy support)
        for endpoint_name, provider in self.providers.items():
            if not provider.models:
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
