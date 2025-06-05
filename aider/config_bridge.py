"""
Configuration bridge for aider's unified configuration system.

This module provides backwards compatibility between the legacy multi-source 
configuration system and the new unified .aider.yml configuration system.
It allows for gradual migration while maintaining all existing functionality.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from aider.args import get_parser
from aider.config import AiderConfig, ConfigManager, ProviderConfig
from aider.models import endpoint_model_manager


class ConfigBridge:
    """
    Bridge between legacy and new configuration systems.
    
    This class handles:
    1. Loading new .aider.yml config files when they exist
    2. Converting legacy CLI args + env vars to AiderConfig format
    3. Providing a unified interface for both systems
    4. Gradual migration support
    """
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = None
        self.legacy_args = None
        self.use_legacy = False

    def load_configuration(self, argv=None, git_root=None):
        """
        Load configuration using the appropriate system.
        
        Priority:
        1. If .aider.yml exists -> use new unified config system
        2. Otherwise -> use legacy args + env var system with conversion
        """
        if argv is None:
            argv = sys.argv[1:]

        # Check for new config file
        config_file = self._find_new_config_file(git_root)
        
        if config_file and self._should_use_new_config(argv):
            # Use new configuration system
            self.use_legacy = False
            return self._load_new_config(config_file, argv)
        else:
            # Use legacy system with conversion
            self.use_legacy = True
            return self._load_legacy_config_with_conversion(argv, git_root)

    def _find_new_config_file(self, git_root=None):
        """Find .aider.yml config file in search path."""
        search_paths = [Path.cwd()]
        
        if git_root:
            search_paths.append(Path(git_root))
        
        search_paths.append(Path.home())
        
        for path in search_paths:
            for config_name in [".aider.yml", ".aider.yaml"]:
                config_file = path / config_name
                if config_file.exists():
                    return config_file
        
        return None

    def _should_use_new_config(self, argv):
        """Determine if we should use the new config system."""
        # Check for special flags that indicate new system usage
        new_system_flags = [
            "--init-config", "--list-models",
            "--validate-config", "--show-config"
        ]
        
        return any(flag in argv for flag in new_system_flags)

    def _load_new_config(self, config_file, argv):
        """Load configuration using the new unified system."""
        try:
            # Load the configuration file
            self.config = self.config_manager.load_config(config_file)
            
            # Apply any CLI overrides
            self._apply_cli_overrides_to_new_config(argv)
            
            # Set up the endpoint model manager
            endpoint_model_manager.set_config(self.config)
            
            # Convert to legacy-compatible namespace for backwards compatibility
            legacy_args = self._convert_new_config_to_legacy_args()
            
            return legacy_args, self.config
            
        except Exception as e:
            print(f"Error loading new configuration: {e}", file=sys.stderr)
            print("Falling back to legacy configuration system...")
            return self._load_legacy_config_with_conversion(argv, None)

    def _load_legacy_config_with_conversion(self, argv, git_root):
        """Load configuration using legacy system and convert to new format."""
        
        # Use existing legacy argument parsing
        default_config_files = self._get_legacy_config_files(git_root)
        parser = get_parser(default_config_files, git_root)
        
        # Load environment variables
        from aider.main import load_dotenv_files
        load_dotenv_files(git_root, getattr(parser.parse_known_args(argv)[0], 'env_file', '.env'), 'utf-8')
        
        # Parse arguments
        self.legacy_args = parser.parse_args(argv)
        
        # Convert legacy configuration to new format
        self.config = self._convert_legacy_to_new_config(self.legacy_args)
        
        # Set up the endpoint model manager with converted config
        endpoint_model_manager.set_config(self.config)
        
        return self.legacy_args, self.config

    def _get_legacy_config_files(self, git_root):
        """Get legacy config file search paths."""
        conf_fname = Path(".aider.conf.yml")
        default_config_files = []
        
        try:
            default_config_files += [conf_fname.resolve()]  # CWD
        except OSError:
            pass

        if git_root:
            git_conf = Path(git_root) / conf_fname  # git root
            if git_conf not in default_config_files:
                default_config_files.append(git_conf)
        
        default_config_files.append(Path.home() / conf_fname)  # homedir
        return list(map(str, default_config_files))

    def _convert_legacy_to_new_config(self, args) -> AiderConfig:
        """Convert legacy args to new AiderConfig format."""
        
        # Create provider configurations from environment and args
        providers = {}
        
        # OpenAI provider
        openai_key = getattr(args, 'openai_api_key', None) or os.getenv('OPENAI_API_KEY')
        openai_base = getattr(args, 'openai_api_base', None) or os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        if openai_key:
            providers["openai_default"] = ProviderConfig(
                type="openai",
                api_key=openai_key,
                base_url=openai_base,
                models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            )

        # Anthropic provider
        anthropic_key = getattr(args, 'anthropic_api_key', None) or os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            providers["anthropic_default"] = ProviderConfig(
                type="anthropic",
                api_key=anthropic_key,
                models=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"]
            )

        # Ollama provider (always available)
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        providers["ollama_default"] = ProviderConfig(
            type="ollama",
            base_url=ollama_host,
            models=["llama3:8b", "codellama:7b", "mistral:7b"]
        )

        # Add other providers based on environment variables
        provider_env_map = {
            "groq": ("GROQ_API_KEY", None),
            "deepseek": ("DEEPSEEK_API_KEY", None),
            "gemini": ("GEMINI_API_KEY", None),
        }
        
        for provider_type, (key_env, base_env) in provider_env_map.items():
            api_key = os.getenv(key_env)
            if api_key:
                base_url = os.getenv(base_env) if base_env else None
                providers[f"{provider_type}_default"] = ProviderConfig(
                    type=provider_type,
                    api_key=api_key,
                    base_url=base_url
                )

        # Create the configuration
        from aider.config import GitConfig, OutputConfig, CacheConfig, RepomapConfig, VoiceConfig, AnalyticsConfig
        
        config = AiderConfig(
            providers=providers,
            model=getattr(args, 'model', None) or 'gpt-4o',
            weak_model=getattr(args, 'weak_model', None),
            editor_model=getattr(args, 'editor_model', None),
            edit_format=getattr(args, 'edit_format', None) or 'diff',
            git=GitConfig(
                auto_commits=getattr(args, 'auto_commits', True),
                dirty_commits=getattr(args, 'dirty_commits', True),
                attribute_author=getattr(args, 'attribute_author', True),
                attribute_committer=getattr(args, 'attribute_committer', True),
                dry_run=getattr(args, 'dry_run', False),
            ),
            output=OutputConfig(
                user_input_color=getattr(args, 'user_input_color', '#00cc00'),
                tool_output_color=getattr(args, 'tool_output_color', None),
                tool_error_color=getattr(args, 'tool_error_color', '#FF2222'),
                tool_warning_color=getattr(args, 'tool_warning_color', '#FFA500'),
                assistant_output_color=getattr(args, 'assistant_output_color', '#0088ff'),
                code_theme=getattr(args, 'code_theme', 'default'),
                stream=getattr(args, 'stream', True),
                pretty=getattr(args, 'pretty', True),
                show_diffs=getattr(args, 'show_diffs', False),
                verbose=getattr(args, 'verbose', False),
            ),
            cache=CacheConfig(
                cache_prompts=getattr(args, 'cache_prompts', False),
                cache_keepalive_pings=getattr(args, 'cache_keepalive_pings', 0),
            ),
            repomap=RepomapConfig(
                map_tokens=getattr(args, 'map_tokens', 1024),
                map_refresh=getattr(args, 'map_refresh', 'auto'),
                map_multiplier_no_files=getattr(args, 'map_multiplier_no_files', 2.0),
            ),
            voice=VoiceConfig(
                voice_format=getattr(args, 'voice_format', 'wav'),
                voice_language=getattr(args, 'voice_language', 'en'),
            ),
            analytics=AnalyticsConfig(
                analytics=getattr(args, 'analytics', True),
                analytics_log=getattr(args, 'analytics_log', None),
            ),
            vim=getattr(args, 'vim', False),
            encoding=getattr(args, 'encoding', 'utf-8'),
            check_update=getattr(args, 'check_update', True),
        )
        
        return config

    def _convert_new_config_to_legacy_args(self):
        """Convert new config format to legacy args namespace for backwards compatibility."""
        from argparse import Namespace
        
        # Create a namespace that mimics the legacy args structure
        args = Namespace()
        
        # Map configuration values to legacy arg names
        args.model = self.config.model
        args.weak_model = self.config.weak_model  
        args.editor_model = self.config.editor_model
        args.edit_format = self.config.edit_format
        args.vim = self.config.vim
        args.encoding = self.config.encoding
        args.check_update = self.config.check_update
        
        # Git settings
        args.auto_commits = self.config.git.auto_commits
        args.dirty_commits = self.config.git.dirty_commits
        args.attribute_author = self.config.git.attribute_author
        args.attribute_committer = self.config.git.attribute_committer
        args.dry_run = self.config.git.dry_run
        
        # Output settings
        args.user_input_color = self.config.output.user_input_color
        args.tool_output_color = self.config.output.tool_output_color
        args.tool_error_color = self.config.output.tool_error_color
        args.tool_warning_color = self.config.output.tool_warning_color
        args.assistant_output_color = self.config.output.assistant_output_color
        args.code_theme = self.config.output.code_theme
        args.stream = self.config.output.stream
        args.pretty = self.config.output.pretty
        args.show_diffs = self.config.output.show_diffs
        args.verbose = self.config.output.verbose
        
        # Cache settings
        args.cache_prompts = self.config.cache.cache_prompts
        args.cache_keepalive_pings = self.config.cache.cache_keepalive_pings
        
        # Repomap settings
        args.map_tokens = self.config.repomap.map_tokens
        args.map_refresh = self.config.repomap.map_refresh
        args.map_multiplier_no_files = self.config.repomap.map_multiplier_no_files
        
        # Voice settings
        args.voice_format = self.config.voice.voice_format
        args.voice_language = self.config.voice.voice_language
        
        # Analytics settings
        args.analytics = self.config.analytics.analytics
        args.analytics_log = self.config.analytics.analytics_log
        
        # Set default values for other required legacy fields
        args.files = []
        args.git = True
        args.verify_ssl = True
        args.timeout = None
        
        return args

    def _apply_cli_overrides_to_new_config(self, argv):
        """Apply command line overrides to new configuration."""
        # Simple CLI override parser for essential arguments
        i = 0
        while i < len(argv):
            arg = argv[i]
            
            if arg == "--model" and i + 1 < len(argv):
                self.config.model = argv[i + 1]
                i += 1
            elif arg == "--weak-model" and i + 1 < len(argv):
                self.config.weak_model = argv[i + 1]
                i += 1
            elif arg == "--editor-model" and i + 1 < len(argv):
                self.config.editor_model = argv[i + 1]
                i += 1
            elif arg == "--edit-format" and i + 1 < len(argv):
                self.config.edit_format = argv[i + 1]
                i += 1
            elif arg == "--verbose":
                self.config.output.verbose = True
            elif arg == "--vim":
                self.config.vim = True
            
            i += 1

    def get_model_manager(self):
        """Get the endpoint-aware model manager."""
        return endpoint_model_manager

    def create_model(self, model_spec, **kwargs):
        """Create a model using the endpoint-aware system."""
        return endpoint_model_manager.create_model(model_spec, **kwargs)


# Global configuration bridge instance
config_bridge = ConfigBridge()