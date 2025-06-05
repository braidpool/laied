#!/usr/bin/env python

"""
Simplified argument parser for aider's unified configuration system.

This module provides a streamlined CLI interface that focuses on essential
operations while delegating most configuration to the .aider.yml file.
"""

import argparse
import os
import sys
from pathlib import Path

from aider import __version__
from aider.config import AiderConfig, ConfigManager, load_config, create_sample_config


class ConfigArgumentParser:
    """
    Argument parser that integrates with the unified configuration system.
    
    This parser handles essential CLI operations and loads configuration from
    .aider.yml files, providing backwards compatibility where needed.
    """
    
    def __init__(self):
        self.parser = self._create_parser()
        self.config_manager = ConfigManager()
        self.config = None

    def _create_parser(self):
        """Create the simplified argument parser."""
        parser = argparse.ArgumentParser(
            description="aider is AI pair programming in your terminal",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Configuration is now managed through a single .aider.yml file.
Use 'aider --init-config' to create a sample configuration file.

For help and documentation: https://aider.chat
"""
        )

        # Essential operational arguments
        parser.add_argument(
            "--version",
            action="version", 
            version=f"aider {__version__}",
            help="Show aider version"
        )

        parser.add_argument(
            "--config", 
            metavar="FILE",
            help="Path to configuration file (default: search for .aider.yml)"
        )

        parser.add_argument(
            "--init-config",
            action="store_true",
            help="Create a sample .aider.yml configuration file and exit"
        )

        parser.add_argument(
            "--migrate-config", 
            action="store_true",
            help="Migrate existing configuration to new .aider.yml format and exit"
        )

        parser.add_argument(
            "--list-models",
            action="store_true", 
            help="List available models from configuration and exit"
        )

        parser.add_argument(
            "--validate-config",
            action="store_true",
            help="Validate configuration file and exit"
        )

        parser.add_argument(
            "--show-config",
            action="store_true",
            help="Show current configuration and exit"
        )

        # Help and information
        parser.add_argument(
            "--help-config",
            action="store_true",
            help="Show configuration file help and examples"
        )

        # Files to add to the chat (positional arguments)
        parser.add_argument(
            "files",
            nargs="*",
            help="Files to add to the chat session"
        )

        # Legacy compatibility arguments (override config file)
        parser.add_argument(
            "--model",
            help="Model to use (overrides config file setting)"
        )

        parser.add_argument(
            "--weak-model",
            help="Weak model to use (overrides config file setting)"
        )

        parser.add_argument(
            "--editor-model", 
            help="Editor model to use (overrides config file setting)"
        )

        parser.add_argument(
            "--edit-format",
            choices=["diff", "whole", "udiff", "editor-diff", "editor-whole"],
            help="Edit format to use (overrides config file setting)"
        )

        # Debug and verbose modes
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )

        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug output"
        )

        # Backwards compatibility for common environment variable settings
        parser.add_argument(
            "--openai-api-key",
            help="OpenAI API key (consider using config file instead)"
        )

        parser.add_argument(
            "--anthropic-api-key", 
            help="Anthropic API key (consider using config file instead)"
        )

        return parser

    def parse_args(self, args=None):
        """Parse arguments and load configuration."""
        if args is None:
            args = sys.argv[1:]

        parsed_args = self.parser.parse_args(args)

        # Handle special actions that should exit immediately
        if parsed_args.init_config:
            self._handle_init_config()
            sys.exit(0)

        if parsed_args.migrate_config:
            self._handle_migrate_config()
            sys.exit(0)

        if parsed_args.help_config:
            self._show_config_help()
            sys.exit(0)

        # Load configuration
        try:
            self.config = load_config(parsed_args.config)
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            print("Use 'aider --init-config' to create a sample configuration file.")
            sys.exit(1)

        # Apply CLI overrides to configuration
        self._apply_cli_overrides(parsed_args)

        # Handle other special actions that need config
        if parsed_args.validate_config:
            self._handle_validate_config()
            sys.exit(0)

        if parsed_args.list_models:
            self._handle_list_models()
            sys.exit(0)

        if parsed_args.show_config:
            self._handle_show_config()
            sys.exit(0)

        # Set up legacy environment variables for backwards compatibility
        self._setup_legacy_environment()

        return parsed_args, self.config

    def _apply_cli_overrides(self, args):
        """Apply command line argument overrides to configuration."""
        if args.model:
            self.config.model = args.model

        if args.weak_model:
            self.config.weak_model = args.weak_model

        if args.editor_model:
            self.config.editor_model = args.editor_model

        if args.edit_format:
            self.config.edit_format = args.edit_format

        if args.verbose:
            self.config.output.verbose = True

        # Handle legacy API key arguments
        if args.openai_api_key:
            # Add or update OpenAI provider with the provided key
            if "openai_default" not in self.config.providers:
                from aider.config import ProviderConfig
                self.config.providers["openai_default"] = ProviderConfig(
                    type="openai",
                    api_key=args.openai_api_key,
                    base_url="https://api.openai.com/v1"
                )
            else:
                self.config.providers["openai_default"].api_key = args.openai_api_key

        if args.anthropic_api_key:
            # Add or update Anthropic provider with the provided key
            if "anthropic_default" not in self.config.providers:
                from aider.config import ProviderConfig
                self.config.providers["anthropic_default"] = ProviderConfig(
                    type="anthropic",
                    api_key=args.anthropic_api_key
                )
            else:
                self.config.providers["anthropic_default"].api_key = args.anthropic_api_key

    def _setup_legacy_environment(self):
        """Set up environment variables for backwards compatibility."""
        # Set up environment variables based on the current configuration
        # This ensures existing code that depends on environment variables continues to work
        for endpoint_name, provider in self.config.providers.items():
            if provider.api_key:
                env_key = f"{provider.type.upper()}_API_KEY"
                os.environ[env_key] = provider.api_key

            if provider.base_url:
                base_url_map = {
                    "openai": "OPENAI_API_BASE",
                    "ollama": "OLLAMA_HOST",
                    "anthropic": "ANTHROPIC_API_BASE",
                }
                env_key = base_url_map.get(provider.type)
                if env_key:
                    os.environ[env_key] = provider.base_url

    def _handle_init_config(self):
        """Handle --init-config flag."""
        try:
            config_path = create_sample_config()
            print(f"Created sample configuration file: {config_path}")
            print("\nEdit this file to configure your models and API keys.")
            print("Documentation: https://aider.chat/docs/config.html")
        except Exception as e:
            print(f"Error creating configuration file: {e}", file=sys.stderr)
            sys.exit(1)

    def _handle_migrate_config(self):
        """Handle --migrate-config flag."""
        print("Configuration migration is not yet implemented.")
        print("Please manually create a .aider.yml file using 'aider --init-config'")
        print("and transfer your settings from environment variables and CLI args.")

    def _handle_validate_config(self):
        """Handle --validate-config flag."""
        try:
            # Configuration was already loaded successfully
            print("✓ Configuration file is valid")
            
            # Validate model references
            if self.config.model:
                resolved = self.config.resolve_model_name(self.config.model)
                if resolved:
                    print(f"✓ Main model '{self.config.model}' resolves to '{resolved}'")
                else:
                    print(f"✗ Main model '{self.config.model}' could not be resolved")

            if self.config.weak_model:
                resolved = self.config.resolve_model_name(self.config.weak_model)
                if resolved:
                    print(f"✓ Weak model '{self.config.weak_model}' resolves to '{resolved}'")
                else:
                    print(f"✗ Weak model '{self.config.weak_model}' could not be resolved")

            # Validate providers
            for name, provider in self.config.providers.items():
                if provider.api_key:
                    print(f"✓ Provider '{name}' has API key configured")
                else:
                    print(f"! Provider '{name}' has no API key (may be required)")

        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            sys.exit(1)

    def _handle_list_models(self):
        """Handle --list-models flag."""
        print("Available Models:")
        print("==================")

        for endpoint_name, provider in self.config.providers.items():
            print(f"\n{endpoint_name} ({provider.type}):")
            if provider.models:
                for model in provider.models:
                    print(f"  - {endpoint_name}/{model}")
            else:
                print(f"  - {endpoint_name}/<any-model>")

        if self.config.model_aliases:
            print("\nModel Aliases:")
            print("==============")
            for alias, target in self.config.model_aliases.items():
                print(f"  {alias} -> {target}")

    def _handle_show_config(self):
        """Handle --show-config flag."""
        print("Current Configuration:")
        print("=====================")
        print(f"Config file: {self.config_manager.config_path or 'default'}")
        print(f"Main model: {self.config.model}")
        if self.config.weak_model:
            print(f"Weak model: {self.config.weak_model}")
        if self.config.editor_model:
            print(f"Editor model: {self.config.editor_model}")
        print(f"Edit format: {self.config.edit_format}")
        print(f"Providers: {len(self.config.providers)}")

    def _show_config_help(self):
        """Show configuration file help and examples."""
        print("""
Aider Configuration Help
========================

Aider now uses a single .aider.yml configuration file for all settings.

Configuration File Locations (searched in order):
1. Path specified with --config
2. .aider.yml in current directory
3. .aider.yml in git repository root
4. .aider.yml in home directory

Sample Configuration File:
--------------------------

# Multiple endpoints per provider type
providers:
  openai_main:
    type: openai
    api_key: "sk-your-openai-key"
    base_url: "https://api.openai.com/v1"
    models: ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    
  openai_azure:
    type: openai
    api_key: "azure-key"
    base_url: "https://mycompany.openai.azure.com"
    models: ["gpt-4-azure"]
    
  local_ollama:
    type: ollama
    base_url: "http://localhost:11434"
    models: ["llama3:8b", "codellama:7b"]
    
  remote_ollama:
    type: ollama
    base_url: "http://gpu-server:11434"
    models: ["llama3:70b"]

# Model aliases for easy switching  
model_aliases:
  sonnet: "anthropic_main/claude-3-5-sonnet-20241022"
  gpt4: "openai_main/gpt-4"
  gpt4-azure: "openai_azure/gpt-4-azure"
  local-llama: "local_ollama/llama3:8b"
  gpu-llama: "remote_ollama/llama3:70b"

# Default models
model: "sonnet"
weak_model: "openai_main/gpt-3.5-turbo"
editor_model: "local_ollama/codellama:7b"

# Model-specific settings
model_settings:
  "openai_main/gpt-4":
    max_tokens: 4096
    temperature: 0.1
  "local_ollama/*":  # Wildcard for all local models
    temperature: 0.0

# Git settings
git:
  auto_commits: true
  commit_prefix: "aider: "

# Output settings
output:
  user_input_color: "#00cc00"
  pretty: true
  stream: true

# Other settings
edit_format: "diff"
vim: false

Commands:
---------
aider                           # Start aider with config file settings
aider --init-config             # Create sample .aider.yml file
aider --migrate-config          # Convert old settings to new format
aider --list-models             # Show available models
aider --validate-config         # Check configuration file
aider --show-config             # Show current config summary
aider --config /path/to/config  # Use specific config file

Legacy Compatibility:
--------------------
aider --model gpt-4             # Override model from command line
aider --openai-api-key sk-...   # Set API key from command line

For more information: https://aider.chat/docs/config.html
""")


# Global parser instance for backwards compatibility
_global_parser = None


def get_parser(default_config_files=None, git_root=None):
    """
    Get argument parser for backwards compatibility.
    
    This function maintains the same interface as the original args.py
    but returns a simplified parser that works with the new config system.
    """
    global _global_parser
    if _global_parser is None:
        _global_parser = ConfigArgumentParser()
    return _global_parser.parser


def parse_args(args=None):
    """
    Parse command line arguments and return (args, config).
    
    This function provides backwards compatibility with existing code
    that expects the args object and configuration data.
    """
    parser = ConfigArgumentParser()
    return parser.parse_args(args)


if __name__ == "__main__":
    args, config = parse_args()
    print(f"Parsed args: {args}")
    print(f"Config: {config}")