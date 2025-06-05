#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

from aider import __version__


def get_simple_parser():
    """
    Simplified argument parser for the unified configuration system.
    
    This parser only handles essential CLI operations and configuration file
    specification. All other settings are handled through the configuration file.
    """
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
        help="Create a sample .aider.yml configuration file"
    )

    parser.add_argument(
        "--migrate-config",
        action="store_true",
        help="Migrate existing configuration to new .aider.yml format"
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

    # Legacy support for model specification (for backwards compatibility)
    parser.add_argument(
        "--model",
        help="Model to use (overrides config file setting)"
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

    return parser


def parse_args(args=None):
    """Parse command line arguments using the simplified parser."""
    parser = get_simple_parser()
    
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parser.parse_args(args)
    
    # Handle special actions that should exit immediately
    if parsed_args.help_config:
        show_config_help()
        sys.exit(0)
    
    return parsed_args


def show_config_help():
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

# Providers define endpoints and API keys
providers:
  openai_main:
    type: openai
    api_key: "sk-your-openai-key"
    base_url: "https://api.openai.com/v1" 
    models: ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    
  local_ollama:
    type: ollama
    base_url: "http://localhost:11434"
    models: ["llama3:8b", "codellama:7b"]
    
  anthropic_main:
    type: anthropic
    api_key: "sk-ant-your-key"
    models: ["claude-3-5-sonnet-20241022"]

# Model aliases for easy switching
model_aliases:
  sonnet: "anthropic_main/claude-3-5-sonnet-20241022"
  gpt4: "openai_main/gpt-4"
  local-llama: "local_ollama/llama3:8b"

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
aider --config /path/to/config  # Use specific config file

For more information: https://aider.chat/docs/config.html
""")


def create_sample_config(path: Path = None):
    """Create a sample configuration file."""
    if path is None:
        path = Path.cwd() / ".aider.yml"
    
    from aider.config import config_manager
    return config_manager.create_sample_config(path)


if __name__ == "__main__":
    args = parse_args()
    print(f"Parsed args: {args}")