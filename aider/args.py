#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import shtab

from aider import __version__


def get_parser(default_config_files=None, git_root=None):
    """
    Simplified argument parser for aider's unified configuration system.
    
    Most configuration options are now handled through .aider.yml files.
    This parser only handles essential operational arguments and backwards
    compatibility for the most common CLI options.
    """
    
    parser = argparse.ArgumentParser(
        description="aider is AI pair programming in your terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Most configuration is now handled through .aider.yml files.
Use 'aider --init-config' to create a sample configuration file.

For help and documentation: https://aider.chat/docs/
"""
    )

    # Files to add to the chat (positional arguments)
    parser.add_argument(
        "files", 
        metavar="FILE", 
        nargs="*", 
        help="files to edit with an LLM (optional)"
    ).complete = shtab.FILE

    # Essential operational arguments
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version number and exit",
    )

    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create a sample .aider.yml configuration file and exit"
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="CONFIG_FILE",
        help="Specify the config file (default: search for .aider.yml)"
    ).complete = shtab.FILE

    # Most common model arguments for backwards compatibility
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Specify the model to use for the main chat (overrides config file)"
    )

    parser.add_argument(
        "--weak-model",
        metavar="WEAK_MODEL", 
        help="Specify the model to use for commit messages and chat history summarization"
    )

    parser.add_argument(
        "--edit-format",
        metavar="EDIT_FORMAT",
        choices=["diff", "whole", "udiff", "editor-diff", "editor-whole", "architect"],
        help="Specify what edit format the LLM should use"
    )

    # Common API key arguments for backwards compatibility
    parser.add_argument(
        "--openai-api-key",
        help="Specify the OpenAI API key (consider using config file instead)"
    )

    parser.add_argument(
        "--anthropic-api-key",
        help="Specify the Anthropic API key (consider using config file instead)"
    )

    parser.add_argument(
        "--openai-api-base",
        help="Specify the OpenAI API base URL (consider using config file instead)"
    )

    # Essential operational modes
    parser.add_argument(
        "--message",
        "--msg",
        "-m",
        metavar="MESSAGE",
        help="Specify a single message to send the LLM, process reply then exit"
    )

    parser.add_argument(
        "--message-file",
        "-f",
        metavar="MESSAGE_FILE",
        help="Specify a file containing the message to send the LLM"
    ).complete = shtab.FILE

    parser.add_argument(
        "--gui",
        "--browser",
        action="store_true",
        help="Run aider in your browser",
        default=False
    )

    parser.add_argument(
        "--yes-always",
        action="store_true",
        help="Always say yes to every confirmation",
        default=False
    )

    # Git operations
    parser.add_argument(
        "--auto-commits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable auto commit of LLM changes"
    )

    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit all pending changes with a suitable commit message, then exit",
        default=False
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without modifying files",
        default=False
    )

    # Output and debugging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
        default=False
    )

    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Disable pretty, colorized output",
        default=False
    )

    parser.add_argument(
        "--show-repo-map",
        action="store_true",
        help="Print the repo map and exit (debug)",
        default=False
    )

    # File operations
    parser.add_argument(
        "--file",
        action="append",
        metavar="FILE",
        help="specify a file to edit (can be used multiple times)"
    ).complete = shtab.FILE

    parser.add_argument(
        "--read",
        action="append", 
        metavar="FILE",
        help="specify a read-only file (can be used multiple times)"
    ).complete = shtab.FILE

    # Essential settings
    parser.add_argument(
        "--vim",
        action="store_true",
        help="Use VI editing mode in the terminal",
        default=False
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Specify the encoding for input and output"
    )

    # Configuration validation and help
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

    # Shell completions
    supported_shells_list = sorted(list(shtab.SUPPORTED_SHELLS))
    parser.add_argument(
        "--shell-completions",
        metavar="SHELL",
        choices=supported_shells_list,
        help=f"Print shell completion script for the specified shell and exit"
    )

    # Legacy attributes for backwards compatibility with main.py
    # These are set to defaults since they're not configurable via CLI anymore
    parser.set_defaults(
        env_file=".env",
        git=True,
        verify_ssl=True,
        timeout=None,
        analytics_disable=False,
        dark_mode=False,
        light_mode=False,
        user_input_color="#00cc00",
        tool_error_color="#FF2222", 
        tool_warning_color="#FFA500",
        assistant_output_color="#0088ff",
        code_theme="default",
        pretty=True
    )

    return parser


def main():
    """Test the simplified argument parser."""
    parser = get_parser()
    args = parser.parse_args()
    print(f"Parsed args: {args}")


if __name__ == "__main__":
    main()