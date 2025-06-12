"""Utilities for discovering local LLM servers and updating configuration."""

import json
import re
import socket
import subprocess
from typing import Any, Dict, List, Optional

import requests

from .dataclasses import AiderConfig, ProviderConfig


def check_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is open and responding."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.error, ConnectionRefusedError, OSError):
        return False


def check_openai_compatible_api(base_url: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Check if an endpoint is OpenAI-compatible and return available models.

    Args:
        base_url: Base URL of the API (e.g., "http://localhost:1234")
        timeout: Request timeout in seconds

    Returns:
        Dict with 'models' list and 'info' dict if compatible, None otherwise
    """
    try:
        # Ensure base_url has proper format
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"

        if not base_url.endswith("/v1"):
            if base_url.endswith("/"):
                base_url = base_url + "v1"
            else:
                base_url = base_url + "/v1"

        # Try to get models list
        models_url = f"{base_url}/models"
        response = requests.get(models_url, timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                models = []
                for model in data["data"]:
                    if isinstance(model, dict) and "id" in model:
                        models.append(model["id"])

                # Try to get additional info about the server
                info = {}
                try:
                    # Some servers provide info at /v1/ or have custom endpoints
                    info_response = requests.get(base_url, timeout=2.0)
                    if info_response.status_code == 200:
                        try:
                            info_data = info_response.json()
                            if isinstance(info_data, dict):
                                info = info_data
                        except json.JSONDecodeError:
                            pass
                except requests.RequestException:
                    pass

                return {"models": models, "info": info, "base_url": base_url}

    except requests.RequestException:
        pass

    return None


def scan_running_llm_processes() -> List[Dict[str, Any]]:
    """
    Scan for running llama.cpp and vLLM processes to detect custom ports.

    Returns:
        List of discovered processes with their port information
    """

    discovered_processes = []

    try:
        # Get all running processes
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return discovered_processes

        lines = result.stdout.split("\n")

        for line in lines:
            if not line.strip():
                continue

            # Look for llama.cpp processes
            if (
                "llama.cpp" in line.lower()
                or "llama-cpp" in line.lower()
                or "/llama" in line
                or "llama-server" in line.lower()
                or "llamacpp" in line.lower()
            ):
                # Extract port from --port argument
                port_match = re.search(r"--port\s+(\d+)", line)
                if port_match:
                    port = int(port_match.group(1))
                    discovered_processes.append(
                        {
                            "name": f"llamacpp-{port}",
                            "port": port,
                            "description": "llama.cpp server (process)",
                            "process_type": "llamacpp",
                        }
                    )

            # Look for vLLM processes
            elif "vllm" in line.lower() or "python -m vllm" in line:
                # Extract port from --port argument
                port_match = re.search(r"--port\s+(\d+)", line)
                if port_match:
                    port = int(port_match.group(1))
                    discovered_processes.append(
                        {
                            "name": f"vllm-{port}",
                            "port": port,
                            "description": "vLLM server (process)",
                            "process_type": "vllm",
                        }
                    )
                else:
                    # vLLM default port is 8000 if no --port specified
                    discovered_processes.append(
                        {
                            "name": "vllm-8000",
                            "port": 8000,
                            "description": "vLLM server (process, default port)",
                            "process_type": "vllm",
                        }
                    )

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        # ps command failed or timed out
        pass

    return discovered_processes


def discover_local_llm_servers(interactive: bool = False, io=None) -> Dict[str, Dict[str, Any]]:
    """
    Discover local LLM servers on common ports and running processes.

    Args:
        interactive: Whether to prompt user for confirmation on discovered processes
        io: InputOutput object for user interaction (required if interactive=True)

    Returns:
        Dict mapping provider names to their configurations
    """
    discovered = {}

    # Define common local LLM server configurations
    candidates = [
        {
            "name": "lm-studio",
            "host": "localhost",
            "port": 1234,
            "description": "LM Studio",
        },
        {
            "name": "vllm",
            "host": "localhost",
            "port": 8000,
            "description": "vLLM or llama.cpp server",
        },
        {
            "name": "llamacpp",
            "host": "localhost",
            "port": 8080,
            "description": "llama.cpp server",
        },
    ]

    # Check common ports first
    for candidate in candidates:
        host = candidate["host"]
        port = candidate["port"]
        name = candidate["name"]

        # Check if port is open
        if check_port_open(host, port):
            base_url = f"http://{host}:{port}"

            # Check if it's OpenAI-compatible
            api_info = check_openai_compatible_api(base_url)
            if api_info and api_info["models"]:
                provider_type = "openai"  # All discovered servers use OpenAI-compatible API

                discovered[name] = {
                    "type": provider_type,
                    "base_url": base_url,
                    "models": api_info["models"],
                    "description": candidate["description"],
                    "auto_discovered": True,
                }

    # Scan for running processes with custom ports
    running_processes = scan_running_llm_processes()
    for process in running_processes:
        port = process["port"]
        name = process["name"]

        # Skip if we already found this port in common ports scan
        base_url = f"http://localhost:{port}"
        if any(existing["base_url"] == base_url for existing in discovered.values()):
            continue

        # Check if the process port is actually open and responding
        if check_port_open("localhost", port):
            # Check if it's OpenAI-compatible
            api_info = check_openai_compatible_api(base_url)
            if api_info and api_info["models"]:
                # Ask user if interactive mode is enabled
                should_add = True
                if interactive and io:
                    models_count = len(api_info["models"])
                    should_add = io.confirm_ask(
                        (
                            f"Found {process['description']} running on port {port} with"
                            f" {models_count} models. Add to configuration?"
                        ),
                        default="y",
                    )

                if should_add:
                    discovered[name] = {
                        "type": "openai",
                        "base_url": base_url,
                        "models": api_info["models"],
                        "description": process["description"],
                        "auto_discovered": True,
                        "process_discovered": True,
                    }
                elif interactive and io:
                    io.tool_output(f"Skipping {process['description']} on port {port}")

    return discovered


def update_config_with_discoveries(
    config: AiderConfig, discovered: Dict[str, Dict[str, Any]], verbose: bool = False
) -> bool:
    """
    Update configuration with discovered local servers.

    Args:
        config: Configuration to update
        discovered: Discovered servers from discover_local_llm_servers()
        verbose: Whether to print verbose output

    Returns:
        True if any changes were made, False otherwise
    """
    if not discovered:
        return False

    changes_made = False

    # Initialize providers dict if it doesn't exist
    if not config.providers:
        config.providers = {}

    for provider_name, provider_info in discovered.items():
        existing = config.providers.get(provider_name)

        if not existing:
            # Add new provider
            config.providers[provider_name] = ProviderConfig(
                type=provider_info["type"],
                base_url=provider_info["base_url"],
                models=provider_info["models"],
            )
            changes_made = True
            if verbose:
                print(
                    f"Added {provider_info['description']} at {provider_info['base_url']} with"
                    f" {len(provider_info['models'])} models"
                )

        elif existing.base_url == provider_info["base_url"]:
            # Update models list for existing provider
            if set(existing.models or []) != set(provider_info["models"]):
                existing.models = provider_info["models"]
                changes_made = True
                if verbose:
                    print(
                        f"Updated models for {provider_name}: {len(provider_info['models'])} models"
                        " found"
                    )

        # If base_url differs, don't automatically override (user may have customized)

    return changes_made
