# Aider Configuration System Migration Guide

## Overview

Aider now supports a unified configuration system through a single `.aider.yml` file, which replaces the complex multi-source configuration approach (CLI args, environment variables, and multiple config files). This new system adds support for **multiple endpoints per provider type** while maintaining backwards compatibility.

## Key Benefits

1. **Single Configuration Source**: One file to configure everything
2. **Multiple Endpoints**: Support multiple OpenAI, Ollama, etc. endpoints
3. **Clear Model Naming**: Explicit endpoint-aware model specifications
4. **Backwards Compatibility**: Existing configurations continue to work
5. **Better Validation**: Configuration errors caught early with clear messages

## Configuration File Format

### Basic Structure

```yaml
# .aider.yml - Unified configuration file

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

# Other settings
git:
  auto_commits: true
  commit_prefix: "aider: "

output:
  user_input_color: "#00cc00"
  pretty: true
  stream: true

edit_format: "diff"
vim: false
```

## Enhanced Model Naming

### New Format: `endpoint_name/model_name`

- `openai_main/gpt-4` - GPT-4 on main OpenAI endpoint
- `openai_azure/gpt-4-azure` - GPT-4 on Azure endpoint
- `local_ollama/llama3:8b` - Llama3 8B on local Ollama
- `remote_ollama/llama3:70b` - Llama3 70B on remote Ollama server

### Backwards Compatibility

- `openai/gpt-4` → automatically maps to first OpenAI provider
- `ollama/llama3` → automatically maps to first Ollama provider
- Pure model names → use default provider for that model type

## Usage Examples

### Multiple OpenAI Endpoints

```yaml
providers:
  openai_main:
    type: openai
    api_key: "sk-main-key"
    base_url: "https://api.openai.com/v1"
    
  openai_azure:
    type: openai
    api_key: "azure-key" 
    base_url: "https://company.openai.azure.com"
    
  openai_custom:
    type: openai
    api_key: "custom-key"
    base_url: "https://custom-proxy.com/v1"

# Use specific endpoints
model: "openai_main/gpt-4"           # Main OpenAI
# model: "openai_azure/gpt-4"        # Azure endpoint
# model: "openai_custom/gpt-4"       # Custom endpoint
```

### Multiple Ollama Instances

```yaml
providers:
  local_ollama:
    type: ollama
    base_url: "http://localhost:11434"
    models: ["llama3:8b", "codellama:13b"]
    
  gpu_server:
    type: ollama
    base_url: "http://gpu-server:11434"
    models: ["llama3:70b", "codellama:34b"]
    
  cloud_ollama:
    type: ollama
    base_url: "https://ollama.mycloud.com"
    api_key: "optional-auth-token"
    models: ["llama3:405b"]

# Easy switching between local and remote
model: "local_ollama/llama3:8b"      # For development
# model: "gpu_server/llama3:70b"     # For heavy tasks
# model: "cloud_ollama/llama3:405b"  # For production
```

## Migration Path

### 1. Current System (Still Supported)

```bash
# Environment variables
export OPENAI_API_KEY="sk-..."
export OLLAMA_HOST="http://localhost:11434"

# Command line
aider --model gpt-4 --weak-model gpt-3.5-turbo
```

### 2. New Unified System

```yaml
# .aider.yml
providers:
  openai_default:
    type: openai
    api_key: "sk-..."
  ollama_default:
    type: ollama
    base_url: "http://localhost:11434"

model: "openai_default/gpt-4"
weak_model: "openai_default/gpt-3.5-turbo"
```

```bash
# Simplified command line
aider  # Uses configuration file
```

## Commands

### Initialize Configuration

```bash
aider --init-config                 # Create sample .aider.yml
aider --init-config --path mydir/   # Create in specific location
```

### Validate Configuration

```bash
aider --validate-config             # Check configuration file
aider --show-config                 # Show current settings
aider --list-models                 # Show available models
```

### Migration Assistance

```bash
aider --migrate-config              # Convert old settings (planned)
aider --help-config                 # Show configuration help
```

## Advanced Features

### Wildcard Model Settings

```yaml
model_settings:
  "openai_main/*":              # All models on openai_main
    temperature: 0.1
  "local_ollama/*":             # All local Ollama models
    temperature: 0.0
    stream: true
  "*/gpt-4":                    # GPT-4 on any endpoint
    max_tokens: 4096
```

### Environment-Specific Configurations

```yaml
# Development
providers:
  dev_ollama:
    type: ollama
    base_url: "http://localhost:11434"
  dev_openai:
    type: openai
    api_key: "sk-dev-key"

model: "dev_ollama/llama3:8b"       # Fast local model

# Production (separate .aider.yml)
providers:
  prod_openai:
    type: openai
    api_key: "sk-prod-key"
  prod_anthropic:
    type: anthropic
    api_key: "sk-ant-prod-key"

model: "prod_anthropic/claude-3-5-sonnet-20241022"  # Production model
```

## Configuration File Locations

Aider searches for configuration files in this order:

1. Path specified with `--config`
2. `.aider.yml` in current directory
3. `.aider.yml` in git repository root
4. `.aider.yml` in home directory

## Breaking Changes

The new system is designed to be backwards compatible, but some advanced use cases may require updates:

1. **Multiple API keys for same provider**: Now explicitly supported through multiple endpoints
2. **Provider-specific base URLs**: Now configured per endpoint rather than globally
3. **Model-specific settings**: Now use endpoint-aware naming

## Implementation Files

The new configuration system consists of:

- `aider/config.py` - Core configuration classes and management
- `aider/models.py` - Enhanced with `EndpointAwareModelManager`
- `aider/config_bridge.py` - Backwards compatibility layer
- `aider/args_new.py` - Simplified argument parser
- `aider/args_simple.py` - Alternative minimal parser

## Getting Started

1. **Try the new system**:
   ```bash
   aider --init-config
   # Edit .aider.yml with your API keys and preferences
   aider
   ```

2. **Validate your configuration**:
   ```bash
   aider --validate-config
   aider --list-models
   ```

3. **Use multiple endpoints**:
   ```yaml
   # Add multiple providers in .aider.yml
   providers:
     openai_main: { type: openai, api_key: "sk-main..." }
     openai_backup: { type: openai, api_key: "sk-backup..." }
   model: "openai_main/gpt-4"
   ```

The new configuration system provides a much cleaner, more powerful way to manage aider's settings while maintaining full backwards compatibility with existing workflows.