import os
from dotenv import load_dotenv
import yaml
from typing import Dict, Any, Optional

from .llm_config import LLMConfig
from .optimizer_config import OptimizerConfig

# Load environment variables
load_dotenv()

# Default config file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
CLASS_CONFIG_MAP = {
    "llm": LLMConfig,
    "optimizer": OptimizerConfig
}

def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def create_config(config_dict: Dict[str, Any], config_name: str) -> Any:
    """Create a config object with values from config dict and environment variables."""
    config_values = config_dict.get(config_name, {})
    config_class = CLASS_CONFIG_MAP[config_name]
    
    kwargs = {}
    
    # Get default values from an empty instance
    defaults = config_class().__dict__
    
    for key in defaults:
        if key in config_values:
            kwargs[key] = config_values[key]
    
    # Special handling for API key from environment variables
    if config_name == "llm":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        else:
            print("Warning: OPENAI_API_KEY not found in environment variables")
    
    # Create and return a config instance
    return config_class(**kwargs)

# Load YAML configuration once
yaml_config = load_yaml_config()

# Create config objects
LLM_CONFIG = create_config(yaml_config, "llm")
OPTIMIZER_CONFIG = create_config(yaml_config, "optimizer")

def reload_config(config_path: Optional[str] = None) -> None:
    """Reload configuration from a specified file."""
    global LLM_CONFIG, OPTIMIZER_CONFIG, yaml_config
    yaml_config = load_yaml_config(config_path)
    LLM_CONFIG = create_config(yaml_config, "llm") 
    OPTIMIZER_CONFIG = create_config(yaml_config, "optimizer")

__all__ = [
    "LLM_CONFIG",
    "OPTIMIZER_CONFIG",
    "reload_config"
]

if __name__ == "__main__":
    print(LLM_CONFIG.__dict__)
    print(OPTIMIZER_CONFIG)