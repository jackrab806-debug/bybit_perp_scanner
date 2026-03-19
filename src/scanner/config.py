"""Configuration management for Bybit Perp Scanner."""

import json
import os
from typing import Dict, Any, List, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional
    pass


# Default configuration values
DEFAULT_CONFIG = {
    "POLL_SECONDS": 60,
    "VOL_EMA_ALPHA": 0.2,
    "VOL_SPIKE_RATIO": 3.0,
    "OI_DELTA_1H_THRESHOLD": 0.10,
    "FUNDING_ABS_THRESHOLD": 0.0005,
    "USE_BASIS_FILTER": False,
    "BASIS_ABS_THRESHOLD": 0.005,
    "MIN_TURNOVER24H_USDT": 20000000,
    "DEBUG": False,
    "USE_WEBHOOK": False,
    "WEBHOOK_URL": "",
    "USE_TELEGRAM": False,
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "SYMBOLS_INCLUDE": [],
    "SYMBOLS_EXCLUDE": [],
    "MAX_SYMBOLS": 300
}


def load_config(config_file: str = "scanner_config.json") -> Dict[str, Any]:
    """
    Load configuration from multiple sources with priority:
    1. Environment variables (highest priority)
    2. JSON config file
    3. Default values (lowest priority)
    
    Args:
        config_file: Path to the JSON configuration file
        
    Returns:
        Dictionary containing merged configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load from JSON file if it exists
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration")
    
    # Override with environment variables
    env_overrides = {
        "POLL_SECONDS": _get_env_int("POLL_SECONDS"),
        "VOL_EMA_ALPHA": _get_env_float("VOL_EMA_ALPHA"),
        "VOL_SPIKE_RATIO": _get_env_float("VOL_SPIKE_RATIO"),
        "OI_DELTA_1H_THRESHOLD": _get_env_float("OI_DELTA_1H_THRESHOLD"),
        "FUNDING_ABS_THRESHOLD": _get_env_float("FUNDING_ABS_THRESHOLD"),
        "USE_BASIS_FILTER": _get_env_bool("USE_BASIS_FILTER"),
        "BASIS_ABS_THRESHOLD": _get_env_float("BASIS_ABS_THRESHOLD"),
        "MIN_TURNOVER24H_USDT": _get_env_int("MIN_TURNOVER24H_USDT"),
    "DEBUG": _get_env_bool("DEBUG"),
        "USE_WEBHOOK": _get_env_bool("USE_WEBHOOK"),
        "WEBHOOK_URL": os.getenv("WEBHOOK_URL"),
        "USE_TELEGRAM": _get_env_bool("USE_TELEGRAM"),
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
        "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
        "SYMBOLS_INCLUDE": _get_env_list("SYMBOLS_INCLUDE"),
        "SYMBOLS_EXCLUDE": _get_env_list("SYMBOLS_EXCLUDE"),
        "MAX_SYMBOLS": _get_env_int("MAX_SYMBOLS"),
    }
    
    # Apply non-None environment overrides
    for key, value in env_overrides.items():
        if value is not None:
            config[key] = value
    
    return config


def _get_env_int(key: str) -> Union[int, None]:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            print(f"Warning: Invalid integer value for {key}: {value}")
    return None


def _get_env_float(key: str) -> Union[float, None]:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Invalid float value for {key}: {value}")
    return None


def _get_env_bool(key: str) -> Union[bool, None]:
    """Get boolean value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        return value.lower() in ('true', '1', 'yes', 'on')
    return None


def _get_env_list(key: str) -> Union[List[str], None]:
    """Get list value from environment variable (comma-separated)."""
    value = os.getenv(key)
    if value is not None:
        return [item.strip() for item in value.split(',') if item.strip()]
    return None


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    
    # Validate numeric ranges
    if config["POLL_SECONDS"] < 1:
        errors.append("POLL_SECONDS must be >= 1")
    
    if not (0 < config["VOL_EMA_ALPHA"] <= 1):
        errors.append("VOL_EMA_ALPHA must be between 0 and 1")
    
    if config["VOL_SPIKE_RATIO"] < 1:
        errors.append("VOL_SPIKE_RATIO must be >= 1")
    
    if config["OI_DELTA_1H_THRESHOLD"] < 0:
        errors.append("OI_DELTA_1H_THRESHOLD must be >= 0")
    
    if config["FUNDING_ABS_THRESHOLD"] < 0:
        errors.append("FUNDING_ABS_THRESHOLD must be >= 0")
    
    if config["BASIS_ABS_THRESHOLD"] < 0:
        errors.append("BASIS_ABS_THRESHOLD must be >= 0")
    
    if config["MIN_TURNOVER24H_USDT"] < 0:
        errors.append("MIN_TURNOVER24H_USDT must be >= 0")
    
    if config["MAX_SYMBOLS"] < 1:
        errors.append("MAX_SYMBOLS must be >= 1")
    
    # Validate Telegram configuration
    if config["USE_TELEGRAM"]:
        if not config["TELEGRAM_BOT_TOKEN"]:
            errors.append("TELEGRAM_BOT_TOKEN is required when USE_TELEGRAM is true")
        if not config["TELEGRAM_CHAT_ID"]:
            errors.append("TELEGRAM_CHAT_ID is required when USE_TELEGRAM is true")
    
    # Validate webhook configuration
    if config["USE_WEBHOOK"]:
        if not config["WEBHOOK_URL"]:
            errors.append("WEBHOOK_URL is required when USE_WEBHOOK is true")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True
