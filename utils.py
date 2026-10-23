"""
utils.py - Configuration loaders, secret protection, and input validators
Part of the AI Coding Assistant academic project.
"""

import os
import re
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

# Maximum input lengths to prevent accidental oversized payloads
MAX_PROMPT_LENGTH = 10000
MAX_CODE_LENGTH = 20000
MAX_ERROR_MSG_LENGTH = 5000
MAX_EXPECTED_BEHAVIOR_LENGTH = 5000

# Supported providers and their configuration mapping
PROVIDER_CONFIG_MAP = {
    "Google Gemini": {
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.5-flash",
        "docs_url": "https://aistudio.google.com/",
        "setup_hint": (
            "Set GEMINI_API_KEY in your local .env file. "
            "Get a key at Google AI Studio (https://aistudio.google.com/)."
        ),
    },
    "OpenAI": {
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "docs_url": "https://platform.openai.com/",
        "setup_hint": (
            "Set OPENAI_API_KEY in your local .env file. "
            "Verify billing and credits at OpenAI (https://platform.openai.com/)."
        ),
    },
    "Anthropic Claude": {
        "key_env": "ANTHROPIC_API_KEY",
        "model_env": "ANTHROPIC_MODEL",
        "default_model": "claude-3-5-haiku-20241022",
        "docs_url": "https://console.anthropic.com/",
        "setup_hint": (
            "Set ANTHROPIC_API_KEY in your local .env file. "
            "Verify credits at Anthropic Console (https://console.anthropic.com/)."
        ),
    },
}


def load_environment() -> None:
    """
    Safely load environment variables from the local .env file.
    Does not raise an error if .env is missing.
    """
    load_dotenv(override=False)


def mask_api_key(api_key: Optional[str]) -> str:
    """
    Return a masked representation of an API key for safe UI status display.
    Never prints or returns the full key.

    Examples:
        None or ""       -> "[Not Configured]"
        "AIzaSy..."      -> "••••••••••••Sy..."
    """
    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        return "[Not Configured]"
    cleaned = api_key.strip()
    if len(cleaned) <= 6:
        return "••••••"
    return f"••••••••••••{cleaned[-4:]}"


def get_provider_status(provider_name: str) -> Dict[str, Optional[str]]:
    """
    Inspect the environment configuration for a specific provider
    without exposing private credentials.
    """
    load_environment()
    config = PROVIDER_CONFIG_MAP.get(provider_name)
    if not config:
        return {
            "provider_name": provider_name,
            "configured": False,
            "masked_key": "[Unsupported Provider]",
            "model": "unknown",
            "setup_hint": "Unknown provider selected.",
        }

    raw_key = os.getenv(config["key_env"], "").strip()
    model = os.getenv(config["model_env"], config["default_model"]).strip()
    is_configured = bool(raw_key)

    return {
        "provider_name": provider_name,
        "configured": is_configured,
        "masked_key": mask_api_key(raw_key),
        "model": model or config["default_model"],
        "setup_hint": config["setup_hint"],
        "docs_url": config["docs_url"],
    }


def validate_generation_input(prompt: str) -> Tuple[bool, Optional[str]]:
    """
    Validate the user's code generation prompt.
    Returns (is_valid, error_message).
    """
    if not prompt or not prompt.strip():
        return False, "Please enter a description of the Python code you would like to generate."

    if len(prompt) > MAX_PROMPT_LENGTH:
        return (
            False,
            f"The prompt is too long ({len(prompt):,} characters). "
            f"Please limit your request to {MAX_PROMPT_LENGTH:,} characters.",
        )

    return True, None


def validate_debugging_input(
    code: str,
    error_message: Optional[str] = None,
    expected_behavior: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate inputs for the debugging workflow.
    - code: required (non-empty, <= MAX_CODE_LENGTH)
    - error_message: optional (if provided, <= MAX_ERROR_MSG_LENGTH)
    - expected_behavior: optional (if provided, <= MAX_EXPECTED_BEHAVIOR_LENGTH)
    Returns (is_valid, error_message).
    """
    if not code or not code.strip():
        return False, "Please paste the Python code that needs debugging."

    if len(code) > MAX_CODE_LENGTH:
        return (
            False,
            f"The code snippet is too long ({len(code):,} characters). "
            f"Please limit code to {MAX_CODE_LENGTH:,} characters.",
        )

    if error_message and len(error_message) > MAX_ERROR_MSG_LENGTH:
        return (
            False,
            f"The error message is too long ({len(error_message):,} characters). "
            f"Please limit to {MAX_ERROR_MSG_LENGTH:,} characters.",
        )

    if expected_behavior and len(expected_behavior) > MAX_EXPECTED_BEHAVIOR_LENGTH:
        return (
            False,
            f"The expected behavior description is too long ({len(expected_behavior):,} characters). "
            f"Please limit to {MAX_EXPECTED_BEHAVIOR_LENGTH:,} characters.",
        )

    return True, None


def sanitize_error_message(
    error: Exception,
    provider_name: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Clean an exception string to prevent leaking private credentials,
    file system paths, or raw authorization headers to the user interface.
    Redacts both known pattern signatures and actual configured keys
    (Gemini, OpenAI, Anthropic, or explicitly passed keys) if present.
    """
    sanitized = str(error)

    # 1. Redact actual configured API keys from environment and direct parameters
    keys_to_redact = set()
    if api_key and isinstance(api_key, str) and len(api_key.strip()) >= 6:
        keys_to_redact.add(api_key.strip())

    for env_var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.getenv(env_var, "").strip()
        if val and len(val) >= 6:
            keys_to_redact.add(val)

    for secret in keys_to_redact:
        sanitized = sanitized.replace(secret, "[REDACTED_API_KEY]")

    # 2. Remove potential key patterns (e.g. AIza..., sk-..., ant-..., key=...)
    sanitized = re.sub(r"(AIza[0-9A-Za-z-_]{20,})", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"(sk-[0-9A-Za-z-_]{20,})", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"(ant-[0-9A-Za-z-_]{20,})", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"(key|token|auth)=['\"][^'\"]+['\"]", r"\1=[REDACTED]", sanitized, flags=re.IGNORECASE)

    # Check for known HTTP status code signatures
    if "401" in sanitized or "unauthorized" in sanitized.lower() or "authentication" in sanitized.lower():
        return (
            f"Authentication failed for {provider_name}. "
            "Please check that your API key is correctly configured in your local .env file."
        )
    if "429" in sanitized or "quota" in sanitized.lower() or "rate limit" in sanitized.lower():
        return (
            f"Rate limit or quota reached for {provider_name}. "
            "Please wait a moment before trying again or verify your account quota."
        )
    if "404" in sanitized or "model" in sanitized.lower() and "not found" in sanitized.lower():
        return (
            f"The selected model was not found on {provider_name}. "
            "Please verify the model name configured in your local .env file."
        )
    if "timeout" in sanitized.lower() or "connection" in sanitized.lower():
        return (
            f"Network connection to {provider_name} timed out. "
            "Please check your internet connection and try again."
        )
    if "503" in sanitized or "high demand" in sanitized.lower() or "service unavailable" in sanitized.lower():
        return (
            f"{provider_name} is currently experiencing temporary high server demand (HTTP 503). "
            "The assistant automatically tried multiple times, but the model server is still busy. "
            "Please wait a few seconds and try your request again."
        )

    # General fallback
    return f"{provider_name} request failed: {sanitized}"


def extract_python_code(raw_text: str) -> str:
    """
    Safely extract Python code from markdown code fences (```python ... ```)
    if present. If no valid code block is found, returns an empty string
    to prevent conversational markdown from being treated as executable code.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    match = re.search(r"```python\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback to general code block
    match_general = re.search(r"```\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match_general:
        return match_general.group(1).strip()

    return ""
