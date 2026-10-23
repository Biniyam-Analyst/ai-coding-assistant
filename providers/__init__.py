"""
Providers package for AI Coding Assistant.
Supports modular, provider-independent LLM adapters.
"""

from .base import BaseLLMProvider, ProviderResponse

__all__ = ["BaseLLMProvider", "ProviderResponse"]
