"""
providers/base.py - Provider-independent LLM Interface and Response Dataclass
Part of the AI Coding Assistant academic project.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ProviderResponse:
    """
    Standardized response object returned by all AI providers.
    Ensures a consistent contract across Google Gemini, OpenAI, and Anthropic.
    """
    success: bool
    explanation: str = ""
    code: str = ""
    key_lines_explanation: Optional[str] = None
    change_summary: Optional[str] = None
    verification_steps: Optional[str] = None
    raw_content: Optional[str] = None
    error_message: Optional[str] = None
    provider_name: str = ""
    model_name: str = ""


class BaseLLMProvider(ABC):
    """
    Abstract Base Class defining the contract that every LLM provider adapter
    must implement.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the human-readable display name of the provider."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns the configured model name for this provider."""
        pass

    @abstractmethod
    def validate_credentials(self) -> Tuple[bool, Optional[str]]:
        """
        Verifies if required API credentials are configured in the environment.
        Returns (is_configured, status_or_setup_message).
        """
        pass

    @abstractmethod
    def generate_code(self, prompt: str, context: Optional[str] = None) -> ProviderResponse:
        """
        Generate Python code from a natural-language description.
        Must return a standardized ProviderResponse.
        """
        pass

    @abstractmethod
    def debug_code(
        self,
        code: str,
        error_message: Optional[str] = None,
        expected_behavior: Optional[str] = None,
    ) -> ProviderResponse:
        """
        Debug Python code given source code and an error message / observed behavior.
        Must return a standardized ProviderResponse.
        """
        pass
