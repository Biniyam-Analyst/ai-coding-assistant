"""
ai_service.py - Central AI Service Layer & Provider Factory
Part of the AI Coding Assistant academic project.
"""

from typing import Dict, List, Optional
import utils
from providers.base import BaseLLMProvider, ProviderResponse
from providers.gemini_provider import GeminiProvider


class _DeferredProvider(BaseLLMProvider):
    """
    Temporary placeholder adapter for providers scheduled in subsequent phases (Phase 6).
    Ensures safe handling if selected in the UI without crashing.
    """

    def __init__(self, name: str, phase_note: str):
        self._name = name
        self._phase_note = phase_note

    @property
    def provider_name(self) -> str:
        return self._name

    def get_model_name(self) -> str:
        return "provisional-model"

    def validate_credentials(self):
        return False, f"The {self._name} adapter is scheduled for {self._phase_note}."

    def generate_code(self, prompt: str, context: Optional[str] = None) -> ProviderResponse:
        return ProviderResponse(
            success=False,
            error_message=(
                f"The {self._name} provider adapter is scheduled for implementation in {self._phase_note}. "
                "Please select 'Google Gemini' to generate code with the active SDK."
            ),
            provider_name=self._name,
            model_name=self.get_model_name(),
        )

    def debug_code(
        self,
        code: str,
        error_message: Optional[str] = None,
        expected_behavior: Optional[str] = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            success=False,
            error_message=(
                f"The {self._name} provider adapter is scheduled for implementation in {self._phase_note}. "
                "Please select 'Google Gemini' to debug code with the active SDK."
            ),
            provider_name=self._name,
            model_name=self.get_model_name(),
        )


# Registry of supported providers
SUPPORTED_PROVIDERS: List[str] = [
    "Google Gemini",
    "OpenAI",
    "Anthropic Claude",
]


def get_available_providers() -> List[str]:
    """Return the list of available providers for UI selection."""
    return list(SUPPORTED_PROVIDERS)


def get_provider(provider_name: str = "Google Gemini") -> BaseLLMProvider:
    """
    Factory function to retrieve the appropriate provider adapter.
    """
    clean_name = provider_name.strip()

    if clean_name == "Google Gemini":
        return GeminiProvider()

    if clean_name == "OpenAI":
        # Will be replaced with concrete OpenAIProvider in Phase 6
        return _DeferredProvider("OpenAI", "Phase 6")

    if clean_name == "Anthropic Claude":
        # Will be replaced with concrete AnthropicProvider in Phase 6
        return _DeferredProvider("Anthropic Claude", "Phase 6")

    raise ValueError(f"Unsupported provider requested: {provider_name}")


def generate_code(
    prompt: str,
    context: Optional[str] = None,
    provider_name: str = "Google Gemini",
) -> ProviderResponse:
    """
    Validate inputs and request Python code generation from the selected AI provider.
    Never executes the generated code.
    """
    # 1. Validate prompt input
    is_valid, validation_msg = utils.validate_generation_input(prompt)
    if not is_valid:
        return ProviderResponse(
            success=False,
            error_message=validation_msg,
            provider_name=provider_name,
        )

    # 2. Resolve provider adapter
    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        return ProviderResponse(
            success=False,
            error_message=str(e),
            provider_name=provider_name,
        )

    # 3. Delegate to provider adapter
    return provider.generate_code(prompt=prompt, context=context)


def debug_code(
    code: str,
    error_message: Optional[str] = None,
    expected_behavior: Optional[str] = None,
    provider_name: str = "Google Gemini",
) -> ProviderResponse:
    """
    Validate inputs and request code debugging assistance from the selected AI provider.
    Never executes user-submitted or corrected code.
    """
    # 1. Validate debugging input
    is_valid, validation_msg = utils.validate_debugging_input(
        code=code,
        error_message=error_message,
        expected_behavior=expected_behavior,
    )
    if not is_valid:
        return ProviderResponse(
            success=False,
            error_message=validation_msg,
            provider_name=provider_name,
        )

    # 2. Resolve provider adapter
    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        return ProviderResponse(
            success=False,
            error_message=str(e),
            provider_name=provider_name,
        )

    # 3. Delegate to provider adapter
    return provider.debug_code(
        code=code,
        error_message=error_message,
        expected_behavior=expected_behavior,
    )
