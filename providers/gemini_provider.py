"""
providers/gemini_provider.py - Google Gemini adapter using the official google-genai SDK
Part of the AI Coding Assistant academic project.
"""

import os
import re
import time
from typing import Optional, Tuple
from google import genai
from google.genai import errors as genai_errors

from .base import BaseLLMProvider, ProviderResponse
import prompts
import utils


class GeminiProvider(BaseLLMProvider):
    """
    Adapter for Google Gemini models using the synchronous official google-genai SDK.
    Reads credentials safely from local environment variables.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        utils.load_environment()
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        # Default model is provisional and verified against Google AI Studio SDK docs
        self._model_name = (
            model_name
            or os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
            or "gemini-2.5-flash"
        )

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    def get_model_name(self) -> str:
        return self._model_name

    def validate_credentials(self) -> Tuple[bool, Optional[str]]:
        """
        Verify if GEMINI_API_KEY is present without revealing the key.
        """
        utils.load_environment()
        key = self._api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            return (
                False,
                "GEMINI_API_KEY is not configured in your local .env file.\n"
                "To use Google Gemini, please copy .env.example to .env and set your GEMINI_API_KEY.\n"
                "You can generate a key at: https://aistudio.google.com/",
            )
        return True, None

    def _get_client(self) -> genai.Client:
        """Initialize the official Google Gen AI Client."""
        key = self._api_key or os.getenv("GEMINI_API_KEY", "").strip()
        return genai.Client(api_key=key)

    def _parse_sections(self, raw_text: str) -> dict:
        """
        Extract code and sections from the structured tutor output.
        Safeguards:
        - Extracts code specifically from Section 2 to avoid accidentally picking up
          buggy snippets quoted in Section 1.
        - Captures Section 3 'Explanation of Key Lines' for code generation or
          'Summary of Changes' for debugging.
        - Prevents conversational Markdown from being treated as Python code if no
          code fences exist.
        """
        sections = {
            "explanation": "",
            "code": "",
            "key_lines_explanation": None,
            "change_summary": None,
            "verification_steps": None,
        }

        # 1. Root Cause or Approach section
        approach_match = re.search(
            r"#{1,4}\s*(?:1\.?\s*)?(?:Approach|Conceptual Overview|Root Cause Analysis).*?\n(.*?)(?=\n#{1,4}\s*(?:2\.|\b(?:Corrected|Python Code))|\Z)",
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )
        if approach_match:
            sections["explanation"] = approach_match.group(1).strip()
        else:
            # Fallback: if no explicit section 1, strip code blocks from raw text
            cleaned = re.sub(r"```(?:python)?.*?```", "", raw_text, flags=re.DOTALL).strip()
            sections["explanation"] = cleaned or raw_text

        # 2. Extract Code: Prefer Section 2 (Python Code / Corrected Python Code)
        section_2_match = re.search(
            r"#{1,4}\s*(?:2\.?\s*)?(?:Corrected\s+Python\s+Code|Corrected\s+Code|Python\s+Code).*?\n(.*?)(?=\n#{1,4}\s*(?:[34]\.|\b(?:Summary|Explanation of Key|Key Lines|How to Test|Verification))|\Z)",
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )
        if section_2_match:
            sec2_code = utils.extract_python_code(section_2_match.group(1))
            if sec2_code:
                sections["code"] = sec2_code
            else:
                sections["code"] = ""
        else:
            # Fallback to general code extraction only if Section 2 header was completely absent
            sections["code"] = utils.extract_python_code(raw_text)

        # 3a. Explanation of Key Lines (for Code Generation)
        key_lines_match = re.search(
            r"#{1,4}\s*(?:3\.?\s*)?(?:Explanation of Key Lines|Key Lines Explanation|Key Lines|Explanation of Code).*?\n(.*?)(?=\n#{1,4}\s*(?:4\.|\b(?:How to Test|Verification|Edge Cases))|\Z)",
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )
        if key_lines_match:
            sections["key_lines_explanation"] = key_lines_match.group(1).strip()

        # 3b. Summary of Changes (for Code Debugging)
        changes_match = re.search(
            r"#{1,4}\s*(?:3\.?\s*)?(?:Summary of Changes|Changes Made).*?\n(.*?)(?=\n#{1,4}\s*(?:4\.|\b(?:How to Test|Verification|Edge Cases))|\Z)",
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )
        if changes_match:
            sections["change_summary"] = changes_match.group(1).strip()

        # 4. Verification / How to Test section
        verify_match = re.search(
            r"#{1,4}\s*(?:4\.?\s*)?(?:How to Test|Verification Steps|Edge Cases).*?\n(.*?)(?=\n#{1,4}\s*(?:5\.|\b[A-Z])|\Z)",
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )
        if verify_match:
            sections["verification_steps"] = verify_match.group(1).strip()

        return sections

    def _is_transient_overload_error(self, error: Exception) -> bool:
        """
        Check if an error is a confirmed transient HTTP 503 or temporary high-demand error.
        Retries genuine temporary service-unavailable conditions while failing fast on
        client errors (400, 401, 403, 404, 429) or non-transient exceptions.
        """
        code = getattr(error, "code", None)
        status = getattr(error, "status", None)
        err_str = str(error)
        err_msg = err_str.lower()

        # Non-transient client status codes or client errors fail fast without retry
        if code in (400, 401, 403, 404, 429):
            return False
        if re.search(r"\b(400|401|403|404|429)\b", err_str):
            if any(term in err_msg for term in (
                "unauthorized",
                "invalid api key",
                "quota",
                "resource has been exhausted",
                "not found",
                "forbidden",
                "permission denied",
                "bad request",
            )):
                return False

        # Confirmed 503 HTTP status code or UNAVAILABLE gRPC/API status
        if code == 503 or status == "UNAVAILABLE":
            return True

        # Check for explicit HTTP 503 in error representation
        if re.search(r"\b503\b", err_str):
            return True

        # Check for confirmed transient demand/capacity indicators from Google API
        transient_indicators = (
            "high demand",
            "spikes in demand",
            "service unavailable",
            "temporarily unavailable",
            "temporarily overloaded",
            "model is overloaded",
            "server is overloaded",
        )
        return any(indicator in err_msg for indicator in transient_indicators)

    def _generate_with_retry(self, client: genai.Client, contents: str):
        """
        Execute client.models.generate_content with limited retries.
        Safeguards:
        - Retries only confirmed transient HTTP 503 / high demand errors.
        - Maximum of 2 retries (3 total attempts).
        - Delays: 2.0 seconds after attempt 1, 4.0 seconds after attempt 2.
        - Fails fast on all other errors (e.g., 401, 403, 404, 429).
        """
        delays = [2.0, 4.0]
        max_retries = len(delays)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return client.models.generate_content(
                    model=self.get_model_name(),
                    contents=contents,
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries and self._is_transient_overload_error(e):
                    time.sleep(delays[attempt])
                    continue
                # If not transient or retries exhausted, raise immediately
                raise last_error

    def generate_code(self, prompt: str, context: Optional[str] = None) -> ProviderResponse:
        """
        Generate Python code with Google Gemini.
        """
        is_valid, validation_msg = self.validate_credentials()
        if not is_valid:
            return ProviderResponse(
                success=False,
                error_message=validation_msg,
                provider_name=self.provider_name,
                model_name=self.get_model_name(),
            )

        full_prompt = prompts.build_generation_prompt(prompt, context)

        try:
            client = self._get_client()
            response = self._generate_with_retry(client, full_prompt)

            if not response or not getattr(response, "text", None):
                return ProviderResponse(
                    success=False,
                    error_message=(
                        "Google Gemini returned an empty response. "
                        "The model may have filtered the response or returned no tokens."
                    ),
                    provider_name=self.provider_name,
                    model_name=self.get_model_name(),
                )

            raw_text = response.text
            parsed = self._parse_sections(raw_text)

            return ProviderResponse(
                success=True,
                explanation=parsed["explanation"],
                code=parsed["code"],
                key_lines_explanation=parsed["key_lines_explanation"],
                change_summary=parsed["change_summary"],
                verification_steps=parsed["verification_steps"],
                raw_content=raw_text,
                provider_name=self.provider_name,
                model_name=self.get_model_name(),
            )

        except genai_errors.APIError as e:
            sanitized = utils.sanitize_error_message(e, self.provider_name, api_key=self._api_key)
            return ProviderResponse(
                success=False,
                error_message=sanitized,
                provider_name=self.provider_name,
                model_name=self.get_model_name(),
            )
        except Exception as e:
            sanitized = utils.sanitize_error_message(e, self.provider_name, api_key=self._api_key)
            return ProviderResponse(
                success=False,
                error_message=sanitized,
                provider_name=self.provider_name,
                model_name=self.get_model_name(),
            )

    def debug_code(
        self,
        code: str,
        error_message: Optional[str] = None,
        expected_behavior: Optional[str] = None,
    ) -> ProviderResponse:
        """
        Debug Python code with Google Gemini.
        """
        is_valid, validation_msg = self.validate_credentials()
        if not is_valid:
            return ProviderResponse(
                success=False,
                error_message=validation_msg,
                provider_name=self.provider_name,
                model_name=self.get_model_name(),
            )

        full_prompt = prompts.build_debugging_prompt(
            code=code,
            error_message=error_message,
            expected_behavior=expected_behavior,
        )

        try:
            client = self._get_client()
            response = self._generate_with_retry(client, full_prompt)

            if not response or not getattr(response, "text", None):
                return ProviderResponse(
                    success=False,
                    error_message=(
                        "Google Gemini returned an empty response for debugging. "
                        "Please verify the inputs and try again."
                    ),
                    provider_name=self.provider_name,
                    model_name=self.get_model_name(),
                )

            raw_text = response.text
            parsed = self._parse_sections(raw_text)

            return ProviderResponse(
                success=True,
                explanation=parsed["explanation"],
                code=parsed["code"],
                key_lines_explanation=parsed["key_lines_explanation"],
                change_summary=parsed["change_summary"],
                verification_steps=parsed["verification_steps"],
                raw_content=raw_text,
                provider_name=self.provider_name,
                model_name=self.get_model_name(),
            )

        except genai_errors.APIError as e:
            sanitized = utils.sanitize_error_message(e, self.provider_name, api_key=self._api_key)
            return ProviderResponse(
                success=False,
                error_message=sanitized,
                provider_name=self.provider_name,
                model_name=self.get_model_name(),
            )
        except Exception as e:
            sanitized = utils.sanitize_error_message(e, self.provider_name, api_key=self._api_key)
            return ProviderResponse(
                success=False,
                error_message=sanitized,
                provider_name=self.provider_name,
                model_name=self.get_model_name(),
            )
