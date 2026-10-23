"""
test_ai_service.py - Unit tests for AI service and Gemini provider with mocks
Academic automated testing suite. Does NOT require a real API key or internet access.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

import ai_service
import prompts
import utils
from providers.base import ProviderResponse
from providers.gemini_provider import GeminiProvider


# ---------------------------------------------------------------------------
# Test Fixtures & Sample Mock Responses
# ---------------------------------------------------------------------------

MOCK_GENERATION_TEXT = """
### 1. Approach & Conceptual Overview
To calculate the factorial of a number in Python, we can use recursion or the math module.
Here we implement a beginner-friendly iterative approach using a simple for-loop.

### 2. Python Code
```python
def factorial(n: int) -> int:
    \"\"\"Calculate the factorial of a non-negative integer.\"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

### 3. Explanation of Key Lines
Line 6 initializes `result` to 1 because multiplying by 1 preserves values.
Line 7 loops from 1 up to `n` inclusive.

### 4. How to Test & Learn
Run `factorial(5)` in your Python console; the expected output is 120.
"""

MOCK_DEBUGGING_TEXT = """
### 1. Root Cause Analysis
The original code raised an `IndexError: list index out of range` because the loop condition
used `len(items)` as an inclusive upper index, but Python sequences are 0-indexed.

### 2. Corrected Python Code
```python
def print_first_and_last(items: list) -> None:
    if not items:
        print("List is empty")
        return
    print(f"First: {items[0]}, Last: {items[-1]}")
```

### 3. Summary of Changes
- Changed the indexing of the last item to `items[-1]` which is safe and idiomatic.
- Added a check for empty lists to prevent runtime crashes.

### 4. Verification Steps & Edge Cases
Test with an empty list `[]` to verify boundary handling, and test with `[10, 20, 30]`.
"""


# ---------------------------------------------------------------------------
# Provider Factory & Deferred Provider Tests
# ---------------------------------------------------------------------------

def test_available_providers():
    """Verify that all three specified providers are registered."""
    providers = ai_service.get_available_providers()
    assert "Google Gemini" in providers
    assert "OpenAI" in providers
    assert "Anthropic Claude" in providers


def test_provider_factory_gemini():
    """Verify the factory returns a GeminiProvider instance."""
    provider = ai_service.get_provider("Google Gemini")
    assert isinstance(provider, GeminiProvider)
    assert provider.provider_name == "Google Gemini"


def test_provider_factory_unsupported():
    """Verify the factory rejects unsupported providers."""
    with pytest.raises(ValueError, match="Unsupported provider"):
        ai_service.get_provider("NonExistentProvider")


def test_deferred_provider_phase_handling():
    """
    Verify that selecting deferred providers (OpenAI / Anthropic) in Phase 3
    returns an informative non-crashing response.
    """
    resp = ai_service.generate_code("Write hello world", provider_name="OpenAI")
    assert not resp.success
    assert "Phase 6" in resp.error_message

    resp_debug = ai_service.debug_code("x = 1", "err", provider_name="Anthropic Claude")
    assert not resp_debug.success
    assert "Phase 6" in resp_debug.error_message


# ---------------------------------------------------------------------------
# Input Validation Tests (No API Calls Triggered)
# ---------------------------------------------------------------------------

def test_generate_code_blank_prompt():
    """Blank or whitespace prompts must be rejected cleanly."""
    resp = ai_service.generate_code("")
    assert not resp.success
    assert "Please enter a description" in resp.error_message

    resp_whitespace = ai_service.generate_code("   \n\t  ")
    assert not resp_whitespace.success


def test_generate_code_oversized_prompt():
    """Oversized prompts (>10,000 chars) must be rejected."""
    oversized = "a" * (utils.MAX_PROMPT_LENGTH + 1)
    resp = ai_service.generate_code(oversized)
    assert not resp.success
    assert "too long" in resp.error_message


def test_debug_code_blank_inputs():
    """Missing or whitespace-only code must be rejected cleanly, while empty error message is accepted."""
    resp1 = ai_service.debug_code("", "ZeroDivisionError")
    assert not resp1.success
    assert "paste the Python code" in resp1.error_message

    resp_ws = ai_service.debug_code("   \n\t  ", "ZeroDivisionError")
    assert not resp_ws.success
    assert "paste the Python code" in resp_ws.error_message

    # Empty error message and expected behavior are now allowed (validated as True)
    is_valid, err = utils.validate_debugging_input("x = 1 / 0", error_message="", expected_behavior="")
    assert is_valid is True
    assert err is None


def test_debug_code_oversized_inputs():
    """Oversized code or error message must be rejected."""
    oversized_code = "x = 1\n" * (utils.MAX_CODE_LENGTH + 10)
    resp = ai_service.debug_code(oversized_code, "SyntaxError")
    assert not resp.success
    assert "too long" in resp.error_message


# ---------------------------------------------------------------------------
# Gemini Provider Tests (Mocked google-genai SDK)
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_generate_code_success(mock_client_class):
    """Test successful Python code generation with mocked Gemini response."""
    # Setup mock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = MOCK_GENERATION_TEXT
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.generate_code(
        prompt="Write a function to calculate factorial",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert "def factorial" in response.code
    assert "Approach" in response.explanation or "factorial" in response.explanation.lower()
    assert response.provider_name == "Google Gemini"
    expected_model = ai_service.get_provider("Google Gemini").get_model_name()
    assert response.model_name == expected_model
    assert "factorial(5)" in response.verification_steps
    assert response.key_lines_explanation is not None
    assert "Line 6" in response.key_lines_explanation


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_code_success(mock_client_class):
    """Test successful Python code debugging with mocked Gemini response."""
    # Setup mock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = MOCK_DEBUGGING_TEXT
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="items = [1, 2]; print(items[len(items)])",
        error_message="IndexError: list index out of range",
        expected_behavior="Print the first and last item of the list",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert "def print_first_and_last" in response.code
    assert "IndexError" in response.explanation
    assert response.change_summary is not None
    assert response.verification_steps is not None
    assert response.provider_name == "Google Gemini"


@patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True)
def test_gemini_missing_api_key():
    """Verify that missing GEMINI_API_KEY returns clear setup instructions."""
    provider = GeminiProvider(api_key="")
    response = provider.generate_code("Write a bubble sort")

    assert response.success is False
    assert "GEMINI_API_KEY is not configured" in response.error_message
    assert "https://aistudio.google.com/" in response.error_message


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_empty_model_response(mock_client_class):
    """Verify graceful handling when Gemini returns an empty response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ""  # Empty text
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.generate_code(
        prompt="Write something",
        provider_name="Google Gemini",
    )

    assert response.success is False
    assert "empty response" in response.error_message.lower()


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_api_error_handling(mock_client_class):
    """Verify handling of APIError and status codes like 429 / Rate Limit."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 Resource has been exhausted (quota)")
    mock_client_class.return_value = mock_client

    response = ai_service.generate_code(
        prompt="Write a binary search tree",
        provider_name="Google Gemini",
    )

    assert response.success is False
    assert "quota" in response.error_message.lower() or "rate limit" in response.error_message.lower()


# ---------------------------------------------------------------------------
# Secret Protection & Sanitization Tests
# ---------------------------------------------------------------------------

def test_secret_masking():
    """Assert mask_api_key hides private keys and never exposes full string."""
    fake_key = "AIzaSySecretApiKeyForGemini12345"
    masked = utils.mask_api_key(fake_key)
    assert fake_key not in masked
    assert masked.endswith("2345")
    assert "••••" in masked

    assert utils.mask_api_key(None) == "[Not Configured]"
    assert utils.mask_api_key("") == "[Not Configured]"


def test_secret_sanitization_in_errors():
    """Assert that raw credentials in exception strings are redacted."""
    dummy_key = "AIzaSySensitiveSecretKey123456789"
    raw_error = Exception(f"Failed to authenticate with key {dummy_key} at endpoint")
    sanitized = utils.sanitize_error_message(raw_error, "Google Gemini")

    assert dummy_key not in sanitized
    assert "Authentication failed" in sanitized or "[REDACTED_API_KEY]" in sanitized

    # Verify 503 produces friendly message without raw JSON
    raw_503 = Exception("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}")
    sanitized_503 = utils.sanitize_error_message(raw_503, "Google Gemini")
    assert "temporary high server demand" in sanitized_503
    assert "{'error'" not in sanitized_503


# ---------------------------------------------------------------------------
# Retry Mechanism & Resilience Tests (Mocked - No Live Calls)
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.time.sleep", return_value=None)
@patch("providers.gemini_provider.genai.Client")
def test_gemini_generate_code_transient_503_retry_success(mock_client_class, mock_sleep):
    """
    Test transient 503 error on code generation:
    Attempt 1 fails with 503 High Demand, Attempt 2 succeeds.
    Verifies generate_code call_count == 2 and mock_sleep was called with 2.0s delay.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = MOCK_GENERATION_TEXT

    err_503 = Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")
    mock_client.models.generate_content.side_effect = [err_503, mock_response]
    mock_client_class.return_value = mock_client

    response = ai_service.generate_code(
        prompt="Write a function to calculate factorial",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(2.0)
    assert "def factorial" in response.code


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.time.sleep", return_value=None)
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_transient_503_retry_success(mock_client_class, mock_sleep):
    """
    Test transient 503 error:
    Attempt 1 fails with 503 High Demand, Attempt 2 succeeds.
    Verifies call_count == 2 and mock_sleep called with 2.0s delay.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = MOCK_DEBUGGING_TEXT

    err_503 = Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")
    mock_client.models.generate_content.side_effect = [err_503, mock_response]
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def add(a, b): return a - b",
        error_message="Returns difference instead of sum",
        expected_behavior="Should return the sum of a and b",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(2.0)
    assert response.code is not None


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.time.sleep", return_value=None)
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_repeated_503_exhausted_retries(mock_client_class, mock_sleep):
    """
    Test repeated 503 errors:
    All 3 attempts fail with 503 High Demand.
    Verifies exactly 3 attempts (initial + 2 retries), mock_sleep called twice (2.0s, 4.0s),
    and response fails gracefully with a sanitized message.
    """
    mock_client = MagicMock()
    err_503 = Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")
    mock_client.models.generate_content.side_effect = err_503
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def add(a, b): return a - b",
        error_message="Returns difference instead of sum",
        expected_behavior="Should return the sum of a and b",
        provider_name="Google Gemini",
    )

    assert response.success is False
    assert mock_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)
    assert "503" in response.error_message or "high demand" in response.error_message.lower()


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.time.sleep", return_value=None)
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_non_transient_error_no_retry(mock_client_class, mock_sleep):
    """
    Test non-transient error (e.g. 401 Unauthorized):
    Verifies that it fails immediately without retrying (call_count == 1, sleep never called).
    """
    mock_client = MagicMock()
    err_401 = Exception("401 Unauthorized. Invalid API key provided.")
    mock_client.models.generate_content.side_effect = err_401
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def add(a, b): return a - b",
        error_message="Returns difference instead of sum",
        expected_behavior="Should return the sum of a and b",
        provider_name="Google Gemini",
    )

    assert response.success is False
    assert mock_client.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()
    assert "authentication failed" in response.error_message.lower()


# ---------------------------------------------------------------------------
# Stage 1 Quality Assurance Tests: Section Parsing, Key Lines, & Clean Code
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_code_ignores_section_1_snippet(mock_client_class):
    """
    Verify that when Section 1 contains a quoted snippet of buggy code in a code block,
    the parser extracts the corrected solution from Section 2 instead of the buggy snippet.
    """
    mock_text_with_snippet_in_section_1 = """
### 1. Root Cause Analysis
The student's original code has a bug:
```python
def buggy_add(a, b):
    return a - b
```
Because subtraction was used instead of addition.

### 2. Corrected Python Code
```python
def corrected_add(a, b: int) -> int:
    \"\"\"Return the sum of a and b.\"\"\"
    return a + b
```

### 3. Summary of Changes
- Replaced minus operator with plus operator.

### 4. Verification Steps & Edge Cases
Call `corrected_add(2, 3)` and verify it returns 5.
"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = mock_text_with_snippet_in_section_1
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def buggy_add(a, b): return a - b",
        error_message="Returns subtraction instead of addition",
        expected_behavior="Return sum",
        provider_name="Google Gemini",
    )

    assert response.success is True
    # Crucial assertion: Must contain the corrected function, NOT the buggy one from Section 1
    assert "def corrected_add" in response.code
    assert "buggy_add" not in response.code
    assert response.change_summary is not None
    assert "minus operator" in response.change_summary


def test_extract_python_code_no_fences_returns_empty():
    """
    Verify conversational Markdown without code fences returns an empty string
    to prevent conversational prose from being displayed as Python code.
    """
    conversational_text = (
        "I cannot provide a code snippet for this request because it is ambiguous.\n"
        "Please provide more context or sample input/output."
    )
    extracted = utils.extract_python_code(conversational_text)
    assert extracted == ""
    assert utils.extract_python_code("") == ""
    assert utils.extract_python_code(None) == ""


def test_extract_python_code_with_python_fence():
    """Verify that Python code fences are properly extracted and stripped."""
    text = "Introductory text\n```python\ndef greet():\n    return 'hello'\n```\nTrailing text"
    extracted = utils.extract_python_code(text)
    assert extracted == "def greet():\n    return 'hello'"


def test_extract_python_code_with_generic_fence():
    """Verify that generic code fences are extracted as a fallback."""
    text = "Introductory text\n```\nx = 42\n```"
    extracted = utils.extract_python_code(text)
    assert extracted == "x = 42"


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_conversational_response_no_code(mock_client_class):
    """
    Verify that if the AI returns only conversational text without code blocks,
    response.code is empty and does not leak conversational markdown into code box.
    """
    conversational_response = """
### 1. Approach & Conceptual Overview
Your question is purely theoretical. The time complexity of binary search is O(log n).

### 2. Python Code
No code implementation is necessary for this theoretical query.

### 3. Explanation of Key Lines
N/A

### 4. How to Test & Learn
Review the master theorem in your algorithms textbook.
"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = conversational_response
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.generate_code(
        prompt="Explain the time complexity of binary search",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert response.code == ""
    assert "O(log n)" in response.explanation
    assert response.key_lines_explanation == "N/A"


# ---------------------------------------------------------------------------
# Stage 2 Quality Assurance Tests: Error Handling, Retries, & Input Limits
# ---------------------------------------------------------------------------

def test_is_transient_overload_error_broadened_503():
    """Verify that broadened 503 errors (e.g. Backend Error, code=503) are detected as transient."""
    provider = GeminiProvider(api_key="AIzaFakeTestKeyForUnitTestingOnly123")

    # 1. 503 without explicit "high demand" phrase
    err_503_backend = Exception("503 Backend Error")
    assert provider._is_transient_overload_error(err_503_backend) is True

    # 2. Object with code == 503
    mock_code_503 = Exception("Temporary failure")
    mock_code_503.code = 503
    assert provider._is_transient_overload_error(mock_code_503) is True

    # 3. Object with status == 'UNAVAILABLE'
    mock_status_unavail = Exception("Service unavailable")
    mock_status_unavail.status = "UNAVAILABLE"
    assert provider._is_transient_overload_error(mock_status_unavail) is True

    # 4. Phrase-based indicators
    assert provider._is_transient_overload_error(Exception("The model is overloaded right now")) is True


def test_is_transient_overload_error_non_transient_fast_fail():
    """Verify that non-transient client and system errors are NEVER marked as transient."""
    provider = GeminiProvider(api_key="AIzaFakeTestKeyForUnitTestingOnly123")

    # 400 Bad Request
    err_400 = Exception("400 Bad Request: Invalid arguments provided.")
    err_400.code = 400
    assert provider._is_transient_overload_error(err_400) is False

    # 401 Unauthorized
    err_401 = Exception("401 Unauthorized: Invalid API key.")
    err_401.code = 401
    assert provider._is_transient_overload_error(err_401) is False

    # 403 Forbidden
    err_403 = Exception("403 Forbidden: Permission denied.")
    err_403.code = 403
    assert provider._is_transient_overload_error(err_403) is False

    # 404 Model Not Found
    err_404 = Exception("404 Model not found.")
    err_404.code = 404
    assert provider._is_transient_overload_error(err_404) is False

    # 429 Quota Exhausted
    err_429 = Exception("429 Resource has been exhausted (quota).")
    err_429.code = 429
    assert provider._is_transient_overload_error(err_429) is False

    # Standard exceptions without 503
    assert provider._is_transient_overload_error(ValueError("Invalid syntax")) is False
    assert provider._is_transient_overload_error(KeyError("missing_key")) is False


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.time.sleep", return_value=None)
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_503_backend_error_retries_and_succeeds(mock_client_class, mock_sleep):
    """
    Verify that an HTTP 503 'Backend Error' (which previously would have failed immediately)
    is now properly retried and succeeds on attempt 2.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = MOCK_DEBUGGING_TEXT

    err_503_backend = Exception("503 Backend Error: Internal server error at upstream.")
    mock_client.models.generate_content.side_effect = [err_503_backend, mock_response]
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def divide(a, b): return a / b",
        error_message="ZeroDivisionError",
        expected_behavior="Return 0 when b is 0",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(2.0)


def test_validate_debugging_input_expected_behavior():
    """Verify input validation handles expected_behavior lengths correctly."""
    # Valid expected behavior
    is_valid, msg = utils.validate_debugging_input("x = 1", "err", "returns 1")
    assert is_valid is True
    assert msg is None

    # None is permitted
    is_valid_none, msg_none = utils.validate_debugging_input("x = 1", "err", None)
    assert is_valid_none is True
    assert msg_none is None

    # Oversized expected behavior exceeds limit
    oversized = "x" * (utils.MAX_EXPECTED_BEHAVIOR_LENGTH + 1)
    is_valid_oversized, msg_oversized = utils.validate_debugging_input("x = 1", "err", oversized)
    assert is_valid_oversized is False
    assert "expected behavior description is too long" in msg_oversized


def test_debug_code_oversized_expected_behavior():
    """Verify debug_code rejects oversized expected_behavior before calling provider."""
    oversized = "x" * (utils.MAX_EXPECTED_BEHAVIOR_LENGTH + 1)
    response = ai_service.debug_code(
        code="def foo(): pass",
        error_message="SyntaxError",
        expected_behavior=oversized,
    )
    assert response.success is False
    assert "expected behavior description is too long" in response.error_message


def test_secret_sanitization_actual_configured_keys():
    """
    Verify that actual configured API-key values (even without standard AIza/sk/ant prefixes)
    are strictly redacted from error messages for Gemini, OpenAI, and Anthropic.
    """
    dummy_gemini = "custom_test_gemini_token_987654321"
    dummy_openai = "custom_test_openai_token_123456789"
    dummy_anthropic = "custom_test_anthropic_token_555666777"

    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": dummy_gemini,
            "OPENAI_API_KEY": dummy_openai,
            "ANTHROPIC_API_KEY": dummy_anthropic,
        },
    ):
        raw_error = Exception(
            f"Failed calling backend with gemini key={dummy_gemini}, openai key={dummy_openai}, "
            f"and anthropic key={dummy_anthropic}."
        )
        sanitized = utils.sanitize_error_message(raw_error, "Google Gemini")

        # Confirm none of the actual keys appear in the sanitized output
        assert dummy_gemini not in sanitized
        assert dummy_openai not in sanitized
        assert dummy_anthropic not in sanitized
        assert "[REDACTED_API_KEY]" in sanitized


def test_secret_sanitization_direct_api_key_parameter():
    """Verify explicitly passed api_key parameter is redacted even if not in os.environ."""
    direct_key = "direct_injected_test_key_abc999"
    raw_error = Exception(f"HTTP call rejected for {direct_key}")
    sanitized = utils.sanitize_error_message(raw_error, "Google Gemini", api_key=direct_key)

    assert direct_key not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


# ---------------------------------------------------------------------------
# Stage 3 Quality Assurance Tests: Coverage Expansion
# ---------------------------------------------------------------------------

# 1. utils.get_provider_status tests
def test_get_provider_status_configured():
    """Verify get_provider_status returns configured=True and masked key when key is set."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeTestKey12345", "GEMINI_MODEL": "gemini-2.5-flash"}):
        status = utils.get_provider_status("Google Gemini")
        assert status["configured"] is True
        assert status["provider_name"] == "Google Gemini"
        assert status["masked_key"] == "••••••••••••2345"
        assert status["model"] == "gemini-2.5-flash"
        assert "aistudio.google.com" in status["docs_url"]


def test_get_provider_status_missing_key():
    """Verify get_provider_status returns configured=False and setup hint when key is missing."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True):
        status = utils.get_provider_status("Google Gemini")
        assert status["configured"] is False
        assert status["masked_key"] == "[Not Configured]"
        assert "GEMINI_API_KEY" in status["setup_hint"]


def test_get_provider_status_unsupported():
    """Verify get_provider_status handles unsupported provider gracefully."""
    status = utils.get_provider_status("UnsupportedBackend")
    assert status["configured"] is False
    assert status["masked_key"] == "[Unsupported Provider]"
    assert status["model"] == "unknown"


# 2. utils.extract_python_code expansion tests
def test_extract_python_code_multiple_blocks():
    """Verify that when multiple code blocks exist, the primary block is safely extracted."""
    text = (
        "Here is block 1:\n```python\nx = 10\n```\n"
        "Here is block 2:\n```python\ny = 20\n```\n"
    )
    extracted = utils.extract_python_code(text)
    assert extracted == "x = 10"


def test_extract_python_code_case_insensitive_fence():
    """Verify that case-insensitive fences like ```PYTHON are properly extracted."""
    text = "Code below:\n```PYTHON\nz = 30\n```"
    extracted = utils.extract_python_code(text)
    assert extracted == "z = 30"


# 3. prompts.build_generation_prompt tests
def test_build_generation_prompt_without_context():
    """Verify prompt layout without optional context."""
    p = prompts.build_generation_prompt("Write a binary search function")
    assert prompts.SYSTEM_INSTRUCTION in p
    assert "Write a binary search function" in p
    assert "### Additional Context or Constraints:" not in p
    assert "### 1. Approach & Conceptual Overview" in p
    assert "### 2. Python Code" in p
    assert "### 3. Explanation of Key Lines" in p
    assert "### 4. How to Test & Learn" in p


def test_build_generation_prompt_with_context():
    """Verify optional context section is included when provided."""
    p = prompts.build_generation_prompt(
        "Write a sorting function",
        context="Use only recursion, no loops.",
    )
    assert "### Additional Context or Constraints:" in p
    assert "Use only recursion, no loops." in p


def test_build_generation_prompt_with_whitespace_context():
    """Verify whitespace context is ignored and not included."""
    p = prompts.build_generation_prompt("Write hello world", context="   \n\t  ")
    assert "### Additional Context or Constraints:" not in p


# 4. prompts.build_debugging_prompt tests
def test_build_debugging_prompt_without_expected():
    """Verify debugging prompt layout without optional expected_behavior."""
    p = prompts.build_debugging_prompt(
        code="print(1/0)",
        error_message="ZeroDivisionError",
    )
    assert prompts.SYSTEM_INSTRUCTION in p
    assert "print(1/0)" in p
    assert "ZeroDivisionError" in p
    assert "### Expected Behavior" not in p
    assert "### 1. Root Cause Analysis" in p
    assert "### 2. Corrected Python Code" in p
    assert "### 3. Summary of Changes" in p
    assert "### 4. Verification Steps & Edge Cases" in p


def test_build_debugging_prompt_with_expected():
    """Verify debugging prompt includes expected behavior when provided."""
    p = prompts.build_debugging_prompt(
        code="x = [1]; print(x[5])",
        error_message="IndexError",
        expected_behavior="Should print None if out of bounds",
    )
    assert "### Expected Behavior" in p
    assert "Should print None if out of bounds" in p


def test_build_debugging_prompt_with_whitespace_expected():
    """Verify whitespace expected_behavior is ignored and omitted."""
    p = prompts.build_debugging_prompt("x = 1", "err", expected_behavior="   ")
    assert "### Expected Behavior" not in p


# 5. utils.sanitize_error_message scenarios
def test_sanitize_error_message_429_quota():
    """Verify 429 quota exhaustion produces helpful quota message."""
    err = Exception("429 Resource has been exhausted (quota exceeded for model).")
    sanitized = utils.sanitize_error_message(err, "Google Gemini")
    assert "Rate limit or quota reached" in sanitized


def test_sanitize_error_message_404_model():
    """Verify 404 model not found produces model verification message."""
    err = Exception("404 Model gemini-nonexistent not found.")
    sanitized = utils.sanitize_error_message(err, "Google Gemini")
    assert "selected model was not found" in sanitized


def test_sanitize_error_message_timeout():
    """Verify network timeout errors produce connection instructions."""
    err = Exception("ConnectTimeout: connection timed out after 30000ms")
    sanitized = utils.sanitize_error_message(err, "Google Gemini")
    assert "timed out" in sanitized
    assert "check your internet connection" in sanitized


def test_sanitize_error_message_redacts_auth_headers_and_sk_tokens():
    """Verify sk- and token= headers are sanitized."""
    err = Exception("Failed with token='dummy_bearer_token' and sk-1234567890abcdef1234567890")
    sanitized = utils.sanitize_error_message(err, "OpenAI")
    assert "1234567890abcdef1234567890" not in sanitized
    assert "dummy_bearer_token" not in sanitized


def test_sanitize_error_message_generic_fallback():
    """Verify unclassified exception produces formatted fallback."""
    err = Exception("Unexpected backend parsing anomaly occurred")
    sanitized = utils.sanitize_error_message(err, "Google Gemini")
    assert sanitized == "Google Gemini request failed: Unexpected backend parsing anomaly occurred"


# 6. providers/gemini_provider.py section parsing tests
def test_parse_sections_heading_variations():
    """
    Verify _parse_sections handles heading variations such as two hashes (##)
    and headings without numeric prefixes.
    """
    provider = GeminiProvider(api_key="AIzaFakeTestKeyForUnitTestingOnly123")
    raw_markdown = """
## Approach & Conceptual Overview
This is an overview of the algorithm using two hashes.

## Python Code
```python
def solve():
    return 42
```

## Explanation of Key Lines
Line 2 returns the answer to life.

## How to Test
Call solve() and assert it equals 42.
"""
    parsed = provider._parse_sections(raw_markdown)
    assert "overview of the algorithm" in parsed["explanation"]
    assert parsed["code"] == "def solve():\n    return 42"
    assert "Line 2 returns" in parsed["key_lines_explanation"]
    assert "Call solve()" in parsed["verification_steps"]


def test_parse_sections_missing_sections_fallback():
    """
    Verify _parse_sections falls back gracefully when explicit section headers are absent.
    """
    provider = GeminiProvider(api_key="AIzaFakeTestKeyForUnitTestingOnly123")
    raw_markdown = """
Here is a plain explanation without standard headers.
```python
def helper():
    return True
```
Hope this helps!
"""
    parsed = provider._parse_sections(raw_markdown)
    assert parsed["code"] == "def helper():\n    return True"
    assert "Here is a plain explanation" in parsed["explanation"]
    assert parsed["key_lines_explanation"] is None
    assert parsed["change_summary"] is None


def test_parse_sections_multiple_code_blocks_scopes_to_section_2():
    """
    Verify _parse_sections isolates Section 2 code even when Section 1 and Section 4
    also contain code blocks.
    """
    provider = GeminiProvider(api_key="AIzaFakeTestKeyForUnitTestingOnly123")
    raw_markdown = """
### 1. Root Cause Analysis
The student's broken snippet:
```python
bad_val = 1 / 0
```
was failing due to division by zero.

### 2. Corrected Python Code
```python
def safe_divide(a, b):
    return a / b if b != 0 else 0.0
```

### 3. Summary of Changes
- Added zero check.

### 4. Verification Steps & Edge Cases
You can test with:
```python
assert safe_divide(1, 0) == 0.0
```
"""
    parsed = provider._parse_sections(raw_markdown)
    # Must be the code from Section 2, not Section 1 or Section 4
    assert parsed["code"] == "def safe_divide(a, b):\n    return a / b if b != 0 else 0.0"
    assert "safe_divide" in parsed["code"]
    assert "bad_val" not in parsed["code"]
    assert "assert safe_divide" not in parsed["code"]
    assert parsed["change_summary"] is not None


# ---------------------------------------------------------------------------
# ProviderResponse Contract & Attribute Tests
# ---------------------------------------------------------------------------

def test_provider_response_key_lines_explanation_attribute():
    """
    Verify that ProviderResponse class defines key_lines_explanation
    with a safe default of None and preserves assigned values.
    """
    resp_default = ProviderResponse(success=True)
    assert hasattr(resp_default, "key_lines_explanation")
    assert resp_default.key_lines_explanation is None

    resp_assigned = ProviderResponse(
        success=True,
        key_lines_explanation="Lines 1-3 initialize the loop counter.",
    )
    assert resp_assigned.key_lines_explanation == "Lines 1-3 initialize the loop counter."


def test_ai_service_generate_code_preserves_response_model():
    """
    Verify that all code generation return paths (validation failure, deferred provider,
    unsupported provider, success) return a ProviderResponse instance that defines
    key_lines_explanation.
    """
    # 1. Validation failure return path
    resp_invalid = ai_service.generate_code("")
    assert isinstance(resp_invalid, ProviderResponse)
    assert hasattr(resp_invalid, "key_lines_explanation")
    assert resp_invalid.key_lines_explanation is None

    # 2. Deferred provider return path
    resp_deferred = ai_service.generate_code("print('hi')", provider_name="OpenAI")
    assert isinstance(resp_deferred, ProviderResponse)
    assert hasattr(resp_deferred, "key_lines_explanation")
    assert resp_deferred.key_lines_explanation is None

    # 3. Unsupported provider return path
    resp_unsupported = ai_service.generate_code("print('hi')", provider_name="NonExistentProvider")
    assert isinstance(resp_unsupported, ProviderResponse)
    assert hasattr(resp_unsupported, "key_lines_explanation")
    assert resp_unsupported.key_lines_explanation is None


# ---------------------------------------------------------------------------
# Optional Error Message & Debugging Scenarios Tests
# ---------------------------------------------------------------------------

def test_build_debugging_prompt_code_only():
    """
    Scenario A: When neither error_message nor expected_behavior is supplied,
    the prompt instructs the AI to perform static analysis and code review.
    """
    p = prompts.build_debugging_prompt(
        code="def divide(a, b):\n    return a / b",
        error_message=None,
        expected_behavior=None,
    )
    assert prompts.SYSTEM_INSTRUCTION in p
    assert "def divide(a, b):" in p
    assert "### Error Message or Observed Problem:" not in p
    assert "### Expected Behavior" not in p
    assert "syntax errors" in p or "logical pitfalls" in p
    assert "### 1. Root Cause Analysis" in p
    assert "### 2. Corrected Python Code" in p


def test_build_debugging_prompt_code_plus_expected_only():
    """
    Scenario C: When expected_behavior is provided without an error message,
    the prompt instructs the AI to compare the code against the intended outcome.
    """
    p = prompts.build_debugging_prompt(
        code="def add(a, b):\n    return a - b",
        error_message="",
        expected_behavior="Should return the sum of a and b",
    )
    assert "### Error Message or Observed Problem:" not in p
    assert "### Expected Behavior (as described by student):" in p
    assert "Should return the sum of a and b" in p
    assert "expected behavior" in p.lower()


def test_validate_debugging_input_all_scenarios():
    """
    Verify input validation across all combinations:
    - Code only -> Valid
    - Code + Error -> Valid
    - Code + Expected -> Valid
    - Code + Error + Expected -> Valid
    - Missing code -> Invalid
    """
    # 1. Code only
    ok, err = utils.validate_debugging_input("x = 1")
    assert ok is True and err is None

    ok, err = utils.validate_debugging_input("x = 1", error_message=None, expected_behavior=None)
    assert ok is True and err is None

    ok, err = utils.validate_debugging_input("x = 1", error_message="   ", expected_behavior="   ")
    assert ok is True and err is None

    # 2. Code + Error only
    ok, err = utils.validate_debugging_input("x = 1", error_message="ZeroDivisionError")
    assert ok is True and err is None

    # 3. Code + Expected only
    ok, err = utils.validate_debugging_input("x = 1", expected_behavior="Returns 1")
    assert ok is True and err is None

    # 4. Code + Error + Expected
    ok, err = utils.validate_debugging_input("x = 1", error_message="ZeroDivisionError", expected_behavior="Returns 1")
    assert ok is True and err is None

    # 5. Missing code rejected
    ok, err = utils.validate_debugging_input("")
    assert ok is False and "paste the Python code" in err

    ok, err = utils.validate_debugging_input("   \n\t  ")
    assert ok is False and "paste the Python code" in err

    ok, err = utils.validate_debugging_input(None)
    assert ok is False and "paste the Python code" in err


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_code_only_mock_success(mock_client_class):
    """
    Scenario A end-to-end: Debugging request with code only (no error message,
    no expected behavior) is accepted by ai_service and successfully processed.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = MOCK_DEBUGGING_TEXT
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="items = [1, 2]; print(items[len(items)])",
        error_message=None,
        expected_behavior=None,
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert response.code is not None
    assert mock_client.models.generate_content.call_count == 1
    # Verify the generated prompt did not include an empty error block
    called_prompt = mock_client.models.generate_content.call_args[1]["contents"]
    assert "### Error Message or Observed Problem:" not in called_prompt


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_code_plus_expected_only_mock_success(mock_client_class):
    """
    Scenario C end-to-end: Debugging request with code and expected behavior,
    without error message, succeeds and embeds expected behavior in prompt.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = MOCK_DEBUGGING_TEXT
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def multiply(a, b): return a + b",
        error_message="",
        expected_behavior="Should return the product of a and b",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert mock_client.models.generate_content.call_count == 1
    called_prompt = mock_client.models.generate_content.call_args[1]["contents"]
    assert "### Error Message or Observed Problem:" not in called_prompt
    assert "Should return the product of a and b" in called_prompt


def test_build_debugging_prompt_exact_no_error_phrase():
    """
    Verify prompt explicitly instructs the model to use the exact phrase:
    'No error was identified. The code appears correct for the intended behavior.'
    and instructs not to invent errors or rewrite code unnecessarily.
    """
    p = prompts.build_debugging_prompt(
        code="def square(x): return x * x",
        error_message=None,
        expected_behavior="Returns the square of x",
    )
    assert "No error was identified. The code appears correct for the intended behavior." in p
    assert "Do not invent an error" in p or "do not invent an error" in p.lower()
    assert "do not rewrite the code unnecessarily" in p.lower()


def test_build_debugging_prompt_unclear_behavior_instruction():
    """
    Verify prompt instructs model to state limitation when intended behavior is unclear
    rather than falsely claiming the code is correct.
    """
    p = prompts.build_debugging_prompt(
        code="def process(data): return data",
        error_message=None,
        expected_behavior=None,
    )
    assert "intended behavior is unclear" in p.lower()
    assert "state that limitation clearly" in p.lower()


def test_build_debugging_prompt_distinguishes_static_review():
    """
    Verify prompt clearly distinguishes static code review from runtime code execution.
    """
    p = prompts.build_debugging_prompt(code="x = 10")
    assert "distinguish static code review from runtime code execution" in p.lower()


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_apparently_correct_code_mock(mock_client_class):
    """
    Verify debugging workflow when submitted code is correct:
    - Provider returns success
    - Root cause analysis contains the exact 'No error was identified' phrase
    - No unnecessary code block is produced
    - Change summary records that no changes were needed
    """
    mock_correct_text = """
### 1. Root Cause Analysis
No error was identified. The code appears correct for the intended behavior.
The function `is_even(n)` correctly tests divisibility by 2 using the modulo operator.

### 2. Corrected Python Code
No code modifications are required.

### 3. Summary of Changes
No changes needed because the original code already meets requirements.

### 4. Verification Steps & Edge Cases
Call `is_even(0)` and `is_even(-2)` to verify behavior with zero and negative integers.
"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = mock_correct_text
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def is_even(n):\n    return n % 2 == 0",
        error_message=None,
        expected_behavior="Return True for even numbers, False for odd",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert "No error was identified. The code appears correct for the intended behavior." in response.explanation
    assert response.code == ""
    assert "No changes needed" in response.change_summary
    assert "is_even(0)" in response.verification_steps


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_unclear_intended_behavior_mock(mock_client_class):
    """
    Verify debugging workflow when code has no syntax error but intended behavior is ambiguous:
    - Provider returns success
    - Explanation clearly states the limitation that intended behavior is unclear
    - No code changes are proposed
    """
    mock_unclear_text = """
### 1. Root Cause Analysis
The code is syntactically valid, but the intended behavior is unclear.
Without an error message or expected behavior description, it is not possible to determine
whether `transform(x)` should return a modified copy or mutate the list in place.

### 2. Corrected Python Code
No code modifications are required pending clarification.

### 3. Summary of Changes
No changes made.

### 4. Verification Steps & Edge Cases
Clarify whether in-place mutation or returning a new list is expected.
"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = mock_unclear_text
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def transform(x):\n    return x",
        error_message=None,
        expected_behavior=None,
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert "intended behavior is unclear" in response.explanation.lower()
    assert response.code == ""
    assert "No changes made" in response.change_summary


@patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForUnitTestingOnly123"})
@patch("providers.gemini_provider.genai.Client")
def test_gemini_debug_identifiable_error_mock(mock_client_class):
    """
    Verify debugging workflow when code has an identifiable error:
    - Provider returns success
    - Root cause analysis explains the bug
    - Corrected code is extracted from Section 2
    - Summary of changes is captured
    """
    mock_buggy_text = """
### 1. Root Cause Analysis
Identifiable error: ZeroDivisionError occurs when the `numbers` list is empty because `len(numbers)` is 0.

### 2. Corrected Python Code
```python
def calculate_average(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
```

### 3. Summary of Changes
- Added guard clause to return 0.0 when numbers is empty.

### 4. Verification Steps & Edge Cases
Call `calculate_average([])` and verify it returns 0.0 instead of raising ZeroDivisionError.
"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = mock_buggy_text
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    response = ai_service.debug_code(
        code="def calculate_average(numbers):\n    return sum(numbers) / len(numbers)",
        error_message="ZeroDivisionError: division by zero",
        expected_behavior="Return 0.0 for empty lists",
        provider_name="Google Gemini",
    )

    assert response.success is True
    assert "ZeroDivisionError" in response.explanation
    assert "if not numbers:" in response.code
    assert "Added guard clause" in response.change_summary







