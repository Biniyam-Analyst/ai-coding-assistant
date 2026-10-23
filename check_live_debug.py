"""
check_live_debug.py - Controlled live Gemini API debugging verification
Sends one small debugging request to test live API error diagnosis and code correction.
Never prints, logs, or exposes the private API key.
"""

import sys
import utils
import ai_service


def run_live_debugging_check():
    print("=" * 60)
    print("Controlled Live Gemini API Debugging Verification")
    print("=" * 60)

    # 1. Inspect configuration
    utils.load_environment()
    status = utils.get_provider_status("Google Gemini")

    print(f"Provider:           {status['provider_name']}")
    print(f"Configured Model:   {status['model']}")
    print(f"API Key Masked:     {status['masked_key']}")

    if not status["configured"]:
        print("\n[ERROR] GEMINI_API_KEY is not configured in .env.")
        print("Please configure your private key in .env before running this check.")
        sys.exit(1)

    # 2. Minimal test case specified for debugging
    buggy_code = "def add(a, b): return a - b"
    error_message = "Returns difference instead of sum"
    expected_behavior = "Should return the sum of a and b"

    print(f"\nSubmitting live debugging request:")
    print(f"  Code:     {buggy_code}")
    print(f"  Error:    {error_message}")
    print(f"  Expected: {expected_behavior}")
    print("\nAwaiting response from Google Gen AI API (with transient retry protection)...")

    try:
        response = ai_service.debug_code(
            code=buggy_code,
            error_message=error_message,
            expected_behavior=expected_behavior,
            provider_name="Google Gemini",
        )

        if response.success:
            print("\n[SUCCESS] Live Gemini Debugging call completed successfully!")
            print(f"Model responding:  {response.model_name}")
            print(f"\nCorrected Code:\n{'-' * 40}")
            print(response.code.strip() if response.code else "(No separate code block parsed)")
            print(f"{'-' * 40}")
            if response.change_summary:
                print(f"\nSummary of Changes:\n{response.change_summary.strip()[:200]}")
            if response.explanation:
                print(f"\nRoot Cause Preview:\n{response.explanation.strip()[:200]}...")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n[FAILURE] Live Gemini Debugging call failed.")
            print(f"Sanitized Error: {response.error_message}")
            print("=" * 60)
            sys.exit(1)

    except Exception as e:
        sanitized = utils.sanitize_error_message(e, "Google Gemini")
        print(f"\n[EXCEPTION] Unexpected error during live test: {sanitized}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    run_live_debugging_check()
