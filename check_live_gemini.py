"""
check_live_gemini.py - Controlled live Gemini API check
Sends one small prompt to verify live API connectivity and model response.
Never prints or logs the raw API key.
"""

import os
import sys
import utils
import ai_service


def run_live_check():
    print("=" * 60)
    print("Controlled Live Gemini API Connectivity Verification")
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

    # 2. Authorized single small live request
    test_prompt = "Write a Python function that adds two numbers."
    print(f"\nSending single live test prompt: '{test_prompt}'")
    print("Awaiting response from Google Gen AI API...")

    try:
        response = ai_service.generate_code(
            prompt=test_prompt,
            context="Keep it simple and include type hints and a docstring.",
            provider_name="Google Gemini",
        )

        if response.success:
            print("\n[SUCCESS] Live Gemini API call completed successfully!")
            print(f"Model responding:  {response.model_name}")
            print(f"Extracted Code:\n{'-' * 40}")
            print(response.code.strip() if response.code else "(No separate code block parsed)")
            print(f"{'-' * 40}")
            print("\nConceptual Explanation Preview:")
            # Display first 200 characters of explanation
            preview = response.explanation.strip()[:250] + ("..." if len(response.explanation) > 250 else "")
            print(preview)
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n[FAILURE] Live Gemini API call failed.")
            print(f"Sanitized Error: {response.error_message}")
            print("=" * 60)
            sys.exit(1)

    except Exception as e:
        sanitized = utils.sanitize_error_message(e, "Google Gemini")
        print(f"\n[EXCEPTION] Unexpected error during live test: {sanitized}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    run_live_check()
