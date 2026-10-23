# AI Coding Assistant

> **Academic Project:** Generative AI — Python Code Generation & Debugging for Students  
> **Backend AI Provider:** Google Gemini (`google-genai` SDK)  
> **Frontend Web Interface:** Streamlit  

---

## Overview

The **AI Coding Assistant** is an educational, interactive web application designed to help university students and beginners learn Python, computer science, and data science concepts. It leverages Google Gemini models to synthesize clean, idiomatic Python code from natural language problem descriptions and diagnose bugs from error messages and tracebacks.

All explanations are pedagogical: the assistant prioritizes conceptual understanding, best practices, line-by-line mechanical explanations, and suggested verification test cases.

---

## Core Features

### 1. Natural Language to Python Code Generation
* **Conceptual Overview:** Explains the algorithm, logic, and standard library/data science tools (e.g., NumPy, Pandas, Matplotlib) used.
* **PEP-8 Compliant Code:** Produces clean, readable Python code with inline comments explaining key logic.
* **Key Lines Explanation:** Dedicated breakdown explaining why specific functions, data structures, or paradigms were selected.
* **Suggested Test Cases:** Provides 2–3 reproducible examples with expected inputs and outputs for students to verify in their own local environments.
* **Optional Constraints:** Accepts custom requirements (e.g., "handle empty lists", "use type hints", "no external libraries").

### 2. Python Code Debugging & Root Cause Analysis
* **Flexible Debugging Modes:** Supports code-only analysis (static inspection for syntax bugs, undefined variables, and logical pitfalls), code + error message/traceback, code + expected behavior, or all inputs combined.
* **Optional Diagnostics:** Both "Error Message or Buggy Output" and "Expected Behavior" are optional. Only the buggy Python code is required.
* **Scoped Code Extraction:** Isolates the corrected solution from Section 2, ensuring that any buggy code snippets quoted in Section 1 explanations are never confused with the fix.
* **Summary of Changes:** Bulleted list detailing what lines changed and why the change was required.
* **Verification Steps & Edge Cases:** Advises on testing edge cases (boundary values, empty sequences, unexpected data types).
* **Optional Expected Behavior:** Accepts student description of the intended outcome (bounded to 5,000 characters).

### 3. Provider Inspection & Safe Secret Handling
* **Sidebar Inspection:** Displays configured backend model, provider health, and masked API key (reveals only the final 4 characters).
* **Dynamic Secret Redaction:** Automatically scrubs actual configured API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and standard authorization patterns from error messages.
* **Transient 503 Retry Protection:** Automatically retries temporary server-side high-demand and HTTP 503 errors (delays: 2.0s, 4.0s) while immediately failing fast on non-transient client errors (400, 401, 403, 404, 429).
* **Safe Output Formatting:** If a model returns purely conversational text without markdown code blocks, it displays an informative notice instead of formatting conversational text inside a Python code block.

---

## Safety & Safe Execution Policy

* **No Automatic Code Execution:** AI-generated and user-submitted code is displayed strictly for review and pedagogical analysis. Code is **never executed automatically** on the host server or by the application backend.
* **Credential Protection:** Real API keys are loaded strictly from the local environment (`.env`) and are never committed to version control (`.gitignore` excludes `.env` and `*.env`). Keys are never displayed in full in the UI, logged, or printed.

---

## Technology Stack & Prerequisites

* **Python:** Python 3.10+ (tested on Python 3.14.6)
* **Streamlit (`>=1.40.0`):** Interactive web user interface
* **Google GenAI SDK (`google-genai>=2.0.0`):** Official synchronous client for Google Gemini
* **python-dotenv (`>=1.0.0`):** Local environment variable management
* **pytest (`>=8.0.0`):** Unit and integration testing framework
* **Operating System:** Windows PowerShell, macOS, or Linux

---

## Project Structure

```text
ai_coding_assistant/
├── .streamlit/
│   └── config.toml          # Streamlit UI configuration (dark navy & blue-accent theme)
├── app.py                   # Streamlit web application layout, UI workflows, and safe display
├── ai_service.py            # Central service layer, provider factory, and input validation routing
├── prompts.py               # Structured educational prompt templates and system instructions
├── utils.py                 # Configuration loader, API-key masking, input validators, and error sanitization
├── requirements.txt         # Project dependencies
├── .env.example             # Template for local environment variable configuration
├── .gitignore               # Excludes secrets (.env), virtual environments (.venv), and caches
├── check_live_gemini.py     # Controlled CLI verification script for live code generation
├── check_live_debug.py      # Controlled CLI verification script for live code debugging
├── test_ai_service.py       # Automated test suite (63 tests with mocks; no live API calls needed)
└── providers/
    ├── __init__.py          # Providers package initialization
    ├── base.py              # BaseLLMProvider abstract base class & ProviderResponse dataclass
    └── gemini_provider.py   # Google Gemini adapter with section parsing & transient retry logic
```

---

## Installation & Setup (Windows PowerShell)

### 1. Clone or Navigate to the Project Directory
```powershell
cd C:\Users\HP\OneDrive\Documents\ai_coding_assistant
```

### 2. Activate the Virtual Environment
If using the existing project virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If setting up a fresh environment: `python -m venv .venv` then activate).*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Environment Configuration

1. Copy `.env.example` to create your local `.env` file:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` in a text editor and configure your credentials:
   ```env
   # --- Google Gemini Configuration ---
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```
   * Get an API key at [Google AI Studio](https://aistudio.google.com/).
   * Supported model options include `gemini-2.5-flash`, `gemini-2.0-flash`, or `gemini-1.5-flash`.
   * **Never commit your `.env` file or share your private API key.**

---

## Launching the Web Application

Run the following command in your activated terminal:
```powershell
streamlit run app.py
```
Or directly using the virtual environment interpreter:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
The application will open in your default browser at:
```text
Local URL: http://localhost:8501
```

---

## Running Automated Tests

The test suite runs against mocked provider responses and does **not** require an active internet connection or a real API key:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Test Suite Coverage (63 Tests Total)
* **Provider Factory & Routing (4 tests):** Provider discovery, Gemini adapter creation, unsupported provider rejection, and deferred provider fallback.
* **Input Validation & Scenarios (6 tests):** Blank prompts, oversized prompts (>10,000 chars), missing code, optional error message combinations, and oversized inputs.
* **Gemini Generation & Debugging (10 tests):** Successful code generation, key lines explanation capture, debugging with root cause analysis, code-only debugging, expected-behavior-only debugging, missing API key handling, and empty model responses.
* **Security & Secret Masking (4 tests):** Partial key masking, dynamic redaction of configured environment keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`), and parameter redaction.
* **Retry Mechanism & Resilience (5 tests):** Transient 503 retry success, repeated 503 exhaustion (delays verified), broadened 503 detection (e.g. backend error, status `UNAVAILABLE`), and fast-fail on non-transient errors (400, 401, 403, 404, 429).
* **Code Extraction & Parsing (7 tests):** Prioritizing Section 2 code over Section 1 snippets, case-insensitive fences, multiple code blocks, missing code fences returning empty strings, and conversational response handling.
* **Prompt Construction (6 tests):** Generation and debugging prompt layouts with and without optional context and expected behavior.
* **Provider Status Inspection (3 tests):** Validating status dictionaries for configured keys, missing keys, and unsupported providers.
* **Error Sanitization Scenarios (5 tests):** Redacting 429 quota exhaustion, 404 model not found, network timeouts, and raw auth headers.
* **Heading & Formatting Variations (5 tests):** Two-hash (`##`) headers, non-numeric section titles, missing section fallbacks, and multiple code block isolation.
* **Static Review & Correct Code Handling (8 tests):** Distinguishing static review from execution, enforcing exact `"No error was identified. The code appears correct for the intended behavior."` response, handling ambiguous/unclear intended behavior, and end-to-end mocks.

---

## Current Provider Support

* **Google Gemini (Active & Working):** Fully operational adapter using the official `google-genai` SDK with retry resilience, custom section parsing, and input validation.
* **OpenAI & Anthropic Claude (Deferred / Phase 6):** Registered in provider selection for architecture demonstration. When selected, the factory routes to a safe placeholder adapter (`_DeferredProvider`) that displays an informative status message without crashing the application.

---

## Known Limitations

1. **Third-Party Upstream Deprecation Warning:**
   When running on Python 3.14+, the official `google-genai` package emits an internal warning:
   ```text
   DeprecationWarning: '_UnionGenericAlias' is deprecated and slated for removal in Python 3.17
   ```
   This originates inside `.venv\Lib\site-packages\google\genai\types.py` due to Python 3.14 typing updates. It does not affect application functionality.
2. **Automated UI Testing:**
   Streamlit frontend components in `app.py` are tested through backend service layers and static review; headless simulated browser tests (via `streamlit.testing.v1.AppTest`) are not currently implemented.
