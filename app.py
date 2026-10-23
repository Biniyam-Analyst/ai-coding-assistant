"""
app.py - Streamlit Web Application for AI Coding Assistant
Academic Project: Generative AI: Code Generation & Debugging
Built for Python and Data Science students.
"""

import streamlit as st
import ai_service
import utils

# -----------------------------------------------------------------------------
# Page Configuration & Visual Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Coding Assistant | Generative AI",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for Modern Dark Navy & Blue Accent Theme
st.markdown(
    """
    <style>
    /* Global Typography & Font Smoothing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Application Header */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }

    /* Academic Objective Banner */
    .academic-badge {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-left: 4px solid #3B82F6;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.75rem;
        font-size: 0.92rem;
        color: #CBD5E1;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .academic-badge strong {
        color: #60A5FA;
        font-weight: 600;
    }

    /* Streamlit Bordered Containers (Cards) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #151F32;
        border: 1px solid #26334D !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }

    /* Primary Action Buttons */
    button[kind="primary"] {
        background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%) !important;
        border: 1px solid #60A5FA !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: 0.015em !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
        border-color: #93C5FD !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-1px);
    }
    button[kind="primary"]:active {
        transform: translateY(0);
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.4) !important;
    }

    /* Text Inputs & Textareas */
    .stTextArea textarea, .stTextInput input {
        background-color: #0B1120 !important;
        border: 1px solid #26334D !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    /* Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #0B1120 !important;
        border: 1px solid #26334D !important;
        border-radius: 8px !important;
    }

    /* Code Display Panels */
    div[data-testid="stCodeBlock"] {
        border: 1px solid #2A3B56 !important;
        border-radius: 8px !important;
        background-color: #0A0F1D !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #26334D !important;
        border-radius: 8px !important;
        background-color: #151F32 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
    }

    /* Alert / Status Callouts */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border-width: 1px !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 3.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid rgba(51, 65, 85, 0.5);
        line-height: 1.6;
    }
    .footer-text em {
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Provider Selection & Environment Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/github/explore/main/topics/python/python.png",
        width=50,
    )
    st.title("Configuration")

    # Provider Selection
    available_providers = ai_service.get_available_providers()
    selected_provider = st.selectbox(
        "AI Provider",
        options=available_providers,
        index=0,
        help="Select the AI backend provider. Google Gemini is the active provider with installed SDK.",
    )

    # Provider Status Inspection
    status = utils.get_provider_status(selected_provider)
    st.markdown("---")

    with st.container(border=True):
        st.markdown("##### ⚙️ Provider Status")
        if status["configured"]:
            st.success(f"🟢 **{selected_provider}**: Ready")
            st.caption(f"**Model**: `{status['model']}`")
            st.caption(f"**Key**: `{status['masked_key']}`")
        else:
            st.warning(f"🟡 **{selected_provider}**: Key Missing")
            st.caption(f"**Model**: `{status['model']}`")
            st.info(status["setup_hint"])

    st.markdown("---")
    # Navigation Mode Selector
    st.subheader("Feature Mode")
    app_mode = st.radio(
        "Select Workflow",
        options=["⚡ Code Generation", "🔍 Code Debugging"],
        index=0,
    )

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">AI Coding Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Generative AI: Python Code Generation & Debugging for Students</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="academic-badge">
        <strong>Academic Learning Objective:</strong> Understand how modern Generative AI models
        assist in programming through natural-language code synthesis and automated error diagnosis,
        explaining the underlying Python mechanics for learner comprehension.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Workflow 1: Code Generation
# -----------------------------------------------------------------------------
if app_mode == "⚡ Code Generation":
    st.header("⚡ Natural Language to Python Code")
    st.markdown(
        "Describe what you want to calculate, automate, or visualize in Python. "
        "The AI assistant will formulate a solution with clear conceptual explanations."
    )

    # Example prompts helper
    with st.expander("💡 Click here to view example student requests"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "- *'Calculate the moving average of a stock price list without using external libraries.'*\n"
                "- *'Load a CSV file with Pandas and calculate summary statistics grouped by category.'*"
            )
        with col2:
            st.markdown(
                "- *'Write a function that counts word frequencies in a text file and returns the top 5.'*\n"
                "- *'Create a binary search function and explain its time complexity.'*"
            )

    with st.container(border=True):
        st.markdown("##### 📝 Prompt & Requirements")
        gen_prompt = st.text_area(
            "Programming Request / Prompt",
            height=130,
            placeholder="e.g., Write a function that accepts a list of exam scores and returns the mean, median, and standard deviation.",
            help="Be as specific as possible about input types and desired output.",
        )

        gen_context = st.text_input(
            "Optional Context or Constraints",
            placeholder="e.g., Use NumPy if possible; handle empty lists gracefully; include type hints.",
            help="Optional constraints such as libraries to use or special edge cases.",
        )

        generate_btn = st.button("Generate Python Code", type="primary", use_container_width=True)

    if generate_btn:
        if not gen_prompt.strip():
            st.warning("⚠️ Please enter a programming request before clicking Generate.")
        else:
            with st.spinner(f"Sending request to {selected_provider} ({status['model']})..."):
                result = ai_service.generate_code(
                    prompt=gen_prompt,
                    context=gen_context,
                    provider_name=selected_provider,
                )

            if result.success:
                st.success("✅ Code Generation Completed Successfully!")

                with st.container(border=True):
                    # 1. Approach & Explanation
                    if result.explanation:
                        st.subheader("1. Approach & Conceptual Explanation")
                        st.markdown(result.explanation)

                    # 2. Python Code
                    st.subheader("2. Generated Python Code")
                    if result.code and result.code.strip():
                        st.markdown("*Review the generated code below before copying into your Python environment:*")
                        st.code(result.code.strip(), language="python")
                    else:
                        st.info("ℹ️ No distinct Python code block was identified in the response.")

                    # 3. Explanation of Key Lines
                    key_lines = getattr(result, "key_lines_explanation", None)
                    if key_lines:
                        st.subheader("3. Explanation of Key Lines")
                        st.markdown(key_lines)

                    # 4. Verification & Testing
                    verification_steps = getattr(result, "verification_steps", None)
                    if verification_steps:
                        st.subheader("4. Suggested Test Cases & Next Steps")
                        st.markdown(verification_steps)

                    # Metadata
                    st.caption(f"Provider: {result.provider_name} | Model: `{result.model_name}`")
            else:
                st.error("❌ Generation Request Failed")
                with st.container(border=True):
                    st.markdown(f"> **Details:** {result.error_message}")
                    if "not configured" in (result.error_message or "").lower():
                        st.info("💡 Hint: Configure your API key in the local `.env` file as shown in `.env.example`.")

# -----------------------------------------------------------------------------
# Workflow 2: Code Debugging
# -----------------------------------------------------------------------------
elif app_mode == "🔍 Code Debugging":
    st.header("🔍 Python Code Debugger & Root Cause Explainer")
    st.markdown(
        "Paste your Python code along with an optional error traceback or observed wrong output. "
        "The assistant will perform a static code review to diagnose the root cause or verify correctness."
    )
    st.caption("ℹ️ *Static Code Analysis: Code is analyzed through static inspection of structure and logic without runtime execution.*")

    with st.expander("💡 Click here to view a sample debugging problem"):
        st.code(
            "# Sample Buggy Code:\n"
            "def calculate_average(numbers):\n"
            "    total = sum(numbers)\n"
            "    return total / len(numbers)\n\n"
            "print(calculate_average([]))\n\n"
            "# Sample Error:\n"
            "ZeroDivisionError: division by zero",
            language="python",
        )

    with st.container(border=True):
        st.markdown("##### 🐛 Buggy Code & Diagnostic Inputs")
        debug_code_input = st.text_area(
            "Buggy Python Code",
            height=160,
            placeholder="def my_function(data):\n    # Paste your code here\n    ...",
            help="Paste the Python source code where the issue occurs.",
        )

        debug_error_input = st.text_area(
            "Error Message or Buggy Output (Optional)",
            height=100,
            placeholder="Paste the error message or traceback if available; otherwise, leave blank.",
            help="Optional: Paste the Python traceback or describe the incorrect behavior.",
        )

        debug_expected_input = st.text_input(
            "Expected Behavior (Optional)",
            placeholder="e.g., Should return 0.0 or None when an empty list is passed instead of crashing.",
            help="Describe what the code was intended to do.",
        )

        debug_btn = st.button("Diagnose and Debug Code", type="primary", use_container_width=True)

    if debug_btn:
        if not debug_code_input.strip():
            st.warning("⚠️ Please paste the Python code that needs debugging.")
        else:
            with st.spinner(f"Analyzing code with {selected_provider} ({status['model']})..."):
                result = ai_service.debug_code(
                    code=debug_code_input,
                    error_message=debug_error_input.strip() or None,
                    expected_behavior=debug_expected_input.strip() or None,
                    provider_name=selected_provider,
                )

            if result.success:
                explanation_lower = (result.explanation or "").lower()
                is_no_error = (
                    "no error was identified" in explanation_lower
                    or "code appears correct" in explanation_lower
                )
                is_unclear = (
                    "intended behavior is unclear" in explanation_lower
                    or "behavior is unclear" in explanation_lower
                ) and not (result.code and result.code.strip())

                if is_no_error:
                    st.success("✅ Analysis Completed: No Error Identified")
                    with st.container(border=True):
                        st.info(
                            "**No error was identified. The code appears correct for the intended behavior.**\n\n"
                            "ℹ️ *Static Code Review Notice:* This assessment is based on static analysis of syntax, "
                            "structure, and evident logic. Static review does not execute the code and is not proof "
                            "that the code has no possible runtime errors under untested input conditions or external dependencies."
                        )

                        # 1. Diagnostic Assessment
                        if result.explanation:
                            st.subheader("1. Diagnostic Assessment")
                            st.markdown(result.explanation)

                        # 2. Verified Python Code (only display if verified code was provided; never display empty panel or confusing warning)
                        if result.code and result.code.strip():
                            st.subheader("2. Verified Python Code")
                            st.markdown("*The submitted Python code was verified as correct for the intended behavior:*")
                            st.code(result.code.strip(), language="python")

                        # 3. Summary of Changes
                        change_summary = getattr(result, "change_summary", None)
                        if change_summary:
                            st.subheader("3. Summary of Changes")
                            st.markdown(change_summary)
                        else:
                            st.subheader("3. Summary of Changes")
                            st.markdown("*No modifications were necessary; original code meets the stated requirements.*")

                        # 4. Verification Steps
                        verification_steps = getattr(result, "verification_steps", None)
                        if verification_steps:
                            st.subheader("4. Verification Steps & Edge Cases")
                            st.markdown(verification_steps)

                        # Metadata
                        st.caption(f"Provider: {result.provider_name} | Model: `{result.model_name}`")
                else:
                    st.success("✅ Diagnosis and Correction Completed!")
                    with st.container(border=True):
                        st.caption(
                            "ℹ️ *Static Code Review:* Code is statically inspected without runtime execution."
                        )

                        # 1. Root Cause Analysis
                        if result.explanation:
                            st.subheader("1. Root Cause Analysis")
                            st.markdown(result.explanation)

                        # 2. Corrected Code
                        st.subheader("2. Corrected Python Code")
                        if result.code and result.code.strip():
                            st.markdown("*Review the corrected code below before copying into your Python environment:*")
                            st.code(result.code.strip(), language="python")
                        elif is_unclear:
                            st.info("ℹ️ No code corrections proposed pending clarification of intended behavior.")
                        else:
                            st.info("ℹ️ No distinct Python code block was identified in the response.")

                        # 3. Summary of Changes
                        change_summary = getattr(result, "change_summary", None)
                        if change_summary:
                            st.subheader("3. Summary of Changes Made")
                            st.markdown(change_summary)

                        # 4. Verification Steps
                        verification_steps = getattr(result, "verification_steps", None)
                        if verification_steps:
                            st.subheader("4. Verification Steps & Edge Cases")
                            st.markdown(verification_steps)

                        # Metadata
                        st.caption(f"Provider: {result.provider_name} | Model: `{result.model_name}`")
            else:
                st.error("❌ Debugging Request Failed")
                with st.container(border=True):
                    st.markdown(f"> **Details:** {result.error_message}")
                    if "not configured" in (result.error_message or "").lower():
                        st.info("💡 Hint: Configure your API key in the local `.env` file as shown in `.env.example`.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-text">
        AI Coding Assistant | Generative AI: Code Generation & Debugging | Python, Streamlit & Google Gen AI SDK<br>
        <em>Academic Demonstration Project — Generated code must always be reviewed by a human programmer.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
