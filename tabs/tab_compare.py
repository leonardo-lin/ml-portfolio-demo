"""
Tab 2: Model Comparison
Side-by-side: Base model vs Fine-tuned model responses.
"""

import streamlit as st

try:
    from core.model_manager import ModelManager
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERROR = str(_e)


def render(model_manager=None):
    st.header("Model Comparison")
    if not _DEPS_OK:
        st.error(f"Missing dependency: `{_DEPS_ERROR}`")
        st.info("Please install PyTorch and transformers first (see QLoRA Training tab).")
        return
    st.markdown("Compare base model vs QLoRA fine-tuned responses side by side.")

    adapter_path = st.session_state.get("adapter_path")
    adapter_trained = st.session_state.get("training_done", False) or (adapter_path is not None)

    if not model_manager.is_loaded:
        st.warning("Load a model in the **QLoRA Training** tab first.")
        return

    is_demo = st.session_state.get("demo_mode", False)
    if not adapter_trained and not is_demo:
        st.info("Train and save an adapter in the **QLoRA Training** tab to unlock this comparison.")
        st.markdown("You can still test the base model below:")

    # ── Sample prompts ───────────────────────────────────────────────────────
    sample_prompts = [
        "What is QLoRA and why is it important for large language models?",
        "Explain the advantages of using 4-bit NF4 quantization.",
        "How does RAG reduce hallucination in language models?",
        "Describe the ReAct framework for AI agents.",
        "What is the difference between LoRA rank r=4 and r=16?",
    ]

    prompt_choice = st.selectbox("Sample Prompts", ["(custom)"] + sample_prompts)
    prompt = st.text_area(
        "Your Prompt",
        value=prompt_choice if prompt_choice != "(custom)" else "",
        height=100,
        placeholder="Enter your question here..."
    )

    max_tokens = st.slider("Max New Tokens", 50, 400, 200)

    if st.button("Generate Comparison", type="primary", disabled=not prompt.strip()):
        col_base, col_ft = st.columns(2)

        # -- Base model response --
        with col_base:
            st.subheader("Base Model")
            with st.spinner("Generating..."):
                try:
                    # Temporarily ensure no adapter is active
                    response_base, time_base = model_manager.generate(
                        prompt, max_new_tokens=max_tokens
                    )
                except Exception as e:
                    response_base = f"Error: {e}"
                    time_base = 0.0

            st.markdown(
                f'<div style="background:#1a1a2e;padding:1rem;border-radius:8px;'
                f'border-left:3px solid #a8a8a8;min-height:150px">{response_base}</div>',
                unsafe_allow_html=True,
            )
            tokens_base = len(response_base.split())
            st.caption(f"Time: {time_base:.1f}s  |  Tokens: ~{tokens_base}")

        # -- Fine-tuned response --
        with col_ft:
            st.subheader("Fine-tuned Model")
            can_ft = (adapter_trained and adapter_path) or st.session_state.get("demo_mode", False)
            if can_ft:
                with st.spinner("Loading adapter & generating..."):
                    try:
                        model_manager.load_with_adapter(adapter_path or "demo")
                        response_ft, time_ft = model_manager.generate(
                            prompt, max_new_tokens=max_tokens
                        )
                    except Exception as e:
                        response_ft = f"Error: {e}"
                        time_ft = 0.0
                st.markdown(
                    f'<div style="background:#1a1a2e;padding:1rem;border-radius:8px;'
                    f'border-left:3px solid #00d4ff;min-height:150px">{response_ft}</div>',
                    unsafe_allow_html=True,
                )
                tokens_ft = len(response_ft.split())
                st.caption(f"Time: {time_ft:.1f}s  |  Tokens: ~{tokens_ft}")
            else:
                st.info("Train an adapter first to compare.")
                response_ft = ""
                tokens_ft = 0

        # -- Metrics row --
        if response_base and response_ft:
            st.divider()
            st.subheader("Quick Metrics")
            m1, m2, m3 = st.columns(3)

            tokens_b = len(response_base.split())
            tokens_f = len(response_ft.split())

            # Vocabulary diversity (type-token ratio)
            words_b = response_base.lower().split()
            words_f = response_ft.lower().split()
            ttr_b = len(set(words_b)) / max(len(words_b), 1)
            ttr_f = len(set(words_f)) / max(len(words_f), 1)
            ttr_delta = round((ttr_f - ttr_b) / max(ttr_b, 0.01) * 100, 1)

            # Avg word length
            avg_len_b = sum(len(w) for w in words_b) / max(len(words_b), 1)
            avg_len_f = sum(len(w) for w in words_f) / max(len(words_f), 1)

            with m1:
                st.metric("Response Length", f"{tokens_f} tokens",
                           delta=f"{tokens_f - tokens_b:+d} vs base")
            with m2:
                st.metric("Vocabulary Diversity", f"{ttr_f:.2f}",
                           delta=f"{ttr_delta:+.1f}%")
            with m3:
                st.metric("Avg Word Length", f"{avg_len_f:.1f} chars",
                           delta=f"{avg_len_f - avg_len_b:+.1f}")
