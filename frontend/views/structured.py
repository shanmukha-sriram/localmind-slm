"""Page 3: Structured Output."""

import streamlit as st

from app.core.exceptions import LocalMindError
from app.inference.parameters import GenerationParams
from app.structured_output.generator import generate_structured
from app.structured_output.schemas import SAMPLES, SCHEMAS
from frontend.components.sidebar import installed_models


def render() -> None:
    st.title("🧩 Structured Output")
    models = installed_models()
    if not models:
        st.warning("No models available.")
        return

    c1, c2 = st.columns(2)
    model = c1.selectbox("Model", [m.name for m in models])
    key = c2.selectbox("Schema", list(SCHEMAS), format_func=lambda k: SCHEMAS[k].label)
    c3, c4 = st.columns(2)
    constrained = c3.toggle("Schema-constrained decoding", value=True,
                            help="Off = rely on the prompt alone, so you can watch retries happen.")
    temperature = c4.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    text = st.text_area("Input text", value=SAMPLES[key], height=140, key=f"text_{key}")
    with st.expander("JSON Schema (generated from the Pydantic model)"):
        st.json(SCHEMAS[key].model.model_json_schema())

    if not st.button("Generate", type="primary"):
        return
    params = GenerationParams(temperature=temperature, max_tokens=1024,
                              think=False if model.startswith("qwen3") else None)
    with st.spinner("Generating..."):
        try:
            result = generate_structured(model, key, text, params=params,
                                         constrained=constrained)
        except LocalMindError as exc:
            st.error(str(exc))
            return

    m1, m2, m3 = st.columns(3)
    m1.metric("Attempts", result.attempts)
    m2.metric("Valid", "yes" if result.ok else "no")
    m3.metric("Total wall time", f"{sum(s.wall_s for s in result.stats):.1f}s")
    for i, err in enumerate(result.errors, 1):
        st.warning(f"Attempt {i} failed validation: {err}")
    if result.ok:
        st.success("Output is valid JSON and matches the schema.")
        st.json(result.data.model_dump())
        st.info("Valid does not mean correct. Check the values yourself: schema validation "
                "cannot tell whether the category or summary is actually right.")
    else:
        st.error(f"Failed after {result.attempts} attempts.")
    with st.expander("Raw model output for each attempt"):
        for i, raw in enumerate(result.raw_outputs, 1):
            st.text(f"Attempt {i}:\n{raw}")