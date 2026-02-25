import streamlit as st
from analyzer import analyze_text

st.title("Explainable Logical Fallacy Detection")

user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter valid text.")
    else:
        results = analyze_text(user_input)

        st.markdown("## 📄 Paragraph Analysis")

        color_map = {
            "ad hominem": "#d62728",
            "false dilemma": "#1f77b4",
            "ad populum": "#ff7f0e",
            "false causality": "#9467bd",
            "faulty generalization": "#2ca02c"
        }

        for result in results:
            sentence = result["text"]
            label = result["label"]
            confidence = result["confidence"]
            explanation = result["explanation"]
            lime_words = result["lime_words"]

            color = color_map.get(label.lower(), "#444444")

            st.markdown(f"**{sentence}**")

            if label == "No strong fallacy detected":
                st.markdown("No strong fallacy detected.")
            else:
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>{label}</span> "
                    f"({confidence*100:.1f}%)",
                    unsafe_allow_html=True
                )

            # Show explanation only if it exists
            if explanation:
                st.markdown(explanation)

            # Show LIME words only if they exist
            if lime_words:
                important_words = ", ".join([f'"{word}"' for word, _ in lime_words])
                st.markdown(f"Model focused on terms like {important_words}.")

            st.markdown("---")