import streamlit as st
from analyzer import analyze_text
from article_extractor import extract_article_text

# -------------------------------
# PAGE CONFIG - collapsed sidebar
# -------------------------------
st.set_page_config(initial_sidebar_state="collapsed")

# -------------------------------
# RESULT DISPLAY FUNCTION
# -------------------------------
def display_results(results):

    color_map = {
        "ad hominem": "#d62728",
        "false dilemma": "#1f77b4",
        "ad populum": "#ff7f0e",
        "false causality": "#9467bd",
        "faulty generalization": "#2ca02c",
        "appeal to emotion": "#e377c2",
        "circular reasoning": "#8c564b",
        "equivocation": "#17becf",
        "fallacy of credibility": "#bcbd22",
        "fallacy of extension": "#7f7f7f",
        "fallacy of logic": "#aec7e8",
        "fallacy of relevance": "#ffbb78",
        "intentional": "#ff9896"
    }

    fallacy_count = sum(
        1 for r in results if r["label"] != "No strong fallacy detected"
    )

    total_units = len(results)

    if total_units > 0:
        fallacy_weight = sum(
            r["confidence"] for r in results if r["label"] != "No strong fallacy detected"
        )
        max_possible_weight = total_units
        score = round(((max_possible_weight - fallacy_weight) / max_possible_weight) * 100, 2)
        score = max(0, score)
    else:
        score = 100

    if score >= 90:
        logic_status = "High Quality Reasoning"
    elif score >= 70:
        logic_status = "Mostly Sound Reasoning"
    elif score >= 50:
        logic_status = "Questionable Reasoning"
    else:
        logic_status = "Poor Reasoning"

    st.markdown("## Reasoning Quality Score")
    st.metric("Reasoning Quality Score", f"{score}/100")
    st.info(f"Assessment: **{logic_status}**")

    st.markdown("---")

    detected_fallacies = [r["label"] for r in results if r["label"] != "No strong fallacy detected"]

    if detected_fallacies:
        st.markdown("## 🔎 Fallacies Detected")
        from collections import Counter
        fallacy_counts = Counter(detected_fallacies)
        for fallacy, count in fallacy_counts.most_common():
            color = color_map.get(fallacy.lower(), "#444444")
            st.markdown(
                f"<span style='color:{color}; font-weight:bold'>● {fallacy}</span> — found {count} time(s)",
                unsafe_allow_html=True
            )
        st.markdown("---")
    else:
        st.success("No fallacies detected in this text.")

    st.markdown("## 📄 Paragraph Analysis")

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

        if explanation:
            st.markdown(explanation)

        if lime_words:
            important_words = ", ".join([f'"{word}"' for word, _ in lime_words])
            st.markdown(f"Model focused on terms like {important_words}.")

        st.markdown("---")


# -------------------------------
# APP UI STARTS HERE
# -------------------------------
st.title("AI-Based Logical Fallacy Detector")
st.caption("Detect logical fallacies and evaluate reasoning quality in text or news articles.")

st.sidebar.title("Input Method")

mode = st.sidebar.radio(
    "Choose analysis type:",
    ("Analyze Text", "Analyze Article URL")
)

# TEXT MODE
if mode == "Analyze Text":

    user_input = st.text_area("Enter text to analyze:")

    if st.button("Analyze Text"):

        if user_input.strip() == "":
            st.warning("Please enter valid text.")
        else:
            with st.spinner("Analyzing for logical fallacies..."):
                results = analyze_text(user_input)
            display_results(results)

# URL MODE
elif mode == "Analyze Article URL":

    url_input = st.text_input("Enter article URL:")

    if st.button("Analyze Article"):

        if url_input.strip() == "":
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Fetching article..."):
                article_text = extract_article_text(url_input)

            if not article_text:
                st.error("Could not extract article text from this URL.")
            else:
                with st.spinner("Analyzing for logical fallacies..."):
                    results = analyze_text(article_text)
                display_results(results)