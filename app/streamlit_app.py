import streamlit as st
from analyzer import analyze_text
from article_extractor import extract_article_text


# -------------------------------
# RESULT DISPLAY FUNCTION
# -------------------------------
def display_results(results):

    fallacy_count = sum(
        1 for r in results if r["label"] != "No strong fallacy detected"
    )

    total_units = len(results)

    if total_units > 0:
        score = int(((total_units - fallacy_count) / total_units) * 100)
    else:
        score = 100

    if score >= 90:
        logic_status = "Highly Logical"
    elif score >= 70:
        logic_status = "Mostly Logical"
    elif score >= 50:
        logic_status = "Questionable Reasoning"
    else:
        logic_status = "Logically Weak"

    st.markdown("## 🧠 Logical Health Score")
    st.metric("Logic Score", f"{score}/100")
    st.info(f"Assessment: **{logic_status}**")

    st.markdown("---")
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

        if explanation:
            st.markdown(explanation)

        if lime_words:
            important_words = ", ".join([f'"{word}"' for word, _ in lime_words])
            st.markdown(f"Model focused on terms like {important_words}.")

        st.markdown("---")


# -------------------------------
# APP UI STARTS HERE
# -------------------------------
st.title("LogicLens: AI Fallacy Analyzer")
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
            results = analyze_text(user_input)
            display_results(results)

# URL MODE
elif mode == "Analyze Article URL":

    url_input = st.text_input("Enter article URL:")

    if st.button("Analyze Article"):

        if url_input.strip() == "":
            st.warning("Please enter a valid URL.")
        else:

            with st.spinner("Extracting article text..."):
                article_text = extract_article_text(url_input)

            if not article_text:
                st.error("Could not extract article text from this URL.")
            else:
                results = analyze_text(article_text)
                display_results(results)