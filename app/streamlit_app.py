import streamlit as st
from analyzer import analyze_text

st.title("Explainable Logical Fallacy Detection")

user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter valid text.")
    else:
        results = analyze_text(user_input)

        for result in results:

            st.subheader("Prediction Result")

            st.write("**Input:**", result["text"])
            st.write("**Predicted Fallacy:**", result["label"])
            st.write("**Confidence:**", round(result["confidence"], 4))
            st.write("**Explanation:**", result["explanation"])

            st.write("**Important Words (LIME):**")
            if result["lime_words"]:
                for word, score in result["lime_words"]:
                    st.write(f"{word} (+{score:.4f})")
            else:
                st.write("No strong positive contributions found.")