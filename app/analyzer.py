import nltk
from utils import predict_fallacy
from explanation_generator import generate_explanation
from lime_explainer import explain_text

def analyze_text(text):
    sentences = nltk.sent_tokenize(text)

    results = []

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence == "":
            continue

        label, confidence = predict_fallacy(sentence)

        # If no strong fallacy detected, skip explanation and LIME
        if label == "No strong fallacy detected":
            explanation = None
            lime_words = None
        else:
            explanation = generate_explanation(label)
            lime_words = explain_text(sentence, label)

        results.append({
            "text": sentence,
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "lime_words": lime_words
        })

    return results