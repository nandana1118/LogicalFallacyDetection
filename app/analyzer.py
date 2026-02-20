from utils import predict_fallacy
from explanation_generator import generate_explanation
from lime_explainer import explain_text


def analyze_text(text):
    """
    For now: single sentence analysis.
    Later: can be extended to paragraph-level.
    """

    label, confidence = predict_fallacy(text)
    explanation = generate_explanation(label)
    lime_words = explain_text(text, label)

    result = {
        "text": text,
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "lime_words": lime_words
    }

    return [result]   # returning list for future paragraph support