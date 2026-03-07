import nltk
import re

from utils import predict_fallacy
from explanation_generator import generate_explanation
from lime_explainer import explain_text


def clean_text(text):
    """
    Fix common formatting problems that break sentence tokenization.
    Example: 'economics.Either' -> 'economics. Either'
    """
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    return text


def analyze_text(text):

    text = clean_text(text)

    sentences = nltk.sent_tokenize(text)

    # Step 1: classify each sentence
    sentence_predictions = []

    for sentence in sentences:

        sentence = sentence.strip()

        if sentence == "":
            continue

        label, confidence = predict_fallacy(sentence)

        sentence_predictions.append({
            "text": sentence,
            "label": label,
            "confidence": confidence
        })

    # Step 2: merge consecutive sentences with same fallacy
    merged_results = []

    i = 0
    n = len(sentence_predictions)

    while i < n:

        current = sentence_predictions[i]

        label = current["label"]

        span_text = current["text"]

        confidence = current["confidence"]

        j = i + 1

        # merge consecutive sentences with same label
        while j < n and sentence_predictions[j]["label"] == label:

            span_text += " " + sentence_predictions[j]["text"]

            confidence = max(confidence, sentence_predictions[j]["confidence"])

            j += 1

        # generate explanation only if fallacy exists
        if label == "No strong fallacy detected":
            explanation = None
            lime_words = None
        else:
            explanation = generate_explanation(span_text, label)
            lime_words = explain_text(span_text, label)

        merged_results.append({
            "text": span_text,
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "lime_words": lime_words
        })

        i = j

    return merged_results