import nltk
from utils import predict_fallacy
from explanation_generator import generate_explanation
from lime_explainer import explain_text


def generate_spans(sentences, max_span=3):
    spans = []

    n = len(sentences)

    for size in range(1, min(max_span, n) + 1):
        for i in range(n - size + 1):
            span = " ".join(sentences[i:i + size]).strip()
            spans.append((i, i + size - 1, span))

    return spans


def analyze_text(text):

    sentences = nltk.sent_tokenize(text)

    spans = generate_spans(sentences)

    detected = []

    for start, end, span_text in spans:

        label, confidence = predict_fallacy(span_text)

        if label == "No strong fallacy detected":
            continue

        explanation = generate_explanation(label)
        lime_words = explain_text(span_text, label)

        detected.append({
            "start": start,
            "end": end,
            "text": span_text,
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "lime_words": lime_words
        })

    # If nothing detected return paragraph as normal text
    if not detected:
        return [{
            "text": text,
            "label": "No strong fallacy detected",
            "confidence": None,
            "explanation": None,
            "lime_words": None
        }]

    # Sort by sentence order
    detected = sorted(detected, key=lambda x: x["start"])

    final_results = []
    used = set()

    for d in detected:

        overlap = False

        for i in range(d["start"], d["end"] + 1):
            if i in used:
                overlap = True
                break

        if not overlap:
            final_results.append(d)

            for i in range(d["start"], d["end"] + 1):
                used.add(i)

    return final_results