from app.utils import predict_fallacy
from app.explanation_generator import generate_explanation
from app.lime_explainer import explain_text


while True:
    text = input("\nEnter argument (type exit to stop): ").strip()

    if text.lower() == "exit":
        break

    if text == "":
        print("Please enter a valid argument.")
        continue

    label, confidence = predict_fallacy(text)
    explanation = generate_explanation(label)
    lime_words = explain_text(text, label)

    print("\nPredicted:", label)
    print("Confidence:", round(confidence, 4))
    print("Explanation:", explanation)

    print("\nLIME Important Words (for predicted class):")
    if lime_words:
        for word, score in lime_words:
            print(f"{word} (+{score:.4f})")
    else:
        print("No strong positive word contributions found.")