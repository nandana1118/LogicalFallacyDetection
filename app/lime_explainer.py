import numpy as np
from lime.lime_text import LimeTextExplainer
from app.utils import model, tokenizer, label_encoder
import torch

class_names = list(label_encoder.classes_)
explainer = LimeTextExplainer(class_names=class_names)


def predict_proba(texts):
    all_probs = []

    for text in texts:
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        all_probs.append(probs.cpu().numpy()[0])

    return np.array(all_probs)


def explain_text(text, predicted_label):
    predicted_class = list(label_encoder.classes_).index(predicted_label)

    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=6,
        num_samples=100,
        labels=[predicted_class]
    )

    word_scores = explanation.as_list(label=predicted_class)

    # Keep only positive contributions
    positive_words = [(word, score) for word, score in word_scores if score > 0]

    return positive_words