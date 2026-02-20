# explanation_generator.py

fallacy_explanations = {

    "ad hominem":
        "This argument attacks the person making the claim instead of addressing the actual argument.",

    "ad populum":
        "This argument claims something is true or correct because many people believe it.",

    "appeal to emotion":
        "This argument attempts to persuade by appealing to emotions rather than logical reasoning.",

    "circular reasoning":
        "This argument repeats the same claim as its justification instead of providing independent evidence.",

    "equivocation":
        "This argument uses ambiguous or misleading language, shifting the meaning of a term within the argument.",

    "fallacy of credibility":
        "This argument focuses on the credibility or authority of a person instead of addressing the actual reasoning.",

    "fallacy of extension":
        "This argument misrepresents or exaggerates someone’s position in order to make it easier to attack (also known as a straw man).",

    "fallacy of logic":
        "This argument contains flawed or invalid logical reasoning that does not properly support the conclusion.",

    "fallacy of relevance":
        "This argument introduces information that is irrelevant to the main issue in order to distract from the core argument.",

    "false causality":
        "This argument assumes that because one event follows another, the first event caused the second without sufficient evidence.",

    "false dilemma":
        "This argument presents only two options while ignoring other possible alternatives.",

    "faulty generalization":
        "This argument draws a broad conclusion based on insufficient or limited evidence.",

    "intentional":
        "This argument incorrectly assigns intent or motive to someone without sufficient evidence."
}


def generate_explanation(predicted_label):
    label = predicted_label.lower()

    return fallacy_explanations.get(
        label,
        "This argument contains flawed reasoning, but a detailed explanation is not available."
    )