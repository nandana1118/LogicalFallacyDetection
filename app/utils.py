import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "trained_model", "bert_fallacy_model.pt")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "model", "label_encoder.pkl")

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    local_files_only=True
)

# =========================
# LOAD MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=13,
    local_files_only=True
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.to(device)
model.eval()

# =========================
# LOAD LABEL ENCODER
# =========================

label_encoder = joblib.load(ENCODER_PATH)

# =========================
# PREDICTION FUNCTION
# =========================

def predict_fallacy(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    # Convert probabilities to numpy array
    probabilities = probs.detach().cpu().numpy()[0]

    # Sort probabilities in descending order
    sorted_indices = probabilities.argsort()[::-1]

    top_idx = sorted_indices[0]
    second_idx = sorted_indices[1]

    top_prob = probabilities[top_idx]
    second_prob = probabilities[second_idx]

    margin = top_prob - second_prob

    # UPDATED Decision rule (slightly less strict)
    if top_prob < 0.50 or margin < 0.15:
        return "No strong fallacy detected", float(top_prob)

    predicted_label = label_encoder.inverse_transform([top_idx])[0]

    return predicted_label, float(top_prob)