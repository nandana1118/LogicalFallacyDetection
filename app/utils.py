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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# =========================
# LOAD MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=13
)

model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
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

    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    label = label_encoder.inverse_transform([predicted_class])[0]

    return label, confidence
