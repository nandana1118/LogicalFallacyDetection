import torch
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoModelForSequenceClassification

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=13
)

model.load_state_dict(torch.load("trained_model/bert_fallacy_model.pt", weights_only=False))
model.to(device)
model.eval()

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Load validation data
encodings = torch.load("../data/processed/dev_encodings.pt", weights_only=False)
labels = torch.load("../data/processed/dev_labels.pt", weights_only=False)

input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

y_true = labels.cpu().numpy()
y_pred = predictions.cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

print("\nEvaluation Metrics (Validation Set):")
print("--------------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))