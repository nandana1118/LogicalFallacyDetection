import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
import numpy as np

# =========================
# LOAD PROCESSED DATA
# =========================

train_encodings = torch.load("../data/processed/train_encodings.pt", weights_only=False)
dev_encodings = torch.load("../data/processed/dev_encodings.pt", weights_only=False)

train_labels = torch.load("../data/processed/train_labels.pt", weights_only=False)
dev_labels = torch.load("../data/processed/dev_labels.pt", weights_only=False)

print("Processed data loaded successfully.")

# =========================
# DATASET CLASS
# =========================

class FallacyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FallacyDataset(train_encodings, train_labels)
dev_dataset = FallacyDataset(dev_encodings, dev_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16)

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=13
)

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# =========================
# TRAINING SETTINGS
# =========================

epochs = 4
best_f1 = 0

# =========================
# TRAINING LOOP
# =========================

for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)

    # =========================
    # VALIDATION
    # =========================

    model.eval()
    total_val_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_val_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(dev_loader)
    macro_f1 = f1_score(true_labels, predictions, average="macro")

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")

    # Save best model
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), "trained_model/bert_fallacy_model.pt")
        print("Best model saved.")

# =========================
# FINAL REPORT
# =========================

print("\nFinal Classification Report on Validation Set:")
print(classification_report(true_labels, predictions))