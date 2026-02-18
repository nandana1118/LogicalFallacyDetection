import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW

# =========================
# LOAD PROCESSED DATA
# =========================

train_encodings = torch.load("../data/processed/train_encodings.pt", weights_only=False)
dev_encodings = torch.load("../data/processed/dev_encodings.pt", weights_only=False)
test_encodings = torch.load("../data/processed/test_encodings.pt", weights_only=False)

train_labels = torch.load("../data/processed/train_labels.pt", weights_only=False)
dev_labels = torch.load("../data/processed/dev_labels.pt", weights_only=False)
test_labels = torch.load("../data/processed/test_labels.pt", weights_only=False)


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
# TRAINING LOOP
# =========================

epochs = 2

model.train()

for epoch in range(epochs):
    total_loss = 0

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
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("Average Training Loss:", avg_loss)

# =========================
# SAVE TRAINED MODEL
# =========================

torch.save(model.state_dict(), "trained_model/bert_fallacy_model.pt")
print("\nModel saved successfully.")
