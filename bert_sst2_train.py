import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import random

# ---------- ECE computation ----------
def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            bin_accuracy = accuracies[mask].float().mean()
            bin_confidence = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_confidence - bin_accuracy)
    return ece.item()

# ---------- Config ----------
model_name = "bert-base-uncased"
batch_size = 16
learning_rate = 2e-5
num_epochs = 3
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Set seed ----------
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------- Load Dataset ----------
dataset = load_dataset("glue", "sst2")

# ---------- Load Tokenizer ----------
tokenizer = BertTokenizer.from_pretrained(model_name)

# ---------- Tokenization ----------
def tokenize_function(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rename 'label' to 'labels' for model compatibility
tokenized_datasets = tokenized_datasets.map(lambda x: {"labels": x["label"]})
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---------- Dataloaders ----------
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size)

# ---------- Model ----------
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# ---------- Optimizer and LR Scheduler ----------
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# ---------- Training Loop ----------
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f"\n🟢 Epoch {epoch+1}/{num_epochs}")
    train_progress_bar = tqdm(train_dataloader, desc="Training")

    for batch in train_progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"✅ Average Training Loss: {avg_train_loss:.4f}")

    # ---------- Evaluation ----------
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu())
            all_labels.extend(batch["labels"].cpu().numpy())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels_tensor = torch.tensor(all_labels).to(all_probs.device)

    val_accuracy = accuracy_score(all_labels, all_preds)
    ece_score = compute_ece(all_probs, all_labels_tensor)

    print(f"📊 Validation Accuracy: {val_accuracy:.4f}")
    print(f"📉 ECE Score: {ece_score:.4f}")

# ---------- Save Model ----------
model.save_pretrained("./bert-sst2")
tokenizer.save_pretrained("./bert-sst2")
print("\n💾 Model saved to ./bert-sst2")
