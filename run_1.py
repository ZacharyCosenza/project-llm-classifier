import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

# ----------------------------
# Config
# ----------------------------
SMOKE_TEST = True  # <--- Set True for a quick run, False for full training

DATA_PATH = 'data/train.csv'
WEIGHTS_DIR = "weights/distilbert"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 2 if not SMOKE_TEST else 1
NUM_EPOCHS = 3 if not SMOKE_TEST else 1
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu")
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------
# Data
# ----------------------------
df = pd.read_csv(DATA_PATH)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class ResponseDataset(Dataset):
    def __init__(self, df, max_length=128):
        self.df = df
        self.max_length = max_length

    def outcome_to_class(self, row):
        if row["winner_model_a"] == 1: return 2
        if row["winner_model_b"] == 1: return 0
        return 1

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens_a = tokenizer(row["response_a"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        tokens_b = tokenizer(row["response_b"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        label = torch.tensor(self.outcome_to_class(row), dtype=torch.long)
        return {
            "input_ids_a": tokens_a["input_ids"].squeeze(0),
            "attention_mask_a": tokens_a["attention_mask"].squeeze(0),
            "input_ids_b": tokens_b["input_ids"].squeeze(0),
            "attention_mask_b": tokens_b["attention_mask"].squeeze(0),
            "label": label
        }

df_train, df_temp = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)

train_loader = DataLoader(ResponseDataset(df_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ResponseDataset(df_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(ResponseDataset(df_test), batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# Model
# ----------------------------
def get_base_model(weights='pretrained'):
    if os.path.exists(os.path.join(WEIGHTS_DIR, "pytorch_model.bin")):
        return DistilBertModel.from_pretrained(WEIGHTS_DIR)
    elif weights == 'pretrained':
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if not SMOKE_TEST: model.save_pretrained(WEIGHTS_DIR)
        return model
    else:
        return DistilBertModel(DistilBertConfig())

base_model = get_base_model()

class ResponseScorer(nn.Module):
    def __init__(self, base_model, freeze_base=False):
        super().__init__()
        self.base = base_model
        hidden = self.base.config.dim
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden//2, 3)
        )
        if freeze_base:
            for p in self.base.parameters(): p.requires_grad = False

    def encode(self, ids, mask):
        return self.base(input_ids=ids, attention_mask=mask).last_hidden_state[:,0,:]

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        h = torch.cat([self.encode(ids_a, mask_a), self.encode(ids_b, mask_b)], dim=-1)
        return self.head(h)

model = ResponseScorer(base_model).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ----------------------------
# Training
# ----------------------------
train_stats, val_stats = [], []

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0
    for i, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()
        ids_a = batch["input_ids_a"].to(DEVICE)
        mask_a = batch["attention_mask_a"].to(DEVICE)
        ids_b = batch["input_ids_b"].to(DEVICE)
        mask_b = batch["attention_mask_b"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(ids_a, mask_a, ids_b, mask_b)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if SMOKE_TEST: break  # only one batch in smoke test

    avg_train_loss = running_loss / (1 if SMOKE_TEST else len(train_loader))

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader, 1):
            ids_a = batch["input_ids_a"].to(DEVICE)
            mask_a = batch["attention_mask_a"].to(DEVICE)
            ids_b = batch["input_ids_b"].to(DEVICE)
            mask_b = batch["attention_mask_b"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(ids_a, mask_a, ids_b, mask_b)
            val_loss += loss_fn(logits, labels).item()
            preds = logits.argmax(dim=-1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
            if SMOKE_TEST: break

    avg_val_loss = val_loss / (1 if SMOKE_TEST else len(val_loader))
    val_acc = correct / total
    train_stats.append({"epoch": epoch, "loss": avg_train_loss})
    val_stats.append({"epoch": epoch, "loss": avg_val_loss, "accuracy": val_acc})

    print(f"Epoch {epoch} | Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.4f}")

    # Save checkpoint
    if not SMOKE_TEST:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{TIMESTAMP}_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

# Save CSVs only if not smoke test
if not SMOKE_TEST:
    pd.DataFrame(train_stats).to_csv(f"train_stats_{TIMESTAMP}.csv", index=False)
    pd.DataFrame(val_stats).to_csv(f"val_stats_{TIMESTAMP}.csv", index=False)

# ----------------------------
# Test
# ----------------------------
model.eval()
test_loss, correct, total = 0, 0, 0
test_stats = []
with torch.no_grad():
    for i, batch in enumerate(test_loader, 1):
        ids_a = batch["input_ids_a"].to(DEVICE)
        mask_a = batch["attention_mask_a"].to(DEVICE)
        ids_b = batch["input_ids_b"].to(DEVICE)
        mask_b = batch["attention_mask_b"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(ids_a, mask_a, ids_b, mask_b)
        test_loss += loss_fn(logits, labels).item()
        preds = logits.argmax(dim=-1)
        correct += (preds==labels).sum().item()
        test_stats.append({"batch_loss": loss_fn(logits, labels).item(), "batch_acc": (preds==labels).float().mean().item()})
        if SMOKE_TEST: break

avg_test_loss = test_loss / (1 if SMOKE_TEST else len(test_loader))
test_acc = correct / (1 if SMOKE_TEST else total)
print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")

if not SMOKE_TEST:
    pd.DataFrame(test_stats).to_csv(f"test_stats_{TIMESTAMP}.csv", index=False)
