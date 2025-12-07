import os
import time
import argparse
import traceback
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ------------------------------------------------------------
# Argparse
# ------------------------------------------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--resume_timestamp", type=str, default=None)
    ap.add_argument("--base_model", type=str, default='basebert')
    return ap.parse_args()


# ------------------------------------------------------------
# Script
# ------------------------------------------------------------
def main():
    args = get_args()

    SMOKE_TEST = args.smoke
    DATA_PATH = 'data/train.csv'
    WEIGHTS_DIR = os.path.join('weights', args.base_model)
    ROOT_CKPT = 'checkpoints'
    LR = 1e-5
    MAX_LEN = 512
    BASE_MODEL = args.base_model

    # Create model labels and parameters
    if BASE_MODEL == 'distilbert':
        tokenizer_label = "distilbert-base-uncased"
        model_label = "distilbert-base-uncased"
    elif BASE_MODEL == 'basebert':
        tokenizer_label = "bert-base-uncased"
        model_label = "bert-base-uncased"

    BATCH_SIZE = 1 if SMOKE_TEST else args.batch
    NUM_EPOCHS = 1 if SMOKE_TEST else args.epochs

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    )

    # determine checkpoint directory (new run or resume)
    if args.resume_timestamp is None:
        TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
        CKPT_DIR = os.path.join(ROOT_CKPT, TIMESTAMP)
    else:
        TIMESTAMP = args.resume_timestamp
        CKPT_DIR = os.path.join(ROOT_CKPT, TIMESTAMP)
        if not os.path.isdir(CKPT_DIR):
            raise ValueError(f"Checkpoint directory {CKPT_DIR} does not exist.")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    log_path = os.path.join(CKPT_DIR, "run.log")
    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    try:
        log("=== START RUN ===")
        log(f"device={DEVICE}, smoke={SMOKE_TEST}")

        # ----------------------------
        # Load data
        # ----------------------------
        log("Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_label)

        class ResponseDataset(Dataset):
            def __init__(self, df):
                self.df = df

            def outcome_to_class(self, row):
                if row["winner_model_a"] == 1: return 2
                if row["winner_model_b"] == 1: return 0
                return 1

            def __len__(self): return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                full_a = row["prompt"] + row["response_a"]
                full_b = row["prompt"] + row["response_b"]
                t_a = tokenizer(
                    full_a, max_length=MAX_LEN,
                    padding="max_length", truncation=True, return_tensors="pt"
                )
                t_b = tokenizer(
                    full_b, max_length=MAX_LEN,
                    padding="max_length", truncation=True, return_tensors="pt"
                )
                label = torch.tensor(self.outcome_to_class(row), dtype=torch.long)
                return {
                    "input_ids_a": t_a["input_ids"].squeeze(0),
                    "attention_mask_a": t_a["attention_mask"].squeeze(0),
                    "input_ids_b": t_b["input_ids"].squeeze(0),
                    "attention_mask_b": t_b["attention_mask"].squeeze(0),
                    "label": label
                }

        # split
        df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

        train_loader = DataLoader(ResponseDataset(df_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(ResponseDataset(df_val), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(ResponseDataset(df_test), batch_size=BATCH_SIZE, shuffle=False)

        log("Dataset ready.")

        # ----------------------------
        # Model
        # ----------------------------
        def get_base_model():
            if os.path.exists(os.path.join(WEIGHTS_DIR, "pytorch_model.bin")):
                log("Loading local base model...")
                return BertModel.from_pretrained(WEIGHTS_DIR)
            log("Downloading pretrained base model...")
            model = BertModel.from_pretrained(model_label)
            if not SMOKE_TEST:
                model.save_pretrained(WEIGHTS_DIR)
                log("Saved base model.")
            return model

        base_model = get_base_model()

        class ResponseScorer(nn.Module):
            def __init__(self, base, freeze_base = True):
                super().__init__()
                self.base = base
                hidden = base.config.hidden_size
                self.head = nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden // 2, 3)
                )
                
                if freeze_base:
                    for p in self.base.parameters(): p.requires_grad = False

            def encode(self, ids, mask):
                return self.base(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]

            def forward(self, ids_a, mask_a, ids_b, mask_b):
                h = torch.cat([self.encode(ids_a, mask_a),
                               self.encode(ids_b, mask_b)], dim=-1)
                return self.head(h)

        model = ResponseScorer(base_model).to(DEVICE)
        
        start_epoch = 1
        latest_ckpt = None

        if args.resume_timestamp is not None:
            files = [f for f in os.listdir(CKPT_DIR) if f.startswith("epoch") and f.endswith(".pt")]
            if len(files) > 0:
                # sort by epoch number
                files.sort(key=lambda x: int(x.replace("epoch", "").replace(".pt", "")))
                latest_ckpt = files[-1]
                start_epoch = int(latest_ckpt.replace("epoch", "").replace(".pt", "")) + 1
                log(f"Resuming from {latest_ckpt}, starting at epoch {start_epoch}")
                model.load_state_dict(torch.load(os.path.join(CKPT_DIR, latest_ckpt), map_location=DEVICE))
            else:
                log("Resume timestamp given, but no checkpoints found. Starting from scratch.")
  
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        log("Model constructed.")

        if start_epoch > NUM_EPOCHS:
            log(f"start_epoch={start_epoch} > epochs={NUM_EPOCHS} — skipping training, running test only.")
            goto_test_only = True
        else:
            goto_test_only = False

        # ----------------------------
        # Training
        # ----------------------------
        if not goto_test_only:
            train_stats, val_stats = [], []

            for epoch in range(start_epoch, NUM_EPOCHS + 1):
                log(f"Epoch {epoch} starting...")
                model.train()
                running_loss = 0
                train_pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False)

                for batch in train_pbar:
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

                    # ---- per-batch metrics ----
                    preds = logits.argmax(dim=-1)
                    acc = (preds == labels).float().mean().item()
                    running_loss += loss.item()

                    if SMOKE_TEST:
                        break
                        
                    train_pbar.set_postfix(loss=loss.item(), acc=acc)

                avg_train = running_loss / (1 if SMOKE_TEST else len(train_loader))

                # validation
                model.eval()
                val_loss, correct, total = 0, 0, 0
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Val {epoch}", leave=False)
                    for batch in val_pbar:
                        ids_a = batch["input_ids_a"].to(DEVICE)
                        mask_a = batch["attention_mask_a"].to(DEVICE)
                        ids_b = batch["input_ids_b"].to(DEVICE)
                        mask_b = batch["attention_mask_b"].to(DEVICE)
                        labels = batch["label"].to(DEVICE)

                        logits = model(ids_a, mask_a, ids_b, mask_b)
                        loss = loss_fn(logits, labels)
                        preds = logits.argmax(dim=-1)

                        val_loss += loss.item()
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                        batch_acc = (preds == labels).float().mean().item()

                        if SMOKE_TEST:
                            break

                        val_pbar.set_postfix(loss=loss.item(), acc=batch_acc)

                avg_val = val_loss / (1 if SMOKE_TEST else len(val_loader))
                val_acc = correct / total

                log(f"Epoch {epoch} done. Train={avg_train:.4f}, Val={avg_val:.4f}, Acc={val_acc:.4f}")

                train_stats.append({"epoch": epoch, "loss": avg_train})
                val_stats.append({"epoch": epoch, "loss": avg_val, "acc": val_acc})

                if not SMOKE_TEST:
                    ckpt_path = os.path.join(CKPT_DIR, f"epoch{epoch}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    log(f"Saved checkpoint {ckpt_path}")

            if not SMOKE_TEST:
                pd.DataFrame(train_stats).to_csv(os.path.join(CKPT_DIR, "train.csv"), index=False)
                pd.DataFrame(val_stats).to_csv(os.path.join(CKPT_DIR, "val.csv"), index=False)
                log("Saved training CSVs.")

        # ----------------------------
        # Test
        # ----------------------------
        log("Testing...")
        model.eval()
        test_loss, correct, total = 0, 0, 0
        test_stats = []

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Test", leave=False)
            for batch in test_pbar:
                ids_a = batch["input_ids_a"].to(DEVICE)
                mask_a = batch["attention_mask_a"].to(DEVICE)
                ids_b = batch["input_ids_b"].to(DEVICE)
                mask_b = batch["attention_mask_b"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                logits = model(ids_a, mask_a, ids_b, mask_b)
                loss = loss_fn(logits, labels)
                preds = logits.argmax(dim=-1)

                test_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # ---- per-batch metrics ----
                acc = (preds == labels).float().mean().item()
                test_stats.append({"loss": loss.item(), "acc": acc})

                if SMOKE_TEST:
                    break

        avg_test_loss = test_loss / (1 if SMOKE_TEST else len(test_loader))
        test_acc = correct / (1 if SMOKE_TEST else total)
        log(f"Test done. Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}")

        if not SMOKE_TEST:
            pd.DataFrame(test_stats).to_csv(os.path.join(CKPT_DIR, "test.csv"), index=False)

        log("=== RUN COMPLETE ===")

    except Exception as e:
        err = traceback.format_exc()
        with open(log_path, "a") as f:
            f.write("\nERROR:\n" + err)
        print(err)
        raise


if __name__ == "__main__":
    main()
