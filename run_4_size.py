import os
import time
import argparse
import traceback
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Internal imports
from core import ResponseDataset, ResponseScorer

# ------------------------------------------------------------
# Argparse
# ------------------------------------------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume_timestamp", type=str, default=None)
    ap.add_argument("--base_model", type=str, default='distilbert-base-uncased')
    ap.add_argument("--tag", type=str, default='')
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
    TAG = args.tag
    LR = 1e-5
    MAX_LEN = 512

    # Base LLM models
    BASE_MODEL = args.base_model
    if BASE_MODEL == 'distilbert-base-uncased':
        model_class = DistilBertModel
        tokenizer_class = DistilBertTokenizer
    elif BASE_MODEL == 'bert-base-uncased' or BASE_MODEL == 'bert-large-uncased':
        model_class = BertModel
        tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(BASE_MODEL)

    NUM_EPOCHS = 1 if SMOKE_TEST else args.epochs

    # Detect and allocate CPU/GPU/XPU and distributed GPU
    if torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 1:
        DEVICE = int(os.environ["LOCAL_RANK"])
        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        dist.init_process_group(backend, rank=DEVICE)
    else:
        DEVICE = torch.device(
            "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        )
            
    if SMOKE_TEST:
        BATCH_SIZE = 1
    else:
        if NUM_GPUS > 0:
            BATCH_SIZE = max(1, args.batch // NUM_GPUS)
        else:
            BATCH_SIZE = args.batch

    # determine checkpoint directory (new run or resume)
    if args.resume_timestamp is None:
        TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
        CKPT_DIR = os.path.join(ROOT_CKPT, TIMESTAMP, TAG)
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

        # split
        df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

        train_dataset = ResponseDataset(df_train, tokenizer, MAX_LEN)
        val_dataset = ResponseDataset(df_val, tokenizer, MAX_LEN)
        test_dataset = ResponseDataset(df_test, tokenizer, MAX_LEN)

        if NUM_GPUS > 0:
            train_sampler = DistributedSampler(train_dataset, shuffle=False)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            NUM_WORKERS = max(1, os.cpu_count() // 2)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                      sampler=train_sampler, pin_memory=True, num_workers=NUM_WORKERS)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                                    sampler=val_sampler, pin_memory=True, num_workers=NUM_WORKERS)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                     sampler=test_sampler, pin_memory=True, num_workers=NUM_WORKERS)
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        log("Dataset ready.")

        # ----------------------------
        # Model
        # ----------------------------
        def get_base_model(model_class):
            if os.path.exists(os.path.join(WEIGHTS_DIR, "pytorch_model.bin")):
                log("Loading local base model...")
                return model_class.from_pretrained(WEIGHTS_DIR)
            log("Downloading pretrained base model...")
            model = model_class.from_pretrained(BASE_MODEL)
            if not SMOKE_TEST:
                model.save_pretrained(WEIGHTS_DIR)
                log("Saved base model.")
            return model

        base_model = get_base_model(model_class)
        model = ResponseScorer(base_model)
        model = model.to(DEVICE)
        if NUM_GPUS > 0: 
            model = DDP(model, device_ids=[DEVICE])            
        
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

                if NUM_GPUS > 0:
                    train_sampler.set_epoch(epoch)

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

                if not SMOKE_TEST and (NUM_GPUS == 0 or DEVICE == 0):
                    ckpt_path = os.path.join(CKPT_DIR, f"epoch{epoch}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    log(f"Saved checkpoint {ckpt_path}")

            if not SMOKE_TEST and (NUM_GPUS == 0 or DEVICE == 0):
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

        if not SMOKE_TEST and (NUM_GPUS == 0 or DEVICE == 0):
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