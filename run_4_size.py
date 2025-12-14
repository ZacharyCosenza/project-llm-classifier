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
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if "LOCAL_RANK" in os.environ:
        DEVICE = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(DEVICE)
        dist.init_process_group(backend="gloo")
        USE_DDP = True
        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
    elif NUM_GPUS >= 1:
        DEVICE = torch.device("cuda")
        USE_DDP = False
        WORLD_SIZE = 1
        RANK = 0
    else:
        DEVICE = torch.device(
            "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        )
        USE_DDP = False
        WORLD_SIZE = 1
        RANK = 0
        
    if SMOKE_TEST:
        BATCH_SIZE = 1
    else:
        if USE_DDP:
            BATCH_SIZE = max(1, args.batch // WORLD_SIZE)
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
    
    # Only rank 0 creates directories
    if RANK == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    if USE_DDP:
        dist.barrier()  # Wait for rank 0 to create directories

    log_path = os.path.join(CKPT_DIR, "run.log")
    
    def log(msg, rank_specific=False):
        """Log message. If rank_specific=False, only rank 0 logs."""
        if rank_specific or RANK == 0:
            print(msg)
            if RANK == 0:  # Only rank 0 writes to file
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

    try:
        log(f"=== START RUN (Rank {RANK}/{WORLD_SIZE}) ===", rank_specific=True)
        log(f"device={DEVICE}, smoke={SMOKE_TEST}, world_size={WORLD_SIZE}")

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

        if USE_DDP:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            NUM_WORKERS = max(1, os.cpu_count() // WORLD_SIZE // 2)
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
            try:
                if os.path.exists(os.path.join(WEIGHTS_DIR, "pytorch_model.bin")):
                    log(f"Loading local base model from {WEIGHTS_DIR}", rank_specific=True)
                    model = model_class.from_pretrained(WEIGHTS_DIR)
                else:
                    log(f"Downloading pretrained base model...", rank_specific=True)
                    model = model_class.from_pretrained(BASE_MODEL)
                    # Only rank 0 should save to avoid race conditions
                    if not SMOKE_TEST and RANK == 0:
                        model.save_pretrained(WEIGHTS_DIR)
                        log("Saved base model.")
                
                num_params = sum(p.numel() for p in model.parameters())
                log(f"Base model loaded with {num_params} parameters", rank_specific=True)
                
                return model
            except Exception as e:
                log(f"ERROR loading base model: {e}", rank_specific=True)
                raise

        base_model = get_base_model(model_class)

        if USE_DDP:
            dist.barrier()
            log("All ranks passed barrier after model loading")

        model = ResponseScorer(base_model)

        # Remove pooler if exists
        if hasattr(model.base, 'pooler'):
            model.base.pooler = None

        model = model.to(DEVICE)

        num_params = sum(p.numel() for p in model.parameters())
        log(f"ResponseScorer has {num_params} parameters")

        if USE_DDP:
            model = DDP(model, device_ids=[DEVICE], find_unused_parameters=True)
        
        start_epoch = 1
        latest_ckpt = None

        if args.resume_timestamp is not None and RANK == 0:
            files = [f for f in os.listdir(CKPT_DIR) if f.startswith("epoch") and f.endswith(".pt")]
            if len(files) > 0:
                files.sort(key=lambda x: int(x.replace("epoch", "").replace(".pt", "")))
                latest_ckpt = files[-1]
                start_epoch = int(latest_ckpt.replace("epoch", "").replace(".pt", "")) + 1
                log(f"Resuming from {latest_ckpt}, starting at epoch {start_epoch}")
                
                # Load checkpoint
                checkpoint = torch.load(os.path.join(CKPT_DIR, latest_ckpt), map_location=DEVICE)
                if USE_DDP:
                    model.module.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
            else:
                log("Resume timestamp given, but no checkpoints found. Starting from scratch.")
        
        if USE_DDP:
            # Broadcast start_epoch to all ranks
            start_epoch_tensor = torch.tensor(start_epoch).to(DEVICE)
            dist.broadcast(start_epoch_tensor, src=0)
            start_epoch = start_epoch_tensor.item()
  
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
                
                # Only show progress bar on rank 0
                if RANK == 0:
                    train_pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
                else:
                    train_pbar = train_loader

                if USE_DDP:
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

                    running_loss += loss.item()

                    if SMOKE_TEST:
                        break
                    
                    # Update progress bar only on rank 0
                    if RANK == 0:
                        preds = logits.argmax(dim=-1)
                        acc = (preds == labels).float().mean().item()
                        train_pbar.set_postfix(loss=loss.item(), acc=acc)

                # Aggregate training loss across all GPUs
                if USE_DDP:
                    train_loss_tensor = torch.tensor(running_loss).to(DEVICE)
                    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                    running_loss = train_loss_tensor.item()
                
                avg_train = running_loss / (1 if SMOKE_TEST else len(train_loader) * WORLD_SIZE)

                # ----------------------------
                # Validation
                # ----------------------------
                model.eval()
                val_loss, correct, total = 0, 0, 0
                
                with torch.no_grad():
                    if RANK == 0:
                        val_pbar = tqdm(val_loader, desc=f"Val {epoch}", leave=False)
                    else:
                        val_pbar = val_loader
                        
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

                        if SMOKE_TEST:
                            break

                        if RANK == 0:
                            batch_acc = (preds == labels).float().mean().item()
                            val_pbar.set_postfix(loss=loss.item(), acc=batch_acc)

                # Aggregate metrics across all GPUs
                if USE_DDP:
                    val_loss_tensor = torch.tensor(val_loss).to(DEVICE)
                    correct_tensor = torch.tensor(correct).to(DEVICE)
                    total_tensor = torch.tensor(total).to(DEVICE)
                    
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
                    
                    val_loss = val_loss_tensor.item()
                    correct = correct_tensor.item()
                    total = total_tensor.item()

                avg_val = val_loss / (1 if SMOKE_TEST else len(val_loader) * WORLD_SIZE)
                val_acc = correct / total

                log(f"Epoch {epoch} done. Train={avg_train:.4f}, Val={avg_val:.4f}, Acc={val_acc:.4f}")

                if RANK == 0:
                    train_stats.append({"epoch": epoch, "loss": avg_train})
                    val_stats.append({"epoch": epoch, "loss": avg_val, "acc": val_acc})

                    if not SMOKE_TEST:
                        ckpt_path = os.path.join(CKPT_DIR, f"epoch{epoch}.pt")
                        # Save model.module for DDP, model for non-DDP
                        state_dict = model.module.state_dict() if USE_DDP else model.state_dict()
                        torch.save(state_dict, ckpt_path)
                        log(f"Saved checkpoint {ckpt_path}")

            if RANK == 0 and not SMOKE_TEST:
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
            if RANK == 0:
                test_pbar = tqdm(test_loader, desc="Test", leave=False)
            else:
                test_pbar = test_loader
                
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

                if SMOKE_TEST:
                    break

        # Aggregate test metrics across all GPUs
        if USE_DDP:
            test_loss_tensor = torch.tensor(test_loss).to(DEVICE)
            correct_tensor = torch.tensor(correct).to(DEVICE)
            total_tensor = torch.tensor(total).to(DEVICE)
            
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            
            test_loss = test_loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()

        avg_test_loss = test_loss / (1 if SMOKE_TEST else len(test_loader) * WORLD_SIZE)
        test_acc = correct / total
        
        log(f"Test done. Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}")

        if RANK == 0 and not SMOKE_TEST:
            # Save per-batch stats from rank 0 only
            pd.DataFrame(test_stats).to_csv(os.path.join(CKPT_DIR, "test.csv"), index=False)

        log("=== RUN COMPLETE ===")

    except Exception as e:
        err = traceback.format_exc()
        if RANK == 0:
            with open(log_path, "a") as f:
                f.write("\nERROR:\n" + err)
        print(f"[Rank {RANK}] ERROR: {err}")
        raise
    finally:
        if USE_DDP:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()