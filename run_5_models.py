import os, time, argparse, traceback
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import (BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer,
                          RobertaModel, RobertaTokenizer, ElectraModel, ElectraTokenizerFast,
                          DebertaModel, DebertaTokenizer, AlbertModel, AlbertTokenizer, AutoTokenizer)
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from core import ResponseDataset, ResponseScorer
from utils import create_logger, setup_device_and_distributed

MODEL_REGISTRY = {
    "bert-base-uncased": (BertModel, BertTokenizer),
    "distilbert-base-uncased": (DistilBertModel, DistilBertTokenizer),
    "roberta-base": (RobertaModel, RobertaTokenizer),
    "google/electra-base-discriminator": (ElectraModel, ElectraTokenizerFast),
    "microsoft/deberta-base": (DebertaModel, DebertaTokenizer),
    "albert-base-v2": (AlbertModel, AlbertTokenizer),
}

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume_timestamp", type=str, default=None)
    ap.add_argument("--base_model", type=str, default='distilbert-base-uncased')
    ap.add_argument("--tag", type=str, default='')
    return ap.parse_args()

def main():

    args = get_args()

    SMOKE_TEST, BASE_MODEL, TAG = args.smoke, args.base_model, args.tag
    DATA_PATH, WEIGHTS_DIR, ROOT_CKPT = 'data/train.csv', os.path.join('weights', BASE_MODEL), 'checkpoints'
    LR, MAX_LEN = 1e-5, 512
    
    if BASE_MODEL not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported base_model '{BASE_MODEL}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class, _ = MODEL_REGISTRY[BASE_MODEL]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    NUM_EPOCHS = 1 if SMOKE_TEST else args.epochs
    DEVICE, USE_DDP, WORLD_SIZE, RANK = setup_device_and_distributed()
    BATCH_SIZE = 1 if SMOKE_TEST else (max(1, args.batch // WORLD_SIZE) if USE_DDP else args.batch)
    if args.resume_timestamp is None:
        TIMESTAMP, CKPT_DIR = time.strftime("%Y%m%d_%H%M%S"), os.path.join(ROOT_CKPT, time.strftime("%Y%m%d_%H%M%S"), TAG)
    else:
        TIMESTAMP, CKPT_DIR = args.resume_timestamp, os.path.join(ROOT_CKPT, args.resume_timestamp)
        if not os.path.isdir(CKPT_DIR):
            raise ValueError(f"Checkpoint directory {CKPT_DIR} does not exist.")
    if RANK == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if USE_DDP:
        dist.barrier()
    log = create_logger(os.path.join(CKPT_DIR, "run.log"), RANK)

    try:
        log(f"=== START RUN (Rank {RANK}/{WORLD_SIZE}) ===", rank_specific=True)
        log(f"device={DEVICE}, smoke={SMOKE_TEST}, world_size={WORLD_SIZE}")
        log("Loading dataset...")
        
        df = pd.read_csv(DATA_PATH)
        df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
        
        train_dataset, val_dataset, test_dataset = ResponseDataset(df_train, tokenizer, MAX_LEN), ResponseDataset(df_val, tokenizer, MAX_LEN), ResponseDataset(df_test, tokenizer, MAX_LEN)
        
        if USE_DDP:
            train_sampler, val_sampler, test_sampler = DistributedSampler(train_dataset, shuffle=True), DistributedSampler(val_dataset, shuffle=False), DistributedSampler(test_dataset, shuffle=False)
            NUM_WORKERS = max(1, os.cpu_count() // WORLD_SIZE // 2)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, num_workers=NUM_WORKERS)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, num_workers=NUM_WORKERS)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, pin_memory=True, num_workers=NUM_WORKERS)
        else:
            train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        log("Dataset ready.")
        model = ResponseScorer.from_pretrained(model_class=model_class, base_model_name=BASE_MODEL, weights_dir=WEIGHTS_DIR,
                                               smoke_test=SMOKE_TEST, rank=RANK, log_fn=log)
        if USE_DDP:
            dist.barrier()
            log("All ranks passed barrier after model loading")
        model = model.to(DEVICE)
        
        log(f"ResponseScorer has {sum(p.numel() for p in model.parameters())} parameters")
        
        if USE_DDP:
            model = DDP(model, device_ids=[DEVICE], find_unused_parameters=True)
        
        start_epoch = 1
        
        if args.resume_timestamp is not None and RANK == 0:
            files = [f for f in os.listdir(CKPT_DIR) if f.startswith("epoch") and f.endswith(".pt")]
            if files:
                files.sort(key=lambda x: int(x.replace("epoch", "").replace(".pt", "")))
                latest_ckpt = files[-1]
                start_epoch = int(latest_ckpt.replace("epoch", "").replace(".pt", "")) + 1
                log(f"Resuming from {latest_ckpt}, starting at epoch {start_epoch}")
                checkpoint = torch.load(os.path.join(CKPT_DIR, latest_ckpt), map_location=DEVICE)
                (model.module if USE_DDP else model).load_state_dict(checkpoint)
            else:
                log("Resume timestamp given, but no checkpoints found. Starting from scratch.")
        if USE_DDP:
            start_epoch_tensor = torch.tensor(start_epoch).to(DEVICE)
            dist.broadcast(start_epoch_tensor, src=0)
            start_epoch = start_epoch_tensor.item()
        
        optimizer, loss_fn = torch.optim.AdamW(model.parameters(), lr=LR), nn.CrossEntropyLoss()
        
        log("Model constructed.")
        
        goto_test_only = start_epoch > NUM_EPOCHS
        if goto_test_only:
            log(f"start_epoch={start_epoch} > epochs={NUM_EPOCHS} — skipping training, running test only.")

        if not goto_test_only:
            train_stats, val_stats = [], []
            for epoch in range(start_epoch, NUM_EPOCHS + 1):
                log(f"Epoch {epoch} starting...")
                model.train()
                running_loss = 0
                train_pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False) if RANK == 0 else train_loader
                if USE_DDP:
                    train_sampler.set_epoch(epoch)

                for batch in train_pbar:
                    optimizer.zero_grad()
                    ids_a, mask_a = batch["input_ids_a"].to(DEVICE), batch["attention_mask_a"].to(DEVICE)
                    ids_b, mask_b = batch["input_ids_b"].to(DEVICE), batch["attention_mask_b"].to(DEVICE)
                    labels = batch["label"].to(DEVICE)
                    logits = model(ids_a, mask_a, ids_b, mask_b)
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if SMOKE_TEST:
                        break
                    if RANK == 0:
                        preds = logits.argmax(dim=-1)
                        train_pbar.set_postfix(loss=loss.item(), acc=(preds == labels).float().mean().item())

                if USE_DDP:
                    train_loss_tensor = torch.tensor(running_loss).to(DEVICE)
                    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                    running_loss = train_loss_tensor.item()
                avg_train = running_loss / (1 if SMOKE_TEST else len(train_loader) * WORLD_SIZE)

                model.eval()
                val_loss, correct, total = 0, 0, 0
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Val {epoch}", leave=False) if RANK == 0 else val_loader
                    for batch in val_pbar:
                        ids_a, mask_a = batch["input_ids_a"].to(DEVICE), batch["attention_mask_a"].to(DEVICE)
                        ids_b, mask_b = batch["input_ids_b"].to(DEVICE), batch["attention_mask_b"].to(DEVICE)
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
                            val_pbar.set_postfix(loss=loss.item(), acc=(preds == labels).float().mean().item())

                if USE_DDP:
                    tensors = [torch.tensor(x).to(DEVICE) for x in [val_loss, correct, total]]
                    for t in tensors:
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    val_loss, correct, total = [t.item() for t in tensors]
                avg_val, val_acc = val_loss / (1 if SMOKE_TEST else len(val_loader) * WORLD_SIZE), correct / total
                log(f"Epoch {epoch} done. Train={avg_train:.4f}, Val={avg_val:.4f}, Acc={val_acc:.4f}")

                if RANK == 0:
                    train_stats.append({"epoch": epoch, "loss": avg_train})
                    val_stats.append({"epoch": epoch, "loss": avg_val, "acc": val_acc})
                    if not SMOKE_TEST:
                        ckpt_path = os.path.join(CKPT_DIR, f"epoch{epoch}.pt")
                        torch.save((model.module if USE_DDP else model).state_dict(), ckpt_path)
                        log(f"Saved checkpoint {ckpt_path}")

            if RANK == 0 and not SMOKE_TEST:
                pd.DataFrame(train_stats).to_csv(os.path.join(CKPT_DIR, "train.csv"), index=False)
                pd.DataFrame(val_stats).to_csv(os.path.join(CKPT_DIR, "val.csv"), index=False)
                log("Saved training CSVs.")

        log("Testing...")
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Test", leave=False) if RANK == 0 else test_loader
            for batch in test_pbar:
                ids_a, mask_a = batch["input_ids_a"].to(DEVICE), batch["attention_mask_a"].to(DEVICE)
                ids_b, mask_b = batch["input_ids_b"].to(DEVICE), batch["attention_mask_b"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(ids_a, mask_a, ids_b, mask_b)
                loss = loss_fn(logits, labels)
                preds = logits.argmax(dim=-1)
                test_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                if SMOKE_TEST:
                    break

        if USE_DDP:
            tensors = [torch.tensor(x).to(DEVICE) for x in [test_loss, correct, total]]
            for t in tensors:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            test_loss, correct, total = [t.item() for t in tensors]
        avg_test_loss, test_acc = test_loss / (1 if SMOKE_TEST else len(test_loader) * WORLD_SIZE), correct / total
        log(f"Test done. Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}")
        log("=== RUN COMPLETE ===")

    except Exception as e:
        err = traceback.format_exc()
        if RANK == 0:
            with open(os.path.join(CKPT_DIR, "run.log"), "a") as f:
                f.write("\nERROR:\n" + err)
        print(f"[Rank {RANK}] ERROR: {err}")
        raise
    finally:
        if USE_DDP:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()