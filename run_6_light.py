import os, time, argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import (BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer,
                          RobertaModel, RobertaTokenizer, ElectraModel, ElectraTokenizerFast,
                          DebertaModel, DebertaTokenizer, AlbertModel, AlbertTokenizer, AutoTokenizer)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from core import ResponseDataset, ResponseScorer
from utils import create_logger

MODEL_REGISTRY = {
    "bert-base-uncased": (BertModel, BertTokenizer),
    "distilbert-base-uncased": (DistilBertModel, DistilBertTokenizer),
    "roberta-base": (RobertaModel, RobertaTokenizer),
    "google/electra-base-discriminator": (ElectraModel, ElectraTokenizerFast),
    "microsoft/deberta-base": (DebertaModel, DebertaTokenizer),
    "albert-base-v2": (AlbertModel, AlbertTokenizer),
}

class LightningResponseScorer(pl.LightningModule):
    def __init__(self, model_class, base_model_name, weights_dir, lr=1e-5, smoke_test=False):
        super().__init__()
        self.save_hyperparameters(ignore=['model_class'])
        self.lr = lr
        self.smoke_test = smoke_test

        # Create a simple logging function for model initialization
        def log_fn(msg, rank_specific=False):
            print(msg)

        # Initialize the ResponseScorer model
        self.model = ResponseScorer.from_pretrained(
            model_class=model_class,
            base_model_name=base_model_name,
            weights_dir=weights_dir,
            smoke_test=smoke_test,
            rank=self.global_rank if hasattr(self, 'global_rank') else 0,
            log_fn=log_fn
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        return self.model(ids_a, mask_a, ids_b, mask_b)

    def training_step(self, batch, batch_idx):
        ids_a = batch["input_ids_a"]
        mask_a = batch["attention_mask_a"]
        ids_b = batch["input_ids_b"]
        mask_b = batch["attention_mask_b"]
        labels = batch["label"]

        logits = self(ids_a, mask_a, ids_b, mask_b)
        loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ids_a = batch["input_ids_a"]
        mask_a = batch["attention_mask_a"]
        ids_b = batch["input_ids_b"]
        mask_b = batch["attention_mask_b"]
        labels = batch["label"]

        logits = self(ids_a, mask_a, ids_b, mask_b)
        loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        ids_a = batch["input_ids_a"]
        mask_a = batch["attention_mask_a"]
        ids_b = batch["input_ids_b"]
        mask_b = batch["attention_mask_b"]
        labels = batch["label"]

        logits = self(ids_a, mask_a, ids_b, mask_b)
        loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        self.log('test_acc', acc, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume_timestamp", type=str, default=None)
    ap.add_argument("--base_model", type=str, default='distilbert-base-uncased')
    ap.add_argument("--tag", type=str, default='')
    ap.add_argument("--devices", type=int, default=None, help="Number of devices to use (GPUs or CPUs)")
    ap.add_argument("--strategy", type=str, default="auto", help="Training strategy (auto, ddp, ddp_spawn, etc.)")
    ap.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    return ap.parse_args()


def main():
    args = get_args()

    # Set matmul precision for better performance on modern GPUs
    torch.set_float32_matmul_precision('medium')

    # Helper to print only from rank 0
    def rank_zero_print(*args_print, **kwargs):
        # Check if we're in distributed mode
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            if local_rank == 0:
                print(*args_print, **kwargs)
        else:
            print(*args_print, **kwargs)

    SMOKE_TEST, BASE_MODEL, TAG = args.smoke, args.base_model, args.tag
    DATA_PATH = 'data/train.csv'
    WEIGHTS_DIR = os.path.join('weights', BASE_MODEL)
    ROOT_CKPT = 'checkpoints'
    LR, MAX_LEN = 1e-5, 512

    if BASE_MODEL not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported base_model '{BASE_MODEL}'. Available: {list(MODEL_REGISTRY.keys())}")

    model_class, _ = MODEL_REGISTRY[BASE_MODEL]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    NUM_EPOCHS = 1 if SMOKE_TEST else args.epochs
    BATCH_SIZE = 1 if SMOKE_TEST else args.batch

    # Setup checkpoint directory
    if args.resume_timestamp is None:
        TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
        CKPT_DIR = os.path.join(ROOT_CKPT, TIMESTAMP, TAG)
    else:
        TIMESTAMP = args.resume_timestamp
        CKPT_DIR = os.path.join(ROOT_CKPT, args.resume_timestamp, TAG)
        if not os.path.isdir(os.path.join(ROOT_CKPT, args.resume_timestamp)):
            raise ValueError(f"Checkpoint directory {os.path.join(ROOT_CKPT, args.resume_timestamp)} does not exist.")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    rank_zero_print(f"=== START LIGHTNING RUN ===")
    rank_zero_print(f"smoke={SMOKE_TEST}, epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}")
    rank_zero_print("Loading dataset...")

    # Load and split dataset
    df = pd.read_csv(DATA_PATH)
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = ResponseDataset(df_train, tokenizer, MAX_LEN)
    val_dataset = ResponseDataset(df_val, tokenizer, MAX_LEN)
    test_dataset = ResponseDataset(df_test, tokenizer, MAX_LEN)

    # Determine number of workers - use 0 for CPU to avoid multiprocessing issues
    num_workers = 0 if not torch.cuda.is_available() else args.num_workers

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    rank_zero_print("Dataset ready.")

    # Initialize model
    model = LightningResponseScorer(
        model_class=model_class,
        base_model_name=BASE_MODEL,
        weights_dir=WEIGHTS_DIR,
        lr=LR,
        smoke_test=SMOKE_TEST
    )

    rank_zero_print(f"ResponseScorer has {sum(p.numel() for p in model.parameters())} parameters")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename='epoch{epoch:02d}',
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,
        verbose=True
    )

    # Setup logger
    csv_logger = CSVLogger(save_dir=CKPT_DIR, name='lightning_logs')

    # Determine checkpoint path for resuming
    ckpt_path = None
    if args.resume_timestamp is not None:
        ckpt_files = [f for f in os.listdir(CKPT_DIR) if f.startswith("epoch") and f.endswith(".ckpt")]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(x.replace("epoch", "").replace(".ckpt", "")))
            latest_ckpt = ckpt_files[-1]
            ckpt_path = os.path.join(CKPT_DIR, latest_ckpt)
            rank_zero_print(f"Resuming from {latest_ckpt}")
        else:
            rank_zero_print("Resume timestamp given, but no checkpoints found. Starting from scratch.")

    # Determine accelerator and devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.devices if args.devices is not None else torch.cuda.device_count()
        rank_zero_print(f"Using GPU acceleration with {devices} device(s)")
    else:
        accelerator = "cpu"
        devices = 1  # CPU always uses 1 device in Lightning
        rank_zero_print("Using CPU")

    # Adjust strategy for single device
    strategy = args.strategy
    if devices == 1:
        strategy = "auto"  # Single device doesn't need DDP

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        fast_dev_run=SMOKE_TEST
    )

    # Train
    rank_zero_print("Starting training...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Test
    rank_zero_print("Testing...")
    trainer.test(model, test_loader)

    rank_zero_print("=== RUN COMPLETE ===")


if __name__ == "__main__":
    main()
