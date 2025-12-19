import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ResponseDataset(Dataset):
    def __init__(self, df, tokenizer, MAX_LEN):
        self.df, self.tokenizer, self.MAX_LEN = df, tokenizer, MAX_LEN
    def outcome_to_class(self, row):
        if row["winner_model_a"] == 1: return 2
        if row["winner_model_b"] == 1: return 0
        return 1
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        t_a = self.tokenizer(row["prompt"] + row["response_a"], max_length=self.MAX_LEN,
                             padding="max_length", truncation=True, return_tensors="pt")
        t_b = self.tokenizer(row["prompt"] + row["response_b"], max_length=self.MAX_LEN,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids_a": t_a["input_ids"].squeeze(0), "attention_mask_a": t_a["attention_mask"].squeeze(0),
                "input_ids_b": t_b["input_ids"].squeeze(0), "attention_mask_b": t_b["attention_mask"].squeeze(0),
                "label": torch.tensor(self.outcome_to_class(row), dtype=torch.long)}

class LLMBase(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        if hasattr(self.base, 'pooler'):
            self.base.pooler = None
    @classmethod
    def from_pretrained(cls, model_class, base_model_name, weights_dir, smoke_test=False, rank=0, log_fn=print):
        try:
            weights_path = os.path.join(weights_dir, "pytorch_model.bin")
            if os.path.exists(weights_path):
                log_fn(f"Loading local base model from {weights_dir}", rank_specific=True)
                base_model = model_class.from_pretrained(weights_dir)
            else:
                log_fn(f"Downloading pretrained base model...", rank_specific=True)
                base_model = model_class.from_pretrained(base_model_name)
                if not smoke_test and rank == 0:
                    base_model.save_pretrained(weights_dir)
                    log_fn("Saved base model.")
            num_params = sum(p.numel() for p in base_model.parameters())
            log_fn(f"Base model loaded with {num_params} parameters", rank_specific=True)
            return cls(base_model)
        except Exception as e:
            log_fn(f"ERROR loading base model: {e}", rank_specific=True)
            raise
    def encode(self, ids, mask):
        return self.base(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]

class ResponseScorer(LLMBase):
    def __init__(self, base):
        super().__init__(base)
        hidden = self.base.config.hidden_size
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden // 2, 3))
    def forward(self, ids_a, mask_a, ids_b, mask_b):
        h = torch.cat([self.encode(ids_a, mask_a), self.encode(ids_b, mask_b)], dim=-1)
        return self.head(h)