import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, DistilBertModel

class ResponseDataset(Dataset):
    def __init__(self, df, tokenizer, MAX_LEN):
        self.df = df
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN

    def outcome_to_class(self, row):
        if row["winner_model_a"] == 1: return 2
        if row["winner_model_b"] == 1: return 0
        return 1

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_a = row["prompt"] + row["response_a"]
        full_b = row["prompt"] + row["response_b"]
        t_a = self.tokenizer(
            full_a, max_length=self.MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        t_b = self.tokenizer(
            full_b, max_length=self.MAX_LEN,
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

class ResponseScorer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

        # # Model specific parameters
        # if isinstance(self.base, DistilBertModel):
        #     hidden = self.base.config.dim
        # elif isinstance(self.base, BertModel):
        #     hidden = self.base.config.hidden_size
        # else:
        #     raise TypeError(f"Unsupported base model type: {type(self.base)}")
        hidden = self.base.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 3)
        )

    def encode(self, ids, mask):
        return self.base(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        h = torch.cat([self.encode(ids_a, mask_a),
                    self.encode(ids_b, mask_b)], dim=-1)
        return self.head(h)