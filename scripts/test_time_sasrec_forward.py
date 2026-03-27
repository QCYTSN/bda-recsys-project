from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from datasets.seq_dataset import SeqDataset
from models.time_sasrec import TimeAwareSASRec


def main() -> None:
    clean_user_sequences = torch.load("data/interim/clean_user_sequences.pt")

    dataset = SeqDataset(
        user_sequences=clean_user_sequences,
        max_len=50,
        padding_idx=0,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    input_times = batch["input_times"]

    model = TimeAwareSASRec(
        num_items=1706,
        max_len=50,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
        padding_idx=0,
        num_time_buckets=8,
    )

    logits = model(input_ids, input_times)

    print("input_ids shape:", tuple(input_ids.shape))
    print("input_times shape:", tuple(input_times.shape))
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()