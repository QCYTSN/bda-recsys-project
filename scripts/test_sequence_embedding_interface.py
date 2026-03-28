from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Support running as: python scripts/test_sequence_embedding_interface.py
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from datasets.seq_dataset import SeqDataset
from models.gru4rec import GRU4Rec
from models.sasrec import SASRec
from models.time_sasrec import TimeAwareSASRec


def main() -> None:
    clean_user_sequences = torch.load("data/interim/clean_user_sequences.pt")
    artifacts = torch.load("data/interim/sequence_artifacts.pt")
    num_items = artifacts["num_items"]

    dataset = SeqDataset(
        user_sequences=clean_user_sequences,
        max_len=50,
        padding_idx=0,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    input_times = batch["input_times"]

    gru_model = GRU4Rec(
        num_items=num_items,
        embedding_dim=64,
        hidden_dim=64,
        padding_idx=0,
    )

    sasrec_model = SASRec(
        num_items=num_items,
        max_len=50,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
        padding_idx=0,
    )

    time_sasrec_model = TimeAwareSASRec(
        num_items=num_items,
        max_len=50,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
        padding_idx=0,
        num_time_buckets=8,
    )

    gru_seq_emb = gru_model.get_sequence_embedding(input_ids)
    sasrec_seq_emb = sasrec_model.get_sequence_embedding(input_ids)
    time_sasrec_seq_emb = time_sasrec_model.get_sequence_embedding(input_ids, input_times)

    print("GRU4Rec E_seq shape:", tuple(gru_seq_emb.shape))
    print("SASRec E_seq shape:", tuple(sasrec_seq_emb.shape))
    print("Time-Aware SASRec E_seq shape:", tuple(time_sasrec_seq_emb.shape))


if __name__ == "__main__":
    main()
