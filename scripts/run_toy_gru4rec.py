from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Support running as: python scripts/run_toy_gru4rec.py
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from datasets.seq_dataset import SeqDataset
from models.gru4rec import GRU4Rec


def main() -> None:
    toy_sequences = {
        1: {"items": [3, 8, 12, 6, 20], "times": [10, 20, 35, 80, 120]},
        2: {"items": [5, 9, 4, 11], "times": [15, 18, 60, 100]},
        3: {"items": [7, 2, 14], "times": [8, 40, 90]},
    }

    max_len = 4
    padding_idx = 0
    batch_size = 2
    num_items = 20

    dataset = SeqDataset(
        user_sequences=toy_sequences,
        max_len=max_len,
        padding_idx=padding_idx,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRU4Rec(
        num_items=num_items,
        embedding_dim=32,
        hidden_dim=32,
        padding_idx=padding_idx,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(3):
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]     # [B, L]
            target_ids = batch["target_ids"]   # [B, L]

            logits = model(input_ids)          # [B, L, num_items+1]

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
