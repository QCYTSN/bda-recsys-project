from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.seq_dataset import SeqDataset
from models.gru4rec import GRU4Rec


def main() -> None:
    clean_user_sequences = torch.load("data/interim/clean_user_sequences.pt")

    max_len = 50
    padding_idx = 0
    batch_size = 32
    num_items = 1706
    epochs = 3

    checkpoint_dir = Path("outputs/checkpoints/gru4rec_real")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = SeqDataset(
        user_sequences=clean_user_sequences,
        max_len=max_len,
        padding_idx=padding_idx,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRU4Rec(
        num_items=num_items,
        embedding_dim=64,
        hidden_dim=64,
        padding_idx=padding_idx,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]

            logits, _ = model(input_ids)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_train_loss": avg_loss,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")
        print("-" * 50)


if __name__ == "__main__":
    main()
