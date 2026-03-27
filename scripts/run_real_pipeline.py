from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from datasets.seq_dataset import SeqDataset


def main() -> None:
    clean_user_sequences = torch.load("data/interim/clean_user_sequences.pt")

    dataset = SeqDataset(
        user_sequences=clean_user_sequences,
        max_len=50,
        padding_idx=0,
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(f"Number of samples: {len(dataset)}")
    print("-" * 50)

    first_sample = dataset[0]
    print("First sample:")
    for key, value in first_sample.items():
        print(f"{key}: {value.tolist()}")

    print("-" * 50)

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        for key, value in batch.items():
            print(f"{key}: shape={tuple(value.shape)}")
            print(value[:2] if value.ndim > 1 else value)
        print("-" * 50)

        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()