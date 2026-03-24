from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

# Support running as: python scripts/run_toy_pipeline.py
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from datasets.seq_dataset import SeqDataset


def main() -> None:
    toy_sequences = {
        1: {"items": [3, 8, 12, 6, 20], "times": [10, 20, 35, 80, 120]},
        2: {"items": [5, 9, 4, 11], "times": [15, 18, 60, 100]},
        3: {"items": [7, 2, 14], "times": [8, 40, 90]},
    }

    dataset = SeqDataset(user_sequences=toy_sequences, max_len=4, padding_idx=0)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

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
            print(f"{key}:")
            print(value)
        print("-" * 50)

        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()
