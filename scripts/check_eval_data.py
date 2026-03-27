from __future__ import annotations

import torch


def main() -> None:
    artifacts = torch.load("data/interim/sequence_artifacts.pt")

    train_data = artifacts["train_data"]
    val_data = artifacts["val_data"]
    test_data = artifacts["test_data"]

    print("Number of train entries:", len(train_data))
    print("Number of val entries:", len(val_data))
    print("Number of test entries:", len(test_data))
    print("-" * 50)

    print("Example train entry:")
    print(train_data[0])
    print("-" * 50)

    print("Example val entry:")
    print(val_data[0])
    print("-" * 50)

    print("Example test entry:")
    print(test_data[0])


if __name__ == "__main__":
    main()