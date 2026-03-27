from __future__ import annotations

import torch


def main() -> None:
    clean_user_sequences = torch.load("data/interim/clean_user_sequences.pt")
    artifacts = torch.load("data/interim/sequence_artifacts.pt")

    print("Loaded clean_user_sequences.")
    print("Number of users:", len(clean_user_sequences))

    first_user = next(iter(clean_user_sequences))
    print("First user:", first_user)
    print("First user sequence:", clean_user_sequences[first_user])

    print("-" * 50)

    print("Artifacts keys:", artifacts.keys())
    print("num_users:", artifacts["num_users"])
    print("num_items:", artifacts["num_items"])
    print("padding_idx:", artifacts["padding_idx"])


if __name__ == "__main__":
    main()