from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.seq_dataset import SeqDataset
from models.time_sasrec import TimeAwareSASRec
from utils.metrics import hit_rate_at_k, ndcg_at_k


def left_pad(seq: List[int], max_len: int, pad_value: int = 0) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq


def get_last_valid_index(seq: List[int], pad_value: int = 0) -> int:
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] != pad_value:
            return i
    return 0


def build_user_time_map(clean_user_sequences: dict) -> dict[int, List[int]]:
    return {int(user_id): data["times"] for user_id, data in clean_user_sequences.items()}


def train_one_small_model(
    clean_user_sequences: dict,
    num_items: int,
    max_len: int = 50,
    padding_idx: int = 0,
    batch_size: int = 32,
    epochs: int = 3,
) -> TimeAwareSASRec:
    dataset = SeqDataset(
        user_sequences=clean_user_sequences,
        max_len=max_len,
        padding_idx=padding_idx,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TimeAwareSASRec(
        num_items=num_items,
        max_len=max_len,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
        padding_idx=padding_idx,
        num_time_buckets=8,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"]
            input_times = batch["input_times"]
            target_ids = batch["target_ids"]

            logits = model(input_ids, input_times)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Train Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    return model


def evaluate_on_test(
    model: TimeAwareSASRec,
    test_data: list,
    user_time_map: dict[int, List[int]],
    max_len: int = 50,
    padding_idx: int = 0,
    max_eval_users: int = 200,
) -> None:
    model.eval()

    hr10_scores = []
    ndcg10_scores = []
    hr20_scores = []
    ndcg20_scores = []

    with torch.no_grad():
        for user_id, train_seq, target_item in test_data[:max_eval_users]:
            full_times = user_time_map[int(user_id)]
            train_times = full_times[: len(train_seq)]

            padded_seq = left_pad(train_seq, max_len=max_len, pad_value=padding_idx)
            padded_times = left_pad(train_times, max_len=max_len, pad_value=0)

            input_ids = torch.tensor([padded_seq], dtype=torch.long)
            input_times = torch.tensor([padded_times], dtype=torch.long)

            logits = model(input_ids, input_times)

            last_idx = get_last_valid_index(padded_seq, pad_value=padding_idx)
            final_logits = logits[0, last_idx].clone()

            # do not recommend padding item
            final_logits[padding_idx] = -1e9

            # mask seen items
            for seen_item in train_seq:
                final_logits[seen_item] = -1e9

            ranked_items = torch.argsort(final_logits, descending=True).tolist()

            hr10 = hit_rate_at_k(ranked_items, target_item, 10)
            ndcg10 = ndcg_at_k(ranked_items, target_item, 10)
            hr20 = hit_rate_at_k(ranked_items, target_item, 20)
            ndcg20 = ndcg_at_k(ranked_items, target_item, 20)

            hr10_scores.append(hr10)
            ndcg10_scores.append(ndcg10)
            hr20_scores.append(hr20)
            ndcg20_scores.append(ndcg20)

    mean_hr10 = sum(hr10_scores) / len(hr10_scores)
    mean_ndcg10 = sum(ndcg10_scores) / len(ndcg10_scores)
    mean_hr20 = sum(hr20_scores) / len(hr20_scores)
    mean_ndcg20 = sum(ndcg20_scores) / len(ndcg20_scores)

    print("-" * 50)
    print(f"Test over {len(hr10_scores)} users")
    print(f"HR@10: {mean_hr10:.4f}")
    print(f"NDCG@10: {mean_ndcg10:.4f}")
    print(f"HR@20: {mean_hr20:.4f}")
    print(f"NDCG@20: {mean_ndcg20:.4f}")


def main() -> None:
    clean_user_sequences = torch.load("data/interim/clean_user_sequences.pt")
    artifacts = torch.load("data/interim/sequence_artifacts.pt")

    test_data = artifacts["test_data"]
    num_items = artifacts["num_items"]
    padding_idx = artifacts["padding_idx"]

    user_time_map = build_user_time_map(clean_user_sequences)

    model = train_one_small_model(
        clean_user_sequences=clean_user_sequences,
        num_items=num_items,
        max_len=50,
        padding_idx=padding_idx,
        batch_size=32,
        epochs=3,
    )

    evaluate_on_test(
        model=model,
        test_data=test_data,
        user_time_map=user_time_map,
        max_len=50,
        padding_idx=padding_idx,
        max_eval_users=500,
    )


if __name__ == "__main__":
    main()

