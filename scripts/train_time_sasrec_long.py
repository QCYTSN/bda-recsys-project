from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.train_split_seq_dataset import TrainSplitSeqDataset
from models.time_sasrec import TimeAwareSASRec
from utils.metrics import hit_rate_at_k, ndcg_at_k


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def evaluate_on_split(
    model: TimeAwareSASRec,
    eval_data: list,
    user_time_map: dict[int, List[int]],
    split_name: str,
    max_len: int = 50,
    padding_idx: int = 0,
    max_eval_users: int = 500,
    eval_seed: int = 42,
) -> dict[str, float]:
    model.eval()

    if max_eval_users is None or max_eval_users >= len(eval_data):
        eval_subset = eval_data
    else:
        rng = random.Random(eval_seed)
        eval_subset = rng.sample(eval_data, max_eval_users)

    if len(eval_subset) == 0:
        raise ValueError("No evaluation data available.")

    hr10_scores = []
    ndcg10_scores = []
    hr20_scores = []
    ndcg20_scores = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for user_id, eval_seq, target_item in eval_subset:
            full_times = user_time_map[int(user_id)]
            eval_times = full_times[: len(eval_seq)]

            padded_seq = left_pad(eval_seq, max_len=max_len, pad_value=padding_idx)
            padded_times = left_pad(eval_times, max_len=max_len, pad_value=0)

            input_ids = torch.tensor([padded_seq], dtype=torch.long, device=device)
            input_times = torch.tensor([padded_times], dtype=torch.long, device=device)

            logits = model(input_ids, input_times)

            last_idx = get_last_valid_index(padded_seq, pad_value=padding_idx)
            final_logits = logits[0, last_idx].clone()

            # do not recommend padding item
            final_logits[padding_idx] = -1e9

            # mask seen items
            for seen_item in eval_seq:
                final_logits[seen_item] = -1e9

            ranked_items = torch.argsort(final_logits, descending=True).tolist()

            hr10_scores.append(hit_rate_at_k(ranked_items, target_item, 10))
            ndcg10_scores.append(ndcg_at_k(ranked_items, target_item, 10))
            hr20_scores.append(hit_rate_at_k(ranked_items, target_item, 20))
            ndcg20_scores.append(ndcg_at_k(ranked_items, target_item, 20))

    metrics = {
        "HR@10": sum(hr10_scores) / len(hr10_scores),
        "NDCG@10": sum(ndcg10_scores) / len(ndcg10_scores),
        "HR@20": sum(hr20_scores) / len(hr20_scores),
        "NDCG@20": sum(ndcg20_scores) / len(ndcg20_scores),
    }

    print("-" * 50)
    print(f"{split_name} over {len(hr10_scores)} users")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def main() -> None:
    set_seed(42)

    clean_user_sequences = torch.load("data/interim/clean_user_sequences.pt")
    artifacts = torch.load("data/interim/sequence_artifacts.pt")

    train_data = artifacts["train_data"]
    val_data = artifacts["val_data"]
    test_data = artifacts["test_data"]
    num_items = artifacts["num_items"]
    padding_idx = artifacts["padding_idx"]

    user_time_map = build_user_time_map(clean_user_sequences)

    max_len = 50
    batch_size = 32
    max_epochs = 20
    max_eval_users = 500
    lr = 1e-3
    patience = 5

    checkpoint_dir = Path('outputs/checkpoints/time_sasrec_long')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = TrainSplitSeqDataset(
        train_data=train_data,
        user_time_map=user_time_map,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state_dict = None
    best_epoch = -1
    best_val_hr10 = -1.0
    best_val_metrics = None
    epochs_without_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            input_times = batch["input_times"].to(device)
            target_ids = batch["target_ids"].to(device)

            logits = model(input_ids, input_times)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{max_epochs}")
        print(f"Train Average Loss: {avg_loss:.4f}")

        val_metrics = evaluate_on_split(
            model=model,
            eval_data=val_data,
            user_time_map=user_time_map,
            split_name="Validation",
            max_len=max_len,
            padding_idx=padding_idx,
            max_eval_users=max_eval_users,
            eval_seed=42,
        )

        current_val_hr10 = val_metrics["HR@10"]

        if current_val_hr10 > best_val_hr10:
            best_val_hr10 = current_val_hr10
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            print(f"New best model found at epoch {best_epoch} with HR@10={best_val_hr10:.4f}")
        else:
            epochs_without_improve += 1
            print(
                f"No improvement for {epochs_without_improve} epoch(s) "
                f"(patience={patience})."
            )

        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "best_val_hr10_so_far": best_val_hr10,
                "best_epoch_so_far": best_epoch,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break

    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print("Best validation metrics:")
    for k, v in best_val_metrics.items():
        print(f"{k}: {v:.4f}")

    model.load_state_dict(best_state_dict)

    print("=" * 60)
    print("Evaluating best model on test set...")
    test_metrics = evaluate_on_split(
        model=model,
        eval_data=test_data,
        user_time_map=user_time_map,
        split_name="Test",
        max_len=max_len,
        padding_idx=padding_idx,
        max_eval_users=max_eval_users,
        eval_seed=42,
    )

    print("=" * 60)
    print("Final Summary")
    print(f"Best epoch: {best_epoch}")
    print("Best validation:")
    for k, v in best_val_metrics.items():
        print(f"{k}: {v:.4f}")
    print("Test with best model:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()


