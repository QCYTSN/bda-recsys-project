from __future__ import annotations

import torch


def get_last_non_padding_indices(
    input_ids: torch.Tensor,
    padding_idx: int = 0,
) -> torch.Tensor:
    """
    Get the last non-padding index for each sequence in a batch.

    Args:
        input_ids: [B, L]
        padding_idx: padding token index

    Returns:
        indices: [B]
    """
    non_padding_mask = input_ids.ne(padding_idx)  # [B, L]
    lengths = non_padding_mask.sum(dim=1)         # [B]
    indices = torch.clamp(lengths - 1, min=0)     # [B]
    return indices