from __future__ import annotations

import math
from typing import Sequence


def hit_rate_at_k(ranked_items: Sequence[int], target_item: int, k: int) -> float:
    """
    Hit Rate@K

    Args:
        ranked_items: ranked recommendation list
        target_item: ground-truth next item
        k: top-k cutoff

    Returns:
        1.0 if target_item is in top-k else 0.0
    """
    top_k_items = ranked_items[:k]
    return 1.0 if target_item in top_k_items else 0.0


def ndcg_at_k(ranked_items: Sequence[int], target_item: int, k: int) -> float:
    """
    NDCG@K for a single ground-truth item.

    Args:
        ranked_items: ranked recommendation list
        target_item: ground-truth next item
        k: top-k cutoff

    Returns:
        discounted gain if target_item is in top-k, else 0.0
    """
    top_k_items = ranked_items[:k]

    if target_item in top_k_items:
        rank = top_k_items.index(target_item) + 1
        return 1.0 / math.log2(rank + 1)

    return 0.0