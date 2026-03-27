from __future__ import annotations

from utils.metrics import hit_rate_at_k, ndcg_at_k


def main() -> None:
    ranked_items = [10, 20, 30, 40, 50]

    target_item_1 = 20
    target_item_2 = 60

    print("Case 1: target in top-k")
    print("HR@3:", hit_rate_at_k(ranked_items, target_item_1, 3))
    print("NDCG@3:", ndcg_at_k(ranked_items, target_item_1, 3))

    print("-" * 50)

    print("Case 2: target not in top-k")
    print("HR@3:", hit_rate_at_k(ranked_items, target_item_2, 3))
    print("NDCG@3:", ndcg_at_k(ranked_items, target_item_2, 3))


if __name__ == "__main__":
    main()