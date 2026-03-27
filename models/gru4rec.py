from __future__ import annotations

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    """
    Minimal GRU4Rec model for sequence recommendation.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dim: int = 32,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, L]

        Returns:
            logits: [B, L, num_items + 1]
            hidden_states: [B, L, hidden_dim]
        """
        x = self.item_embedding(input_ids)
        h, _ = self.gru(x)
        logits = self.output_layer(h)
        return logits, h