from __future__ import annotations

import torch
import torch.nn as nn

from utils.model_utils import get_last_non_padding_indices


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

    def get_sequence_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get sequence representation E_seq for each sequence in the batch.

        Args:
            input_ids: [B, L]

        Returns:
            seq_emb: [B, hidden_dim]
        """
        _, hidden_states = self.forward(input_ids)  # [B, L, H]
        last_indices = get_last_non_padding_indices(
            input_ids=input_ids,
            padding_idx=self.padding_idx,
        )  # [B]

        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        seq_emb = hidden_states[batch_indices, last_indices]  # [B, H]
        return seq_emb
