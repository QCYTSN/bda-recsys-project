from __future__ import annotations

import torch
import torch.nn as nn

from utils.model_utils import get_last_non_padding_indices


class GRU4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
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
        self.emb_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        if hidden_dim != embedding_dim:
            self.project = nn.Linear(hidden_dim, embedding_dim)
        else:
            self.project = nn.Identity()

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.item_embedding(input_ids)  # [B, L, E]
        x = self.emb_dropout(x)

        h, _ = self.gru(x)                  # [B, L, H]
        h = self.layer_norm(h)

        seq_features = self.project(h)      # [B, L, E]

        item_embs = self.item_embedding.weight  # [num_items + 1, E]
        logits = torch.matmul(seq_features, item_embs.transpose(0, 1))
        return logits, h

    def get_sequence_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, hidden_states = self.forward(input_ids)
        last_indices = get_last_non_padding_indices(
            input_ids=input_ids,
            padding_idx=self.padding_idx,
        )
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        seq_emb = hidden_states[batch_indices, last_indices]
        return seq_emb
