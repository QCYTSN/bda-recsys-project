from __future__ import annotations

import torch
import torch.nn as nn

from utils.model_utils import get_last_non_padding_indices


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
        )
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        self.emb_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.layer_norm(self.emb_dropout(x))

        causal_mask = self._build_causal_mask(seq_len, device=device)
        padding_mask = input_ids.eq(self.padding_idx)

        hidden_states = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return hidden_states

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self._encode(input_ids)

        item_embs = self.item_embedding.weight  # [num_items + 1, H]
        logits = torch.matmul(hidden_states, item_embs.transpose(0, 1))
        return logits

    def get_sequence_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self._encode(input_ids)
        last_indices = get_last_non_padding_indices(
            input_ids=input_ids,
            padding_idx=self.padding_idx,
        )
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        seq_emb = hidden_states[batch_indices, last_indices]
        return seq_emb
