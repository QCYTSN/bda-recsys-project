from __future__ import annotations

import torch
import torch.nn as nn


class SASRec(nn.Module):
    """
    Minimal SASRec-style model for sequence recommendation.
    """

    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_dim: int = 32,
        num_heads: int = 2,
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
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(hidden_dim, num_items + 1)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns shape [seq_len, seq_len].
        True means masked position.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L]

        Returns:
            logits: [B, L, num_items + 1]
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = self._build_causal_mask(seq_len, device=device)
        padding_mask = input_ids.eq(self.padding_idx)  # [B, L]

        h = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        logits = self.output_layer(h)
        return logits