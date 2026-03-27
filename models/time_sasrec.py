from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from utils.time_features import build_time_diff_bucket_matrix


class TimeAwareMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        num_time_buckets: int = 8,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.time_bias = nn.Embedding(num_time_buckets, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_bucket_matrix: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, L, H]
        time_bucket_matrix: [B, L, L]
        causal_mask: [L, L] (True means masked)
        padding_mask: [B, L] (True means padding)
        """
        B, L, H = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, L, d]
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, L, L]

        # time bias: [B, L, L, heads] -> [B, heads, L, L]
        time_bias = self.time_bias(time_bucket_matrix).permute(0, 3, 1, 2)
        attn_scores = attn_scores + time_bias

        # causal mask: mask future positions
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            -1e9,
        )

        # key padding mask: do not attend to padded keys
        key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        attn_scores = attn_scores.masked_fill(key_padding_mask, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # query padding mask: zero out attention rows for padded queries
        query_padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        attn_weights = attn_weights.masked_fill(query_padding_mask, 0.0)

        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, heads, L, d]
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.out_proj(out)

        return out


class TimeAwareTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        num_time_buckets: int = 8,
    ) -> None:
        super().__init__()

        self.attn = TimeAwareMultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_time_buckets=num_time_buckets,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_bucket_matrix: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attn(x, time_bucket_matrix, causal_mask, padding_mask)
        x = self.attn_norm(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        return x


class TimeAwareSASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        padding_idx: int = 0,
        num_time_buckets: int = 8,
    ) -> None:
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.num_time_buckets = num_time_buckets

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
        )
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TimeAwareTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    num_time_buckets=num_time_buckets,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_layer = nn.Linear(hidden_dim, num_items + 1)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _build_batch_time_bucket_matrix(self, input_times: torch.Tensor) -> torch.Tensor:
        """
        input_times: [B, L]
        return: [B, L, L]
        """
        batch_bucket_mats = []
        for b in range(input_times.size(0)):
            timestamps = input_times[b].tolist()
            bucket_mat = build_time_diff_bucket_matrix(
                timestamps=timestamps,
                max_len=self.max_len,
                pad_value=0,
                num_buckets=self.num_time_buckets,
            )
            batch_bucket_mats.append(bucket_mat)

        return torch.stack(batch_bucket_mats, dim=0).to(input_times.device)

    def forward(self, input_ids: torch.Tensor, input_times: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, L]
        input_times: [B, L]
        """
        B, L = input_ids.shape
        device = input_ids.device

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = self._build_causal_mask(L, device=device)
        padding_mask = input_ids.eq(self.padding_idx)
        time_bucket_matrix = self._build_batch_time_bucket_matrix(input_times)

        for block in self.blocks:
            x = block(
                x=x,
                time_bucket_matrix=time_bucket_matrix,
                causal_mask=causal_mask,
                padding_mask=padding_mask,
            )

        logits = self.output_layer(x)
        return logits