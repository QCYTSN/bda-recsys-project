from __future__ import annotations

import torch
import torch.nn as nn

from utils.model_utils import get_last_non_padding_indices


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
        B, L, H = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # [B, L, L, heads] -> [B, heads, L, L]
        t_bias = self.time_bias(time_bucket_matrix).permute(0, 3, 1, 2)
        attn_scores = attn_scores + t_bias

        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        attn_scores = attn_scores.masked_fill(key_padding_mask, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Keep padded query rows numerically safe.
        query_padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        attn_weights = attn_weights.masked_fill(query_padding_mask, 0.0)

        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, H)
        return self.out_proj(out)


class TimeAwareTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_time_buckets: int) -> None:
        super().__init__()
        self.attn = TimeAwareMultiHeadSelfAttention(hidden_dim, num_heads, dropout, num_time_buckets)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
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
        x_norm = self.attn_norm(x)
        x = x + self.attn(x_norm, time_bucket_matrix, causal_mask, padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TimeAwareSASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        padding_idx: int = 0,
        num_time_buckets: int = 8,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.num_items = num_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.num_time_buckets = num_time_buckets

        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TimeAwareTransformerBlock(hidden_dim, num_heads, dropout, num_time_buckets)
                for _ in range(num_layers)
            ]
        )

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _build_batch_time_bucket_matrix_fast(self, input_times: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # input_times: [B, L], padding_mask: [B, L]
        time_diff = torch.abs(input_times.unsqueeze(2) - input_times.unsqueeze(1))  # [B, L, L]

        # Log-bucket approximation: 0,1,2,... then clamp to valid bucket range.
        bucket_matrix = torch.log(time_diff.to(torch.float32) + 1.0).long()
        bucket_matrix = torch.clamp(bucket_matrix, 0, self.num_time_buckets - 1)

        # Preserve previous behavior for padded positions: map to last bucket.
        pad_pair_mask = padding_mask.unsqueeze(2) | padding_mask.unsqueeze(1)  # [B, L, L]
        bucket_matrix = torch.where(
            pad_pair_mask,
            torch.full_like(bucket_matrix, self.num_time_buckets - 1),
            bucket_matrix,
        )

        return bucket_matrix

    def _encode(self, input_ids: torch.Tensor, input_times: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        device = input_ids.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.item_embedding(input_ids) + self.position_embedding(pos)
        x = self.emb_norm(self.emb_dropout(x))

        causal_mask = self._build_causal_mask(L, device=device)
        padding_mask = input_ids.eq(self.padding_idx)
        time_bucket_matrix = self._build_batch_time_bucket_matrix_fast(input_times, padding_mask)

        for block in self.blocks:
            x = block(x, time_bucket_matrix, causal_mask, padding_mask)
        return x

    def forward(self, input_ids: torch.Tensor, input_times: torch.Tensor) -> torch.Tensor:
        hidden_states = self._encode(input_ids, input_times)

        item_embs = self.item_embedding.weight  # [num_items + 1, H]
        logits = torch.matmul(hidden_states, item_embs.transpose(0, 1))
        return logits

    def get_sequence_embedding(self, input_ids: torch.Tensor, input_times: torch.Tensor) -> torch.Tensor:
        hidden_states = self._encode(input_ids, input_times)
        last_indices = get_last_non_padding_indices(input_ids, self.padding_idx)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        return hidden_states[batch_indices, last_indices]
