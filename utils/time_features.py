from __future__ import annotations

from typing import List

import torch


def left_pad(seq: List[int], max_len: int, pad_value: int = 0) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq


def build_time_diff_bucket_matrix(
    timestamps: List[int],
    max_len: int,
    pad_value: int = 0,
    num_buckets: int = 8,
) -> torch.Tensor:
    """
    Build a [L, L] relative time bucket matrix for one padded sequence.

    timestamps: raw timestamp list before padding
    """
    padded_times = left_pad(timestamps, max_len=max_len, pad_value=pad_value)
    L = max_len

    bucket_mat = torch.zeros((L, L), dtype=torch.long)

    for i in range(L):
        for j in range(L):
            ti = padded_times[i]
            tj = padded_times[j]

            if ti == pad_value or tj == pad_value:
                bucket = num_buckets - 1
            else:
                diff = abs(ti - tj)

                if diff == 0:
                    bucket = 0
                elif diff < 60 * 60 * 24:
                    bucket = 1
                elif diff < 60 * 60 * 24 * 7:
                    bucket = 2
                elif diff < 60 * 60 * 24 * 30:
                    bucket = 3
                elif diff < 60 * 60 * 24 * 90:
                    bucket = 4
                elif diff < 60 * 60 * 24 * 180:
                    bucket = 5
                elif diff < 60 * 60 * 24 * 365:
                    bucket = 6
                else:
                    bucket = 7

            bucket_mat[i, j] = bucket

    return bucket_mat