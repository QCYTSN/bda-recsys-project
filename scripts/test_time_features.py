from __future__ import annotations

import torch

from utils.time_features import build_time_diff_bucket_matrix


def main() -> None:
    timestamps = [1162252800, 1171756800, 1192147200, 1192752000]
    max_len = 6

    bucket_mat = build_time_diff_bucket_matrix(
        timestamps=timestamps,
        max_len=max_len,
        pad_value=0,
        num_buckets=8,
    )

    print("Bucket matrix shape:", tuple(bucket_mat.shape))
    print(bucket_mat)


if __name__ == "__main__":
    main()