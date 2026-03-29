from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class SequenceSample:
    input_ids: List[int]
    target_ids: List[int]
    input_times: List[int]
    attention_mask: List[int]


class TrainSplitSeqDataset(Dataset):
    """
    Prefix-based sequence training dataset built ONLY from train_data.

    train_data format:
        [(user_id, train_seq), ...]

    user_time_map format:
        {
            user_id: [full timestamps in chronological order]
        }

    We align timestamps by taking the first len(train_seq) timestamps for each user.
    """

    def __init__(
        self,
        train_data: List[Tuple[int, List[int]]],
        user_time_map: Dict[int, List[int]],
        max_len: int = 50,
        padding_idx: int = 0,
    ) -> None:
        self.train_data = train_data
        self.user_time_map = user_time_map
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.samples: List[SequenceSample] = []

        self._build_samples()

    def _left_pad(self, seq: List[int], pad_value: int) -> List[int]:
        if len(seq) >= self.max_len:
            return seq[-self.max_len:]
        return [pad_value] * (self.max_len - len(seq)) + seq

    def _build_samples(self) -> None:
        for user_id, train_seq in self.train_data:
            if len(train_seq) < 2:
                continue

            full_times = self.user_time_map[int(user_id)]
            train_times = full_times[: len(train_seq)]

            for end_idx in range(1, len(train_seq)):
                input_items = train_seq[:end_idx]
                target_items = train_seq[1 : end_idx + 1]
                input_times = train_times[:end_idx]

                input_ids_padded = self._left_pad(input_items, self.padding_idx)
                target_ids_padded = self._left_pad(target_items, self.padding_idx)
                input_times_padded = self._left_pad(input_times, 0)

                attention_mask = [
                    1 if x != self.padding_idx else 0
                    for x in input_ids_padded
                ]

                self.samples.append(
                    SequenceSample(
                        input_ids=input_ids_padded,
                        target_ids=target_ids_padded,
                        input_times=input_times_padded,
                        attention_mask=attention_mask,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample.input_ids, dtype=torch.long),
            "target_ids": torch.tensor(sample.target_ids, dtype=torch.long),
            "input_times": torch.tensor(sample.input_times, dtype=torch.long),
            "attention_mask": torch.tensor(sample.attention_mask, dtype=torch.long),
        }