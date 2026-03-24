from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset


@dataclass
class SequenceSample:
    input_ids: List[int]
    target_ids: List[int]
    input_times: List[int]
    attention_mask: List[int]


class SeqDataset(Dataset):
    """
    Minimal sequence recommendation dataset for next-item prediction.

    Input format:
    {
        user_id: {
            "items": [3, 8, 12, 6],
            "times": [10, 20, 40, 90]
        }
    }
    """

    def __init__(
        self,
        user_sequences: Dict[int, Dict[str, List[int]]],
        max_len: int = 4,
        padding_idx: int = 0,
    ) -> None:
        self.user_sequences = user_sequences
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.samples: List[SequenceSample] = []
        self._build_samples()

    def _left_pad(self, seq: List[int], pad_value: int) -> List[int]:
        if len(seq) >= self.max_len:
            return seq[-self.max_len:]
        return [pad_value] * (self.max_len - len(seq)) + seq

    def _build_samples(self) -> None:
        for _, seq_data in self.user_sequences.items():
            items = seq_data["items"]
            times = seq_data["times"]

            if len(items) < 2:
                continue

            for end_idx in range(1, len(items)):
                input_items = items[:end_idx]
                target_items = items[1 : end_idx + 1]
                input_times = times[:end_idx]

                input_ids_padded = self._left_pad(input_items, self.padding_idx)
                target_ids_padded = self._left_pad(target_items, self.padding_idx)
                input_times_padded = self._left_pad(input_times, 0)

                attention_mask = [1 if x != self.padding_idx else 0 for x in input_ids_padded]

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample.input_ids, dtype=torch.long),
            "target_ids": torch.tensor(sample.target_ids, dtype=torch.long),
            "input_times": torch.tensor(sample.input_times, dtype=torch.long),
            "attention_mask": torch.tensor(sample.attention_mask, dtype=torch.long),
        }