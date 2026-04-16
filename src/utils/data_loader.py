"""Data loading utilities."""

from typing import Optional

import torch


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Create a PyTorch DataLoader with common defaults."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class InterferometryDataset(torch.utils.data.Dataset):
    """Minimal dataset skeleton. Replace with your actual dataset."""

    def __init__(self, data_dir: str, transform=None, length: int = 0):
        self.data_dir = data_dir
        self.transform = transform
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        raise NotImplementedError("Implement dataset access logic.")


class PlaceholderDataset(InterferometryDataset):
    """Backward-compatible alias for the minimal dataset skeleton."""

    def __init__(self, length: int = 0):
        super().__init__(data_dir="", transform=None, length=length)
