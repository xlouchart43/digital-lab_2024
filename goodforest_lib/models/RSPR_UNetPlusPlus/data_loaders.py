import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CubeDataset(Dataset):
    """
    Custom dataset class for loading the cubes dataset.
    """

    def __init__(
        self, file_path: str, normalize: bool = True, normalization_stats: list = None
    ):
        """
        Initialize the dataset.

        Args:

            file_path (str): Path to the .npy file containing the dataset.
            normalize (bool): Whether to normalize the dataset.
            normalization_stats (list): List of tuples containing the mean and standard deviation for each channel.
        """
        if file_path.endswith(".npy"):
            data = np.load(file_path, allow_pickle=True)
        elif file_path.endswith(".h5"):
            with h5py.File(file_path, "r") as f:
                data = f["cubes"][:]
        elif file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        # Rotate the cubes randomly
        rotations = np.random.randint(4, size=len(data))
        cubes = np.zeros_like(data)
        cubes[rotations == 0] = data[rotations == 0]
        cubes[rotations == 1] = np.rot90(data[rotations == 1], k=1, axes=[2, 3])
        cubes[rotations == 2] = np.rot90(data[rotations == 2], k=2, axes=[2, 3])
        cubes[rotations == 3] = np.rot90(data[rotations == 3], k=3, axes=[2, 3])
        self.labels = [cube[-1] - 1 for cube in cubes]
        self.features = [cube[:-1] for cube in cubes]
        self.features = np.array(self.features)
        self.labels = np.array(self.labels, dtype=np.uint8)

        self.normalize = normalize
        self.normalization_stats = normalization_stats

        if normalize and normalization_stats is None:
            self.compute_normalization_stats()

        if normalize:
            self.apply_normalization()

    def __len__(self) -> int:
        """
        Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset."""
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return feature, label

    def compute_normalization_stats(self):
        """
        Compute the mean and standard deviation for each channel in the dataset.
        """
        non_black_pixels = np.any(self.features[:11] != 0, axis=0)
        self.normalization_stats = []
        for i in range(11):
            channel_data = self.features[i][non_black_pixels]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            self.normalization_stats.append((mean, std))

    def apply_normalization(self):
        """
        Normalize the dataset using the computed normalization statistics.
        """
        for i in range(11):
            mean, std = self.normalization_stats[i]
            if std > 1e-10:
                self.features[i] = (self.features[i] - mean) / std
            else:
                self.features[i] = self.features[i] - mean


def get_loaders(
    file_path: str,
    val_path: str,
    batch_size: int = 16,
    train_split: float = 0.8,
    normalize=True,
):
    """
    Get the train and validation data loaders.

    Args:
        file_path (str): Path to the .npy file containing the dataset.
        val_path (str): Path to the .npy file containing the validation dataset.
        batch_size (int): Batch size for the data loaders.
        train_split (float): Fraction of the data to be used for training.
        normalize (bool): Whether to normalize the dataset.

    Returns:
        DataLoader: Training data loader.
        DataLoader: Validation data loader.
        list: Normalization statistics."""

    temp_dataset = CubeDataset(file_path, normalize=False)

    if normalize:
        temp_dataset.compute_normalization_stats()
        normalization_stats = temp_dataset.normalization_stats
    else:
        normalization_stats = None

    full_dataset = CubeDataset(
        file_path, normalize=normalize, normalization_stats=normalization_stats
    )
    if val_path is not None:
        val_dataset = CubeDataset(
            val_path, normalize=normalize, normalization_stats=normalization_stats
        )
        train_dataset = full_dataset
    else:
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Print some information about the datasets and loaders
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")

    return train_loader, val_loader, normalization_stats
