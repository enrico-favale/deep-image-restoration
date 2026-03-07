from pathlib import Path
from typing import Tuple, List
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class CompressionDataset(Dataset):
    """
    PyTorch Dataset of (compressed_image, original_image) pairs.

    Reads image pairs from disk by matching filenames:
        data/compressed/ {id}_fourier_25.png -> input tensor
        data/resized/ {id}.jpg -> target tensor

    Applies resize and normalization to both images before returning them.
    Inherits from torch.utils.data.Dataset, making it compatible with
    DataLoader, Subset, and all standard PyTorch data utilities.
    """

    def __init__(
        self,
        input_dir: Path,
        compressed_dir: Path,
        image_size: int = 256,
        suffix: str = "_fourier_25",
    ):
        """
        Initializes the dataset by scanning the compressed directory
        and building a list of valid (compressed, original) path pairs.

        Parameters
        ----------
        - input_dir : Path
            Directory containing the original (ground truth) images.
        - compressed_dir : Path
            Directory containing the compressed images produced by the
            compression pipeline.
        - image_size : int, optional
            Both images are resized to (image_size x image_size) pixels.
        - suffix : str, optional
            Suffix to strip from the compressed filename to recover
            the original image ID.
        """
        
        self.input_dir = Path(input_dir)
        self.compressed_dir = Path(compressed_dir)
        self.image_size = image_size
        self.suffix = suffix

        self.pairs = self._build_pairs()
        if len(self.pairs) == 0:
            raise ValueError(f"No pairs found in {compressed_dir}")

        # Resize to fixed spatial dimensions, then convert to float tensor in [0, 1]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # [0, 255] uint8 -> [0.0, 1.0] float32
            ]
        )

    def _build_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Scans compressed_dir and builds the list of valid path pairs.

        For each compressed file, strips the suffix from the stem to
        recover the original image ID, then searches for a matching file
        in input_dir (trying .jpg, .jpeg, .png in order).
        Pairs where the original file does not exist are silently skipped.

        Returns
        -------
        - pairs : List[Tuple[Path, Path]]
            Ordered list of (compressed_path, raw_path) tuples.
        """
        pairs = []
        for comp_path in sorted(self.compressed_dir.glob("*.png")):
            raw_id = comp_path.stem.replace(self.suffix, "")
            for ext in [".jpg", ".jpeg", ".png"]:
                raw_path = self.input_dir / (raw_id + ext)
                if raw_path.exists():
                    pairs.append((comp_path, raw_path))
                    break
        return pairs

    def __len__(self) -> int:
        """
        Returns the total number of pairs in the dataset.

        Required by PyTorch's Dataset interface — used internally
        by DataLoader to know how many samples exist.

        Returns
        -------
        - length : int
            Number of (compressed, original) pairs available.
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and returns the pair at the given index as tensors.

        Called automatically by DataLoader at each iteration.
        Opens both images in RGB mode (ensures 3 channels even for
        grayscale sources), applies the same transform to both,
        and returns them as a tuple.

        Parameters
        ----------
        - idx : int
            Index of the pair to retrieve (0-based).

        Returns
        -------
        - compressed : torch.Tensor
            Compressed image tensor of shape (3, image_size, image_size),
            dtype float32, values in [0.0, 1.0]. This is the network input.
        - original : torch.Tensor
            Original image tensor of shape (3, image_size, image_size),
            dtype float32, values in [0.0, 1.0]. This is the training target.
        """
        
        comp_path, raw_path = self.pairs[idx]
        compressed = Image.open(comp_path).convert("RGB")
        original = Image.open(raw_path).convert("RGB")
        
        return self.transform(compressed), self.transform(original)

    def get_pair_names(self, idx: int) -> Tuple[str, str]:
        """
        Returns the filenames (not full paths) of a pair.

        Useful in notebooks for displaying which files are being
        visualized or evaluated, without exposing full system paths.

        Parameters
        ----------
        - idx : int
            Index of the pair to retrieve (0-based).

        Returns
        -------
        - comp_name : str
            Filename of the compressed image (e.g. "100007_fourier_25.png").
        - raw_name : str
            Filename of the original image (e.g. "100007.jpg").
        """
        
        comp_path, raw_path = self.pairs[idx]
        
        return comp_path.name, raw_path.name


def split_dataset(
    dataset: CompressionDataset,
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Randomly splits a CompressionDataset into train, val, and test subsets.

    Shuffles the dataset indices with a fixed seed for reproducibility,
    then partitions them according to the given proportions.
    Returns three torch.utils.data.Subset objects that reference the
    original dataset — no data is copied or duplicated.

    Parameters
    ----------
    - dataset : CompressionDataset
        The full dataset to split.
    - splits : Tuple[float, float, float], optional
        Proportions for (train, val, test). Must sum to 1.0.
    - seed : int, optional
        Random seed for reproducible shuffling.

    Returns
    -------
    - train_set : torch.utils.data.Subset
        Subset used for training (~70% of data).
    - val_set : torch.utils.data.Subset
        Subset used for validation during training (~15% of data).
    - test_set : torch.utils.data.Subset
        Subset used for final evaluation after training (~15% of data).
    """
    
    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n * splits[0])
    n_val = int(n * splits[1])

    return (
        Subset(dataset, indices[:n_train]),
        Subset(dataset, indices[n_train : n_train + n_val]),
        Subset(dataset, indices[n_train + n_val :]),
    )


def get_dataloaders(
    dataset: CompressionDataset,
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds and returns the three DataLoaders for train, val, and test.

    Internally calls split_dataset, then wraps each Subset in a
    DataLoader with the appropriate settings:
    - train_loader: shuffle=True  (different order each epoch)
    - val_loader:   shuffle=False (deterministic evaluation)
    - test_loader:  shuffle=False (deterministic evaluation)

    Parameters
    ----------
    - dataset : CompressionDataset
        The full dataset to split and load.
    - splits : Tuple[float, float, float], optional
        Proportions for (train, val, test). Default: (0.70, 0.15, 0.15).
    - batch_size : int, optional
        Number of samples per batch. Default: 8.
    - seed : int, optional
        Random seed passed to split_dataset. Default: 42.
    - num_workers : int, optional
        Number of subprocesses for parallel data loading.
        Set to 0 on Windows or if multiprocessing errors occur.
        Default: 2.

    Returns
    -------
    - train_loader : torch.utils.data.DataLoader
        DataLoader over the training subset, with shuffling enabled.
    - val_loader : torch.utils.data.DataLoader
        DataLoader over the validation subset.
    - test_loader : torch.utils.data.DataLoader
        DataLoader over the test subset.
    """
    train_set, val_set, test_set = split_dataset(dataset, splits, seed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
