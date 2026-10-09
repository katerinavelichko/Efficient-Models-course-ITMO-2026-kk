import os
import tarfile
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.24703223, 0.24348512, 0.26158784)


class IndexedImages(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx]), idx


def _transforms(train: bool):
    ops = []
    if train:
        ops += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(ops)


def _synthetic(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, size=n, dtype=np.int64)
    return images, labels


_CIFAR_DIR = "cifar-10-batches-py"
_CIFAR_ARCHIVE = "cifar-10-python.tar.gz"
_CIFAR_FILES = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]


def _cifar_ready(root: str) -> bool:
    folder = os.path.join(root, _CIFAR_DIR)
    return all(os.path.isfile(os.path.join(folder, name)) for name in _CIFAR_FILES)


def _extract_local_archive(root: str) -> None:
    archive = os.path.join(root, _CIFAR_ARCHIVE)
    if not os.path.isfile(archive):
        return
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)


def _load_cifar(root: str):
    os.makedirs(root, exist_ok=True)
    if not _cifar_ready(root):
        _extract_local_archive(root)
    if _cifar_ready(root):
        train = datasets.CIFAR10(root, train=True, download=False)
        test = datasets.CIFAR10(root, train=False, download=False)
        return train, test
    try:
        train = datasets.CIFAR10(root, train=True, download=True)
        test = datasets.CIFAR10(root, train=False, download=True)
        return train, test
    except Exception as exc:
        folder = os.path.abspath(os.path.join(root, _CIFAR_DIR))
        raise RuntimeError(
            "CIFAR-10 download failed"
        ) from exc


def make_datasets(root: str, synthetic: bool, synthetic_n: int, seed: int):
    train_tf = _transforms(train=True)
    eval_tf = _transforms(train=False)
    if synthetic:
        train_x, train_y = _synthetic(synthetic_n, seed)
        test_x, test_y = _synthetic(max(64, synthetic_n // 4), seed + 17)
    else:
        train, test = _load_cifar(root)
        train_x, train_y = train.data, train.targets
        test_x, test_y = test.data, test.targets
    return (
        IndexedImages(train_x, train_y, train_tf),
        IndexedImages(train_x, train_y, eval_tf),
        IndexedImages(test_x, test_y, eval_tf),
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: Optional[int] = None,
) -> DataLoader:
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def subset_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: Optional[int] = None,
) -> DataLoader:
    subset = Subset(dataset, [int(i) for i in indices])
    return make_loader(subset, batch_size, shuffle, workers, seed)
