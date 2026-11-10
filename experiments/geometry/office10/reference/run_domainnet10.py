#!/usr/bin/env python3
"""DomainNet-10 experiment using all six visual domains.

The ten fixed classes form two interpretable prevalence groups:
five animals and five vehicles. The loader supports the official DomainNet
``<domain>_<split>.txt`` files and conventional ImageFolder layouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import run_digits5 as core


ANIMAL_CLASSES = ("bird", "cat", "dog", "elephant", "horse")
VEHICLE_CLASSES = ("airplane", "bicycle", "car", "motorbike", "truck")
CLASS_NAMES = ANIMAL_CLASSES + VEHICLE_CLASSES
DOMAIN_NAMES = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
GROUP_A = tuple(range(5))
GROUP_B = tuple(range(5, 10))
PROFILE_NAMES = ("animal_heavy", "vehicle_heavy")
GROUP_NAMES = ("animals", "vehicles")
DOMAIN_HATCHES = ("o", "xx", "//", "\\\\", "..", "++")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalise(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


def _dataset_root(root: Path) -> Path:
    for candidate in (root, root / "DomainNet", root / "domainnet"):
        names = {child.name.lower() for child in candidate.iterdir()} if candidate.is_dir() else set()
        if all(domain in names for domain in DOMAIN_NAMES):
            return candidate
    raise FileNotFoundError(
        f"Could not find all six DomainNet domains below {root}. "
        "Expected clipart, infograph, painting, quickdraw, real and sketch."
    )


def _domain_folder(dataset_root: Path, domain: str) -> Path:
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name.lower() == domain:
            return child
    raise FileNotFoundError(f"Missing DomainNet domain: {domain}")


def _split_file(dataset_root: Path, domain: str, split: str) -> Optional[Path]:
    names = (f"{domain}_{split}.txt", f"{domain}_{split}_mini.txt")
    parents = (
        dataset_root, dataset_root / "splits", dataset_root / "txt",
        dataset_root.parent / "splits", dataset_root.parent / "txt",
    )
    for parent in parents:
        for name in names:
            candidate = parent / name
            if candidate.is_file():
                return candidate
    return None


def _resolve_image(dataset_root: Path, domain_folder: Path, relative: str) -> Path:
    relative_path = Path(relative.replace("\\", "/"))
    candidates = [
        dataset_root / relative_path,
        dataset_root.parent / relative_path,
        domain_folder / relative_path,
    ]
    parts = relative_path.parts
    if parts and parts[0].lower() == domain_folder.name.lower():
        candidates.append(domain_folder.joinpath(*parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Split entry does not resolve to an image: {relative}")


class DomainNet10Dataset(Dataset):
    def __init__(self, root: Path, domain: str, split: str, seed: int = 2026):
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        dataset_root = _dataset_root(Path(root))
        domain_folder = _domain_folder(dataset_root, domain)
        class_to_index = {_normalise(name): i for i, name in enumerate(CLASS_NAMES)}
        split_file = _split_file(dataset_root, domain, split)
        samples: List[tuple[Path, int]] = []

        if split_file is not None:
            for line_number, line in enumerate(split_file.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                relative = line.rsplit(maxsplit=1)[0]
                parts = Path(relative.replace("\\", "/")).parts
                if len(parts) < 2:
                    continue
                class_name = _normalise(parts[-2])
                if class_name not in class_to_index:
                    continue
                image = _resolve_image(dataset_root, domain_folder, relative)
                samples.append((image, class_to_index[class_name]))
        else:
            # Accept domain/train/class, domain/test/class, or domain/class.
            split_root = domain_folder / split
            explicit_split = split_root.is_dir()
            search_root = split_root if explicit_split else domain_folder
            all_samples: Dict[int, List[Path]] = {i: [] for i in range(len(CLASS_NAMES))}
            folder_map = {
                _normalise(child.name): child
                for child in search_root.iterdir() if child.is_dir()
            }
            missing = [name for name in CLASS_NAMES if _normalise(name) not in folder_map]
            if missing:
                raise FileNotFoundError(
                    f"{domain} is missing selected classes {missing} below {search_root}."
                )
            for class_name, class_id in class_to_index.items():
                folder = folder_map[class_name]
                all_samples[class_id] = sorted(
                    path for path in folder.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            if explicit_split:
                samples = [
                    (path, class_id)
                    for class_id, paths in all_samples.items() for path in paths
                ]
            else:
                # Deterministic 80/20 fallback when official split files are absent.
                for class_id, paths in all_samples.items():
                    indices = np.arange(len(paths))
                    rng = np.random.default_rng(seed + class_id)
                    rng.shuffle(indices)
                    cut = int(round(0.8 * len(indices)))
                    cut = min(max(cut, 1), max(len(indices) - 1, 1))
                    chosen = indices[:cut] if split == "train" else indices[cut:]
                    samples.extend((paths[int(index)], class_id) for index in chosen)

        if not samples:
            raise ValueError(f"No selected DomainNet-10 samples found for {domain}/{split}.")
        self.samples = [(str(path), int(label)) for path, label in samples]
        self.targets = [label for _, label in self.samples]
        self.classes = list(CLASS_NAMES)
        self.loader = datasets.folder.default_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        return self.loader(path), label


def load_domain_split(root: Path, domain: str, split: str) -> Dataset:
    return DomainNet10Dataset(root, domain, split, seed=2026 + 100 * DOMAIN_NAMES.index(domain))


def domainnet_transform(domain_id: int, image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def build_domainnet_client_specs(
    targets: Sequence[int], clients_per_domain: int, samples_per_client: int,
    majority_fraction: float, seed: int, domain_id: int = 0,
) -> List[core.ClientSpec]:
    """Construct disjoint animal-heavy and vehicle-heavy clients."""
    if clients_per_domain % 2:
        raise ValueError("--clients-per-domain must be even.")
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets, dtype=np.int64)
    pools: Dict[int, List[int]] = {}
    offsets = {class_id: 0 for class_id in range(len(CLASS_NAMES))}
    for class_id in range(len(CLASS_NAMES)):
        pool = np.flatnonzero(targets == class_id).tolist()
        rng.shuffle(pool)
        pools[class_id] = pool

    specs: List[core.ClientSpec] = []
    for local_id in range(clients_per_domain):
        profile_id = 0 if local_id < clients_per_domain // 2 else 1
        counts = core.requested_counts(samples_per_client, profile_id, majority_fraction)
        selected: List[int] = []
        for class_id, count in enumerate(counts):
            start, end = offsets[class_id], offsets[class_id] + count
            if end > len(pools[class_id]):
                raise ValueError(
                    f"Not enough {DOMAIN_NAMES[domain_id]} examples for "
                    f"class {CLASS_NAMES[class_id]!r}: need {end}, have "
                    f"{len(pools[class_id])}. Reduce --samples-per-client."
                )
            selected.extend(pools[class_id][start:end])
            offsets[class_id] = end
        rng.shuffle(selected)
        specs.append(core.ClientSpec(
            local_id, domain_id, profile_id, selected, counts
        ))
    flat = [index for spec in specs for index in spec.indices]
    if len(flat) != len(set(flat)):
        raise RuntimeError("Data leakage: an image was assigned to multiple clients.")
    return specs


def _configure_core() -> None:
    core.CLASS_NAMES = CLASS_NAMES
    core.LOW_DIGITS = GROUP_A
    core.HIGH_DIGITS = GROUP_B
    core.DOMAIN_NAMES = DOMAIN_NAMES
    core.PROFILE_NAMES = PROFILE_NAMES
    core.GROUP_NAMES = GROUP_NAMES
    core.DOMAIN_HATCHES = DOMAIN_HATCHES
    core.load_domain_split = load_domain_split
    core.domain_transform = domainnet_transform
    core.build_client_specs = build_domainnet_client_specs


def _with_defaults(argv: Optional[Sequence[str]]) -> Optional[List[str]]:
    if argv is None:
        return None
    args = list(argv)
    defaults = {
        "--data-dir": "data/domainnet",
        "--output-dir": "outputs/domainnet10_main",
        "--clients-per-domain": "4",
        "--samples-per-client": "100",
        "--majority-fraction": "0.70",
        "--image-size": "224",
        "--batch-size": "32",
        "--prototype-mix": "0.10",
        "--validation-fraction": "0.10",
        "--cluster-min-rounds": "5",
        "--cluster-max-rounds": "30",
        "--cluster-patience": "5",
        "--cluster-monitor-window": "3",
    }
    for flag, value in defaults.items():
        if flag not in args:
            args.extend([flag, value])
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    _configure_core()
    core.main(_with_defaults(argv))


if __name__ == "__main__":
    main()
