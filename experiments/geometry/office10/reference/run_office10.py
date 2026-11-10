#!/usr/bin/env python3
"""Office-Caltech-10 version of the federated prototype-clustering experiment.

Expected directory layout (domain names are case-insensitive):

    data/office10/
      amazon/<class>/*.jpg
      caltech/<class>/*.jpg
      dslr/<class>/*.jpg
      webcam/<class>/*.jpg

The loader also accepts an intermediate office_caltech_10, OfficeCaltech10,
or office-caltech-10 directory. A deterministic stratified 70/30 train/test
split is created because Office-Caltech-10 is commonly distributed without an
official split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import run_digits5 as core


CLASS_NAMES = (
    "backpack", "bike", "calculator", "headphones", "keyboard",
    "laptop_computer", "monitor", "mouse", "mug", "projector",
)
DOMAIN_NAMES = ("amazon", "caltech", "dslr", "webcam")
GROUP_A = (0, 1, 2, 3, 4)
GROUP_B = (5, 6, 7, 8, 9)
PROFILE_NAMES = ("group_a_heavy", "group_b_heavy")
GROUP_NAMES = ("classes 0–4", "classes 5–9")
DOMAIN_HATCHES = ("o", "xx", "//", "\\\\")


class StratifiedImageFolderSplit(Dataset):
    def __init__(self, folder: Path, split: str, train_fraction: float = 0.7,
                 seed: int = 2026):
        full = datasets.ImageFolder(folder)
        if tuple(full.classes) != tuple(sorted(full.classes)):
            raise RuntimeError("ImageFolder class order is not deterministic.")
        if len(full.classes) != 10:
            raise ValueError(
                f"Expected 10 classes in {folder}, found {len(full.classes)}: "
                f"{full.classes}"
            )
        selected = []
        targets = np.asarray(full.targets)
        for class_id in range(len(full.classes)):
            indices = np.flatnonzero(targets == class_id)
            rng = np.random.default_rng(seed + class_id)
            rng.shuffle(indices)
            cut = int(round(len(indices) * train_fraction))
            cut = min(max(cut, 1), max(len(indices) - 1, 1))
            chosen = indices[:cut] if split == "train" else indices[cut:]
            selected.extend(chosen.tolist())
        self.samples = [full.samples[i] for i in selected]
        self.targets = [label for _, label in self.samples]
        self.classes = full.classes
        self.class_to_idx = full.class_to_idx
        self.loader = full.loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self.loader(path)
        return image, label


def _find_domain_folder(root: Path, domain: str) -> Path:
    parents = (
        root,
        root / "office_caltech_10",
        root / "OfficeCaltech10",
        root / "office-caltech-10",
    )
    aliases = {
        "amazon": ("amazon", "Amazon"),
        "caltech": ("caltech", "Caltech", "caltech10", "Caltech10"),
        "dslr": ("dslr", "DSLR"),
        "webcam": ("webcam", "Webcam"),
    }[domain]
    for parent in parents:
        for alias in aliases:
            candidate = parent / alias
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(
        f"Could not find the {domain!r} domain below {root}. See "
        "README_OFFICE10.md for the expected folder layout."
    )


def load_domain_split(root: Path, domain: str, split: str) -> Dataset:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    folder = _find_domain_folder(Path(root), domain)
    domain_seed = 2026 + DOMAIN_NAMES.index(domain) * 100
    return StratifiedImageFolderSplit(folder, split, seed=domain_seed)


def office_transform(domain_id: int, image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


def build_office_client_specs(
    targets: Sequence[int], clients_per_domain: int, samples_per_client: int,
    majority_fraction: float, seed: int, domain_id: int = 0,
) -> List[core.ClientSpec]:
    """Create disjoint clients with an exact group-level majority ratio.

    Unlike Digits-5, the small Office domains cannot guarantee identical
    per-class counts. Samples are balanced across the available classes inside
    each five-class group while preserving the requested 70/30 group split.
    """
    if clients_per_domain % 2:
        raise ValueError("--clients-per-domain must be even.")
    rng = np.random.default_rng(seed)
    target_array = np.asarray(targets)
    pools: Dict[int, List[int]] = {}
    offsets = {class_id: 0 for class_id in range(len(CLASS_NAMES))}
    for class_id in range(len(CLASS_NAMES)):
        pool = np.flatnonzero(target_array == class_id).tolist()
        rng.shuffle(pool)
        pools[class_id] = pool

    def take_from_group(group: Sequence[int], amount: int,
                        client_counts: List[int]) -> List[int]:
        selected: List[int] = []
        for _ in range(amount):
            available = [c for c in group if offsets[c] < len(pools[c])]
            if not available:
                group_names = [CLASS_NAMES[c] for c in group]
                raise ValueError(
                    f"Not enough unused {DOMAIN_NAMES[domain_id]} images in "
                    f"class group {group_names}. Reduce --samples-per-client "
                    "or --clients-per-domain."
                )
            jitter = {c: float(rng.random()) for c in available}
            chosen = min(
                available,
                key=lambda c: (
                    client_counts[c],
                    -(len(pools[c]) - offsets[c]),
                    jitter[c],
                ),
            )
            selected.append(pools[chosen][offsets[chosen]])
            offsets[chosen] += 1
            client_counts[chosen] += 1
        return selected

    specs: List[core.ClientSpec] = []
    for local_id in range(clients_per_domain):
        profile_id = 0 if local_id < clients_per_domain // 2 else 1
        majority_group = GROUP_A if profile_id == 0 else GROUP_B
        minority_group = GROUP_B if profile_id == 0 else GROUP_A
        majority_count = round(samples_per_client * majority_fraction)
        minority_count = samples_per_client - majority_count
        class_counts = [0] * len(CLASS_NAMES)
        indices = take_from_group(majority_group, majority_count, class_counts)
        indices.extend(take_from_group(minority_group, minority_count, class_counts))
        rng.shuffle(indices)
        specs.append(core.ClientSpec(
            local_id, domain_id, profile_id, indices, class_counts
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
    core.domain_transform = office_transform
    core.build_client_specs = build_office_client_specs


def _with_office_defaults(argv: Optional[Sequence[str]]) -> Optional[list[str]]:
    if argv is None:
        return None
    args = list(argv)
    defaults = {
        "--data-dir": "data/office10",
        "--output-dir": "outputs/office10_main",
        "--clients-per-domain": "4",
        "--samples-per-client": "20",
        "--majority-fraction": "0.7",
        "--image-size": "224",
        "--batch-size": "16",
        "--prototype-mix": "0.10",
        "--validation-fraction": "0.20",
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
    core.main(_with_office_defaults(argv))


if __name__ == "__main__":
    main()
