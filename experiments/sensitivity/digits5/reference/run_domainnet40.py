#!/usr/bin/env python3
"""DomainNet-40 federated semantic clustering experiment.

The runner automatically selects 20 well-supported natural/living classes and
20 well-supported manufactured/mobility classes from the official DomainNet
training lists.  Four disjoint clients are created per domain.  The first half
are 70/30 natural-heavy and the second half are 30/70 manufactured-heavy.

Unlike the DomainNet-10 runner, allocation is pooled within each semantic
group.  A rare individual class therefore cannot prevent clients from having
1,000 distinct images, while every selected class is represented whenever the
requested quota permits it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import run_digits5 as core
import run_domainnet10 as base


DOMAIN_NAMES = base.DOMAIN_NAMES
DOMAIN_HATCHES = base.DOMAIN_HATCHES
PROFILE_NAMES = ("natural_heavy", "manufactured_heavy")
GROUP_NAMES = ("natural_living", "manufactured_mobility")

# All names are official DomainNet/QuickDraw category names.  We select the
# best-supported 20 from each candidate bank using the six official train
# lists, so the final 40-class task adapts to the installed cleaned release.
NATURAL_CANDIDATES = (
    "ant", "apple", "asparagus", "banana", "bat", "bear", "bee", "bird",
    "blackberry", "blueberry", "broccoli", "bush", "butterfly", "cactus",
    "camel", "carrot", "cat", "cow", "crab", "crocodile", "dog",
    "dolphin", "dragon", "duck", "elephant", "fish", "flamingo", "flower",
    "frog", "garden", "giraffe", "grapes", "grass", "hedgehog", "horse",
    "kangaroo", "leaf", "lion", "lobster", "monkey", "mosquito", "mouse",
    "mushroom", "octopus", "onion", "owl", "panda", "parrot", "peanut",
    "pear", "penguin", "pig", "pineapple", "potato", "rabbit", "raccoon",
    "rhinoceros", "scorpion", "sea turtle", "shark", "sheep", "snail",
    "snake", "spider", "squirrel", "strawberry", "swan", "tiger", "tree",
    "watermelon", "whale", "zebra",
)

MANUFACTURED_CANDIDATES = (
    "airplane", "alarm clock", "ambulance", "anvil", "axe", "backpack",
    "baseball bat", "basket", "bicycle", "binoculars", "book", "boomerang",
    "bottlecap", "bowtie", "bracelet", "bridge", "broom", "bucket",
    "bulldozer", "bus", "calculator", "camera", "cannon", "canoe", "car",
    "cell phone", "chair", "clock", "coffee cup", "compass", "computer",
    "cooler", "couch", "crayon", "crown", "cruise ship", "cup", "diamond",
    "dishwasher", "diving board", "drill", "drums", "fire hydrant",
    "firetruck", "flashlight", "fork", "frying pan", "guitar", "hammer",
    "headphones", "helicopter", "hot air balloon", "keyboard", "knife",
    "ladder", "laptop", "light bulb", "motorbike", "mug", "pickup truck",
    "pliers", "police car", "projector", "sailboat", "school bus",
    "screwdriver", "skateboard", "speedboat", "submarine", "suitcase",
    "telephone", "television", "tractor", "train", "truck", "van", "wheel",
)

CLASS_NAMES: Tuple[str, ...] = ()
GROUP_A: Tuple[int, ...] = ()
GROUP_B: Tuple[int, ...] = ()


def _argument(argv: Optional[Sequence[str]], flag: str, default: str) -> str:
    if argv is None:
        return default
    values = list(argv)
    if flag not in values:
        return default
    position = values.index(flag)
    if position + 1 >= len(values):
        raise ValueError(f"{flag} requires a value")
    return values[position + 1]


def _training_counts(root: Path) -> Dict[str, Counter]:
    dataset_root = base._dataset_root(Path(root))
    counts: Dict[str, Counter] = {}
    for domain in DOMAIN_NAMES:
        split_file = base._split_file(dataset_root, domain, "train")
        if split_file is None:
            raise FileNotFoundError(
                f"DomainNet-40 requires the official {domain}_train.txt file."
            )
        counter: Counter = Counter()
        for line in split_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            relative = line.rsplit(maxsplit=1)[0]
            parts = Path(relative.replace("\\", "/")).parts
            if len(parts) >= 2:
                counter[base._normalise(parts[-2])] += 1
        counts[domain] = counter
    return counts


def _select_group(
    candidates: Sequence[str], counts: Dict[str, Counter], size: int,
) -> Tuple[str, ...]:
    unique = {base._normalise(name): name for name in candidates}
    scored = []
    for normalised, display_name in unique.items():
        per_domain = [counts[domain][normalised] for domain in DOMAIN_NAMES]
        if min(per_domain) > 0:
            # Maximise worst-domain support first and total support second.
            scored.append((min(per_domain), sum(per_domain), display_name))
    scored.sort(reverse=True)
    if len(scored) < size:
        raise ValueError(
            f"Only {len(scored)} candidate classes occur in every domain; "
            f"need {size}."
        )
    return tuple(item[2] for item in scored[:size])


def select_domainnet40_classes(root: Path) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    counts = _training_counts(root)
    natural = _select_group(NATURAL_CANDIDATES, counts, 20)
    manufactured = _select_group(MANUFACTURED_CANDIDATES, counts, 20)
    return natural, manufactured


class DomainNet40Dataset(base.DomainNet10Dataset):
    pass


def load_domain_split(root: Path, domain: str, split: str):
    return DomainNet40Dataset(
        root, domain, split, seed=2026 + 100 * DOMAIN_NAMES.index(domain)
    )


def _take_from_group(
    pools: Dict[int, List[int]], class_ids: Sequence[int], amount: int,
    rng: np.random.Generator,
) -> List[int]:
    available_total = sum(len(pools[class_id]) for class_id in class_ids)
    if available_total < amount:
        raise ValueError(
            f"Semantic group has only {available_total} unused images but "
            f"{amount} are required. Expand the candidate bank or reduce "
            "--samples-per-client."
        )

    selected: List[int] = []
    # Guarantee coverage before availability-weighted pooled sampling.
    if amount >= len(class_ids):
        for class_id in class_ids:
            if pools[class_id]:
                selected.append(pools[class_id].pop())

    while len(selected) < amount:
        available = [class_id for class_id in class_ids if pools[class_id]]
        weights = np.asarray([len(pools[class_id]) for class_id in available], dtype=float)
        weights /= weights.sum()
        chosen = int(rng.choice(available, p=weights))
        selected.append(pools[chosen].pop())
    return selected


def build_domainnet40_client_specs(
    targets: Sequence[int], clients_per_domain: int, samples_per_client: int,
    majority_fraction: float, seed: int, domain_id: int = 0,
) -> List[core.ClientSpec]:
    if clients_per_domain % 2:
        raise ValueError("--clients-per-domain must be even.")
    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets, dtype=np.int64)
    pools: Dict[int, List[int]] = {}
    for class_id in range(len(CLASS_NAMES)):
        pool = np.flatnonzero(targets_array == class_id).tolist()
        rng.shuffle(pool)
        pools[class_id] = pool

    half = clients_per_domain // 2
    majority = round(samples_per_client * majority_fraction)
    minority = samples_per_client - majority
    required_a = half * majority + half * minority
    required_b = required_a
    available_a = sum(len(pools[c]) for c in GROUP_A)
    available_b = sum(len(pools[c]) for c in GROUP_B)
    if available_a < required_a or available_b < required_b:
        raise ValueError(
            f"{DOMAIN_NAMES[domain_id]} cannot supply disjoint 1,000-sample "
            f"clients: natural {available_a}/{required_a}, manufactured "
            f"{available_b}/{required_b}."
        )

    specs: List[core.ClientSpec] = []
    for local_id in range(clients_per_domain):
        profile_id = 0 if local_id < half else 1
        amount_a = majority if profile_id == 0 else minority
        amount_b = samples_per_client - amount_a
        selected = _take_from_group(pools, GROUP_A, amount_a, rng)
        selected += _take_from_group(pools, GROUP_B, amount_b, rng)
        rng.shuffle(selected)
        class_counts = np.bincount(
            targets_array[selected], minlength=len(CLASS_NAMES)
        ).astype(int).tolist()
        specs.append(core.ClientSpec(
            local_id, domain_id, profile_id, selected, class_counts
        ))

    assigned = [index for spec in specs for index in spec.indices]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("Data leakage: an image was assigned to multiple clients.")
    return specs


def _configure_core(natural: Sequence[str], manufactured: Sequence[str]) -> None:
    global CLASS_NAMES, GROUP_A, GROUP_B
    CLASS_NAMES = tuple(natural) + tuple(manufactured)
    GROUP_A = tuple(range(len(natural)))
    GROUP_B = tuple(range(len(natural), len(CLASS_NAMES)))

    # The inherited loader reads these globals from run_domainnet10.
    base.CLASS_NAMES = CLASS_NAMES
    core.CLASS_NAMES = CLASS_NAMES
    core.LOW_DIGITS = GROUP_A
    core.HIGH_DIGITS = GROUP_B
    core.DOMAIN_NAMES = DOMAIN_NAMES
    core.PROFILE_NAMES = PROFILE_NAMES
    core.GROUP_NAMES = GROUP_NAMES
    core.DOMAIN_HATCHES = DOMAIN_HATCHES
    core.load_domain_split = load_domain_split
    core.domain_transform = base.domainnet_transform
    core.build_client_specs = build_domainnet40_client_specs


def preflight(
    root: Path, clients_per_domain: int = 4, samples_per_client: int = 1000,
    validation_fraction: float = 0.10,
) -> Dict[str, object]:
    natural, manufactured = select_domainnet40_classes(root)
    _configure_core(natural, manufactured)
    required_train = clients_per_domain * samples_per_client
    required_validation = clients_per_domain * max(
        1, round(samples_per_client * validation_fraction)
    )
    report: Dict[str, object] = {
        "natural_classes": list(natural),
        "manufactured_classes": list(manufactured),
        "domains": {},
    }
    feasible = True
    for domain_id, domain in enumerate(DOMAIN_NAMES):
        dataset = load_domain_split(root, domain, "train")
        counts = np.bincount(dataset.targets, minlength=40)
        available = int(counts.sum())
        natural_available = int(counts[:20].sum())
        manufactured_available = int(counts[20:].sum())
        group_train = required_train // 2
        group_reserve = required_validation // 2
        domain_ok = (
            available >= required_train + required_validation
            and natural_available >= group_train + group_reserve
            and manufactured_available >= group_train + group_reserve
        )
        feasible = feasible and domain_ok
        report["domains"][domain] = {
            "available": available,
            "required_train": required_train,
            "required_validation": required_validation,
            "natural_available": natural_available,
            "manufactured_available": manufactured_available,
            "required_per_group": group_train + group_reserve,
            "feasible": domain_ok,
        }
    report["feasible"] = feasible
    return report


def _with_defaults(argv: Optional[Sequence[str]]) -> Optional[List[str]]:
    args = list(sys.argv[1:] if argv is None else argv)
    defaults = {
        "--data-dir": "data/domainnet",
        "--output-dir": "outputs/domainnet40_main",
        "--clients-per-domain": "4",
        "--samples-per-client": "1000",
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
    effective = _with_defaults(argv)
    data_root = Path(_argument(effective, "--data-dir", "data/domainnet"))
    clients = int(_argument(effective, "--clients-per-domain", "4"))
    samples = int(_argument(effective, "--samples-per-client", "1000"))
    validation = float(_argument(effective, "--validation-fraction", "0.10"))
    report = preflight(data_root, clients, samples, validation)
    print("Selected natural/living classes:", report["natural_classes"])
    print("Selected manufactured/mobility classes:", report["manufactured_classes"])
    print("DomainNet-40 preflight:", report["domains"])
    if not report["feasible"]:
        raise ValueError(
            "The selected DomainNet-40 task cannot supply the requested "
            "disjoint training and validation samples in every domain."
        )
    core.main(effective)


if __name__ == "__main__":
    main()
