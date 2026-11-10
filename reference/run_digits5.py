#!/usr/bin/env python3
"""Digits-5 federated prototype clustering under natural domain shift."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = tuple(str(i) for i in range(10))
LOW_DIGITS = (0, 1, 2, 3, 4)
HIGH_DIGITS = (5, 6, 7, 8, 9)
DOMAIN_NAMES = ("mnist", "svhn", "usps", "synth", "mnistm")
PROFILE_NAMES = ("low_digit_heavy", "high_digit_heavy")
GROUP_NAMES = ("digits 0–4", "digits 5–9")
DOMAIN_HATCHES = ("o", "xx", "//", "\\\\", "..")
PROFILE_COLOURS = ("#377eb8", "#e41a1c")


@dataclass
class ClientSpec:
    client_id: int
    domain_id: int
    profile_id: int
    indices: List[int]
    class_counts: List[int]


@dataclass
class ClassStat:
    prototype: np.ndarray
    count: int
    dispersion: float
    raw_support: float = 0.0
    reliability: float = 0.0
    tempered_support: float = 0.0


@dataclass
class GraphPrototypeBank:
    """Server-side class graphs represented as dense, small tensors."""
    raw_nodes: torch.Tensor       # [classes, domains, feature_dim]
    smooth_nodes: torch.Tensor    # [classes, domains, feature_dim]
    mask: torch.Tensor            # [classes, domains]
    reliability: torch.Tensor     # [classes, domains]
    edge_weights: torch.Tensor    # [classes, domains, domains]
    centroids: torch.Tensor       # [classes, feature_dim]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AddGaussianNoise:
    def __init__(self, sigma: float = 0.08, deterministic: bool = False):
        self.sigma = sigma
        self.deterministic = deterministic

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.deterministic:
            generator = torch.Generator(device=tensor.device).manual_seed(314159)
            noise = torch.randn(tensor.shape, generator=generator,
                                device=tensor.device, dtype=tensor.dtype)
        else:
            noise = torch.randn_like(tensor)
        return torch.clamp(tensor + noise * self.sigma, 0, 1)


class RandomResolution:
    """Randomly remove spatial detail, then restore the requested input size."""
    def __init__(self, image_size: int, choices: Sequence[int] = (8, 12, 16, 24)):
        self.image_size = image_size
        self.choices = tuple(choices)

    def __call__(self, image):
        size = random.choice(self.choices)
        image = TF.resize(image, [size, size], transforms.InterpolationMode.BILINEAR)
        return TF.resize(image, [self.image_size, self.image_size],
                         transforms.InterpolationMode.BILINEAR)


def domain_transform(domain_id: int, image_size: int, train: bool) -> transforms.Compose:
    # Keep each dataset's natural appearance; horizontal flips corrupt digits.
    ops: List[object] = [transforms.Resize((image_size, image_size)),
                         transforms.Lambda(lambda image: image.convert("RGB"))]
    if train:
        ops.append(transforms.RandomAffine(
            degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)
        ))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(ops)


class SerializedDigits(Dataset):
    """Load a trusted .pt/.pkl containing (images, labels) or a matching dict."""
    def __init__(self, path: Path):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            import pickle
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        if isinstance(payload, dict):
            image_key = next(k for k in ("images", "data", "x", "X") if k in payload)
            label_key = next(k for k in ("labels", "targets", "y", "Y") if k in payload)
            images, labels = payload[image_key], payload[label_key]
        elif isinstance(payload, (tuple, list)) and len(payload) >= 2:
            images, labels = payload[:2]
        else:
            raise ValueError(f"Unsupported serialized dataset layout: {path}")
        self.images = np.asarray(images)
        self.targets = np.asarray(labels, dtype=np.int64).reshape(-1).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        array = np.asarray(self.images[index])
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if array.dtype != np.uint8:
            scale = 255.0 if float(array.max()) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array), int(self.targets[index])


def load_domain_split(root: Path, domain: str, split: str) -> Dataset:
    aliases = {"mnist": ("mnist", "MNIST"), "svhn": ("svhn", "SVHN"),
               "usps": ("usps", "USPS"), "synth": ("synth", "SynthDigits"),
               "mnistm": ("mnistm", "MNISTM", "MNIST_M")}[domain]
    for alias in aliases:
        folder = root / alias / split
        if folder.is_dir():
            return datasets.ImageFolder(folder)
        for suffix in ("pt", "pkl", "pickle"):
            path = root / alias / f"{split}.{suffix}"
            if path.exists():
                return SerializedDigits(path)
    # Torchvision provides these three domains and handles verified downloads.
    train = split == "train"
    if domain == "mnist":
        return datasets.MNIST(root / "torchvision", train=train, download=True)
    if domain == "svhn":
        return datasets.SVHN(root / "torchvision", split=split, download=True)
    if domain == "usps":
        return datasets.USPS(root / "torchvision", train=train, download=True)
    raise FileNotFoundError(
        f"Missing {domain}/{split} below {root}. Use class folders or a trusted "
        f"{split}.pt/.pkl containing images and labels. See README_DIGITS5.md."
    )


class ClientDataset(Dataset):
    def __init__(self, base: Dataset, spec: ClientSpec, image_size: int,
                 train: bool, two_views: bool = False):
        self.base = base
        self.spec = spec
        self.two_views = two_views
        self.transform = domain_transform(spec.domain_id, image_size, train)

    def __len__(self) -> int:
        return len(self.spec.indices)

    def __getitem__(self, item: int):
        image, label = self.base[self.spec.indices[item]]
        if self.two_views:
            return self.transform(image), self.transform(image), int(label), self.spec.domain_id
        return self.transform(image), int(label), self.spec.domain_id


def build_extra_validation_specs(
    base: Dataset, client_specs: Sequence[ClientSpec], samples_per_client: int,
    seed: int,
) -> List[ClientSpec]:
    """Assign extra, unused domain samples as disjoint client validation data.

    Training clients keep every originally assigned sample. Validation examples
    come only from the unused portion of the same domain's training split and
    approximately follow each client's class profile.
    """
    rng = np.random.default_rng(seed)
    targets = getattr(base, "targets", getattr(base, "labels", None))
    if targets is None:
        raise AttributeError("Validation allocation requires dataset targets/labels.")
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    used_training = {index for spec in client_specs for index in spec.indices}
    pools: Dict[int, List[int]] = {}
    for class_id in range(len(CLASS_NAMES)):
        pool = [
            int(index) for index in np.flatnonzero(targets == class_id)
            if int(index) not in used_training
        ]
        rng.shuffle(pool)
        pools[class_id] = pool

    validation_specs: List[ClientSpec] = []
    for spec in client_specs:
        training_counts = np.asarray(spec.class_counts, dtype=np.float64)
        probabilities = training_counts / max(float(training_counts.sum()), 1.0)
        desired = probabilities * samples_per_client
        selected_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        selected: List[int] = []
        for _ in range(samples_per_client):
            available = [c for c in range(len(CLASS_NAMES)) if pools[c]]
            if not available:
                raise ValueError(
                    f"Not enough unused {DOMAIN_NAMES[spec.domain_id]} images "
                    "for extra validation. Reduce --validation-fraction or "
                    "--samples-per-client."
                )
            jitter = rng.random(len(CLASS_NAMES)) * 1e-6
            class_id = max(
                available,
                key=lambda c: float(desired[c] - selected_counts[c] + jitter[c]),
            )
            selected.append(pools[class_id].pop())
            selected_counts[class_id] += 1
        rng.shuffle(selected)
        validation_specs.append(ClientSpec(
            spec.client_id, spec.domain_id, spec.profile_id,
            selected, selected_counts.tolist(),
        ))

    validation_indices = [i for spec in validation_specs for i in spec.indices]
    if len(validation_indices) != len(set(validation_indices)):
        raise RuntimeError("Validation leakage: an image was assigned twice.")
    if used_training.intersection(validation_indices):
        raise RuntimeError("Train/validation leakage detected.")
    return validation_specs


class DomainTestDataset(Dataset):
    def __init__(self, base: Dataset, domain_id: int, image_size: int):
        self.base = base
        self.domain_id = domain_id
        self.transform = domain_transform(domain_id, image_size, train=False)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item: int):
        image, label = self.base[item]
        return self.transform(image), int(label), self.domain_id


def distribute(total: int, classes: Sequence[int]) -> Dict[int, int]:
    result = {c: total // len(classes) for c in classes}
    for c in classes[: total % len(classes)]:
        result[c] += 1
    return result


def requested_counts(samples: int, profile_id: int, majority_fraction: float) -> List[int]:
    low_n = round(samples * (majority_fraction if profile_id == 0 else 1 - majority_fraction))
    high_n = samples - low_n
    counts = [0] * len(CLASS_NAMES)
    for c, n in distribute(low_n, LOW_DIGITS).items():
        counts[c] = n
    for c, n in distribute(high_n, HIGH_DIGITS).items():
        counts[c] = n
    return counts


def build_client_specs(
    targets: Sequence[int], clients_per_domain: int, samples_per_client: int,
    majority_fraction: float, seed: int, domain_id: int = 0,
) -> List[ClientSpec]:
    if clients_per_domain % 2:
        raise ValueError("--clients-per-domain must be even.")
    rng = np.random.default_rng(seed)
    pools: Dict[int, List[int]] = {}
    for c in range(len(CLASS_NAMES)):
        values = np.flatnonzero(np.asarray(targets) == c).tolist()
        rng.shuffle(values)
        pools[c] = values

    offsets = defaultdict(int)
    specs: List[ClientSpec] = []
    for local_id in range(clients_per_domain):
        profile_id = 0 if local_id < clients_per_domain // 2 else 1
        counts = requested_counts(samples_per_client, profile_id, majority_fraction)
        selected: List[int] = []
        for c, count in enumerate(counts):
            start, end = offsets[c], offsets[c] + count
            if end > len(pools[c]):
                raise ValueError(
                    f"Not enough {DOMAIN_NAMES[domain_id]} class-{c} examples. "
                    "Reduce --samples-per-client or --clients-per-domain."
                )
            selected.extend(pools[c][start:end])
            offsets[c] = end
        rng.shuffle(selected)
        specs.append(ClientSpec(local_id, domain_id, profile_id, selected, counts))
    flat = [idx for spec in specs for idx in spec.indices]
    if len(flat) != len(set(flat)):
        raise RuntimeError("Data leakage: an original image was assigned to multiple clients.")
    return specs


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.strength * grad_output, None


class ResidualDomainAdapter(nn.Module):
    """Small near-identity domain branch operating after the shared encoder."""
    def __init__(self, dimension: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.up(F.relu(self.down(z)))


class FederatedNet(nn.Module):
    def __init__(self, pretrained: bool, num_domains: Optional[int] = None,
                 adapter_bottleneck: int = 64, use_domain_adapters: bool = True):
        super().__init__()
        if num_domains is None:
            num_domains = len(DOMAIN_NAMES)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        dimension = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.use_domain_adapters = use_domain_adapters
        self.domain_adapters = nn.ModuleList([
            ResidualDomainAdapter(dimension, adapter_bottleneck)
            for _ in range(num_domains)
        ])
        self.classifier = nn.Linear(dimension, len(CLASS_NAMES))
        self.domain_classifier = nn.Sequential(
            nn.Linear(dimension, 128), nn.ReLU(), nn.Linear(128, num_domains)
        )
        self.feature_dim = dimension

    def apply_domain_adapters(self, features: torch.Tensor,
                              domains: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_domain_adapters:
            return features
        if domains is None:
            raise ValueError("Domain IDs are required when domain adapters are enabled.")
        adapted = torch.empty_like(features)
        for d, adapter in enumerate(self.domain_adapters):
            selected = domains == d
            if selected.any():
                adapted[selected] = adapter(features[selected])
        return adapted

    def forward(self, x: torch.Tensor, grl_strength: float = 0.0,
                domains: Optional[torch.Tensor] = None):
        features = self.encoder(x)
        adapted = self.apply_domain_adapters(features, domains)
        logits = self.classifier(adapted)
        reversed_features = GradientReversal.apply(features, grl_strength)
        domain_logits = self.domain_classifier(reversed_features)
        return features, logits, domain_logits, adapted


def state_average(
    states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]
) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    output: Dict[str, torch.Tensor] = {}
    for key in states[0]:
        value = states[0][key]
        if not torch.is_floating_point(value):
            output[key] = value.clone()
            continue
        acc = torch.zeros_like(value, dtype=torch.float32)
        for state, weight in zip(states, weights):
            acc += state[key].float() * (weight / total)
        output[key] = acc.to(value.dtype)
    return output


def domain_aware_state_average(
    states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float],
    client_domains: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """Aggregate shared parameters globally and adapter d within domain d."""
    output = state_average(states, weights)
    for d in range(len(DOMAIN_NAMES)):
        selected = [i for i, domain in enumerate(client_domains) if domain == d]
        if not selected:
            continue
        local = state_average([states[i] for i in selected], [weights[i] for i in selected])
        prefix = f"domain_adapters.{d}."
        for key, value in local.items():
            if key.startswith(prefix):
                output[key] = value
    return output


def graph_prototype_nce_loss(
    features: torch.Tensor, labels: torch.Tensor,
    bank: Optional[GraphPrototypeBank], temperature: float,
) -> torch.Tensor:
    """Same-class graph nodes are positives; every other class is negative."""
    if bank is None:
        return features.new_zeros(())
    features = F.normalize(features, dim=1)
    nodes = F.normalize(bank.smooth_nodes.to(features.device), dim=-1)
    mask = bank.mask.to(features.device)
    similarities = torch.einsum("bf,cdf->bcd", features, nodes) / temperature
    # Average every observed domain node. Unlike logsumexp, this cannot obtain a
    # low loss by matching only the easiest same-class domain.
    similarities = similarities.masked_fill(~mask.unsqueeze(0), 0.0)
    valid_count = mask.sum(dim=1).clamp_min(1).to(features.dtype)
    class_logits = similarities.sum(dim=2) / valid_count[None, :]
    return F.cross_entropy(class_logits, labels)


def cross_domain_alignment_loss(
    features: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor,
    bank: Optional[GraphPrototypeBank],
) -> torch.Tensor:
    """Pull each feature toward reliable same-class prototypes in other domains."""
    if bank is None:
        return features.new_zeros(())
    z = F.normalize(features, dim=1)
    nodes = F.normalize(bank.smooth_nodes.to(z.device), dim=-1)
    available = bank.mask.to(z.device)[labels]
    reliability = bank.reliability.to(z.device)[labels]
    domain_ids = torch.arange(nodes.size(1), device=z.device)[None, :]
    positive = available & domain_ids.ne(domains[:, None])
    weights = reliability * positive.to(reliability.dtype)
    selected = nodes[labels]
    distances = 1.0 - torch.einsum("bf,bdf->bd", z, selected)
    per_sample = (distances * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-12)
    valid = positive.any(dim=1)
    return per_sample[valid].mean() if valid.any() else features.new_zeros(())


def class_separation_loss(
    features: torch.Tensor, labels: torch.Tensor, margin: float,
) -> torch.Tensor:
    """Prevent domain alignment from collapsing different semantic classes."""
    z = F.normalize(features, dim=1)
    centres = []
    for c in labels.unique():
        centres.append(F.normalize(z[labels == c].mean(dim=0), dim=0))
    if len(centres) < 2:
        return features.new_zeros(())
    centres = torch.stack(centres)
    distances = 1.0 - centres @ centres.T
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    return F.relu(margin - distances[upper]).pow(2).mean()


def global_centroid_losses(
    features: torch.Tensor, labels: torch.Tensor,
    bank: Optional[GraphPrototypeBank], margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align to the correct global centroid and outrank the nearest wrong one."""
    if bank is None:
        zero = features.new_zeros(())
        return zero, zero
    z = F.normalize(features, dim=1)
    centroids = F.normalize(bank.centroids.to(z.device), dim=1)
    valid_classes = bank.mask.to(z.device).any(dim=1)
    distances = 1.0 - z @ centroids.T
    positive_distance = distances.gather(1, labels[:, None]).squeeze(1)
    valid_positive = valid_classes[labels]
    centre_loss = (positive_distance[valid_positive].mean()
                   if valid_positive.any() else features.new_zeros(()))
    negative_distances = distances.masked_fill(~valid_classes[None, :], float("inf"))
    negative_distances = negative_distances.scatter(1, labels[:, None], float("inf"))
    hard_negative = negative_distances.min(dim=1).values
    valid_triplet = valid_positive & torch.isfinite(hard_negative)
    margin_loss = (F.relu(margin + positive_distance[valid_triplet]
                          - hard_negative[valid_triplet]).pow(2).mean()
                   if valid_triplet.any() else features.new_zeros(()))
    return centre_loss, margin_loss


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor,
                                temperature: float) -> torch.Tensor:
    """Supervised contrastive loss; two views guarantee a positive per anchor."""
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positives_per_anchor = positive.sum(dim=1).clamp_min(1)
    return -((log_prob * positive).sum(dim=1) / positives_per_anchor).mean()


def local_class_weights(loader: DataLoader, gamma: float, device: torch.device) -> torch.Tensor:
    counts = torch.tensor(loader.dataset.spec.class_counts, dtype=torch.float32, device=device)
    weights = (counts + 1e-8).pow(-gamma)
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    return weights / weights[weights > 0].mean().clamp_min(1e-12)


def local_train(
    global_model: FederatedNet, loader: DataLoader, device: torch.device,
    epochs: int, encoder_lr: float, head_lr: float, class_gamma: float,
    domain_lambda: float, prototype_lambda: float, contrastive_lambda: float,
    consistency_lambda: float, alignment_lambda: float, separation_lambda: float,
    separation_margin: float, global_centre_lambda: float,
    global_margin_lambda: float, global_margin: float,
    adapter_lambda: float, contrastive_temperature: float,
    prototype_bank: Optional[GraphPrototypeBank], head_only: bool = False,
    grl_strength: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    model = copy.deepcopy(global_model).to(device)
    model.train()
    if head_only:
        model.encoder.requires_grad_(False)
        model.encoder.eval()
    groups = [
        {"params": model.classifier.parameters(), "lr": head_lr},
        {"params": model.domain_classifier.parameters(), "lr": head_lr},
        {"params": model.domain_adapters.parameters(), "lr": head_lr},
    ]
    if not head_only:
        groups.append({"params": model.encoder.parameters(), "lr": encoder_lr})
    optimizer = torch.optim.SGD(groups, momentum=0.9, weight_decay=1e-4)
    class_weights = local_class_weights(loader, class_gamma, device)
    totals = defaultdict(float)
    seen = 0
    for _ in range(epochs):
        for view1, view2, labels, domains in loader:
            view1, view2 = view1.to(device), view2.to(device)
            labels, domains = labels.to(device), domains.to(device)
            optimizer.zero_grad(set_to_none=True)
            f1, y1, d1, a1 = model(view1, grl_strength, domains)
            f2, y2, d2, a2 = model(view2, grl_strength, domains)
            both_features = torch.cat([f1, f2])
            both_labels = torch.cat([labels, labels])
            cls = 0.5 * (F.cross_entropy(y1, labels, weight=class_weights)
                         + F.cross_entropy(y2, labels, weight=class_weights))
            dom = 0.5 * (F.cross_entropy(d1, domains) + F.cross_entropy(d2, domains))
            align = graph_prototype_nce_loss(
                both_features, both_labels, prototype_bank, contrastive_temperature
            )
            contrast = supervised_contrastive_loss(
                both_features, both_labels, contrastive_temperature
            )
            consistency = (1 - F.cosine_similarity(F.normalize(f1, dim=1),
                                                    F.normalize(f2, dim=1))).mean()
            cross_domain = cross_domain_alignment_loss(
                both_features, both_labels, torch.cat([domains, domains]), prototype_bank
            )
            separation = class_separation_loss(both_features, both_labels, separation_margin)
            global_centre, global_margin_loss = global_centroid_losses(
                both_features, both_labels, prototype_bank, global_margin
            )
            adapter_reg = 0.5 * ((a1 - f1).pow(2).mean() + (a2 - f2).pow(2).mean())
            # During head-only stabilization the frozen encoder cannot respond
            # to representation losses, so only CE/domain-head training is used.
            if head_only:
                loss = cls + domain_lambda * dom
            else:
                loss = (cls + domain_lambda * dom + prototype_lambda * align
                        + contrastive_lambda * contrast
                        + consistency_lambda * consistency
                        + alignment_lambda * cross_domain
                        + separation_lambda * separation
                        + global_centre_lambda * global_centre
                        + global_margin_lambda * global_margin_loss
                        + adapter_lambda * adapter_reg)
            loss.backward()
            optimizer.step()
            for name, value in (("loss", loss), ("classification", cls), ("domain", dom),
                                ("prototype", align), ("contrastive", contrast),
                                ("consistency", consistency),
                                ("cross_domain_alignment", cross_domain),
                                ("class_separation", separation),
                                ("global_centre_alignment", global_centre),
                                ("global_prototype_margin", global_margin_loss),
                                ("adapter_regularization", adapter_reg)):
                totals[name] += float(value.detach()) * labels.size(0)
            seen += labels.size(0)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, {name: value / max(seen, 1) for name, value in totals.items()}


@torch.no_grad()
def extract_signatures(
    model: FederatedNet, loaders: Sequence[DataLoader], device: torch.device,
    tau_count: float, tau_variance: float, reliability_floor: float,
    tempering_beta: float,
) -> List[Dict[int, ClassStat]]:
    model.eval().to(device)
    signatures: List[Dict[int, ClassStat]] = []
    for loader in tqdm(loaders, desc="Extracting client signatures"):
        by_class: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for images, labels, _ in loader:
            features = F.normalize(model.encoder(images.to(device)), dim=1).cpu()
            for c in labels.unique().tolist():
                by_class[int(c)].append(features[labels == c])
        stats: Dict[int, ClassStat] = {}
        total = sum(x.size(0) for chunks in by_class.values() for x in chunks)
        unnormalized: Dict[int, float] = {}
        for c, chunks in by_class.items():
            z = torch.cat(chunks, dim=0)
            proto = F.normalize(z.mean(dim=0), dim=0)
            dispersion = float((1.0 - z @ proto).mean())
            count = z.size(0)
            estimate = (count / (count + tau_count)) * math.exp(-tau_variance * dispersion)
            reliability = reliability_floor + (1 - reliability_floor) * estimate
            value = reliability * ((count + 1e-8) ** tempering_beta)
            unnormalized[c] = value
            stats[c] = ClassStat(proto.numpy(), count, dispersion,
                                 count / total, reliability, 0.0)
        normalizer = sum(unnormalized.values()) + 1e-12
        for c in stats:
            stats[c].tempered_support = unnormalized[c] / normalizer
        signatures.append(stats)
    return signatures


def build_class_domain_graphs(
    signatures: Sequence[Mapping[int, ClassStat]], specs: Sequence[ClientSpec],
    dim: int, graph_lambda: float, graph_eta: float, graph_temperature: float,
    graph_ridge: float,
) -> GraphPrototypeBank:
    """Build and Laplacian-smooth one domain-prototype graph per class.

    Client prototypes are first aggregated *within* each class-domain pair.
    Sample-rich domains do not receive extra nodes or extra centroid mass.
    Setting graph_lambda=0 keeps the unsmoothed multi-prototype baseline.
    """
    classes, domains = len(CLASS_NAMES), len(DOMAIN_NAMES)
    raw = np.zeros((classes, domains, dim), dtype=np.float32)
    smooth = np.zeros_like(raw)
    mask = np.zeros((classes, domains), dtype=bool)
    reliability = np.zeros((classes, domains), dtype=np.float32)
    edges = np.zeros((classes, domains, domains), dtype=np.float32)
    centroids = np.zeros((classes, dim), dtype=np.float32)

    for c in range(classes):
        for d in range(domains):
            entries = [
                (signature[c].prototype, signature[c].reliability)
                for signature, spec in zip(signatures, specs)
                if spec.domain_id == d and c in signature
            ]
            if not entries:
                continue
            weights = np.asarray([r for _, r in entries], dtype=np.float64)
            node = sum(p * r for p, r in entries) / (weights.sum() + 1e-12)
            node /= np.linalg.norm(node) + 1e-12
            raw[c, d] = node
            reliability[c, d] = float(np.clip(weights.mean(), 1e-3, 1.0))
            mask[c, d] = True

        observed = np.flatnonzero(mask[c])
        if not len(observed):
            continue
        p = raw[c, observed].astype(np.float64)
        r = reliability[c, observed].astype(np.float64)
        if len(observed) == 1:
            q = p.copy()
            w = np.zeros((1, 1), dtype=np.float64)
        else:
            cosine = np.clip(p @ p.T, -1.0, 1.0)
            similarity = np.exp((cosine - 1.0) / graph_temperature)
            w = np.sqrt(r[:, None] * r[None, :]) * (
                graph_eta + (1.0 - graph_eta) * similarity
            )
            np.fill_diagonal(w, 0.0)
            laplacian = np.diag(w.sum(axis=1)) - w
            r_matrix = np.diag(r)
            system = r_matrix + graph_lambda * laplacian + graph_ridge * np.eye(len(observed))
            q = np.linalg.solve(system, r_matrix @ p)
            q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        smooth[c, observed] = q.astype(np.float32)
        edges[c][np.ix_(observed, observed)] = w.astype(np.float32)
        # Equal-domain centroid: every observed domain contributes exactly once.
        centroids[c] = q.mean(axis=0)
        centroids[c] /= np.linalg.norm(centroids[c]) + 1e-12

    return GraphPrototypeBank(
        raw_nodes=torch.from_numpy(raw),
        smooth_nodes=torch.from_numpy(smooth),
        mask=torch.from_numpy(mask),
        reliability=torch.from_numpy(reliability),
        edge_weights=torch.from_numpy(edges),
        centroids=torch.from_numpy(centroids),
    )


def graph_diagnostics(bank: GraphPrototypeBank) -> Dict[str, object]:
    raw = bank.raw_nodes.numpy()
    smooth = bank.smooth_nodes.numpy()
    mask = bank.mask.numpy()
    per_class: Dict[str, Dict[str, float]] = {}
    raw_all, smooth_all, displacement_all = [], [], []
    raw_different, smooth_different = [], []
    for c, name in enumerate(CLASS_NAMES):
        observed = np.flatnonzero(mask[c])
        raw_distances, smooth_distances = [], []
        for a in range(len(observed)):
            displacement_all.append(1 - np.dot(raw[c, observed[a]], smooth[c, observed[a]]))
            for b in range(a + 1, len(observed)):
                raw_distances.append(1 - np.dot(raw[c, observed[a]], raw[c, observed[b]]))
                smooth_distances.append(1 - np.dot(smooth[c, observed[a]], smooth[c, observed[b]]))
        raw_all.extend(raw_distances)
        smooth_all.extend(smooth_distances)
        per_class[name] = {
            "domains": int(len(observed)),
            "raw_cross_domain_distance": float(np.mean(raw_distances)) if raw_distances else 0.0,
            "smoothed_cross_domain_distance": float(np.mean(smooth_distances)) if smooth_distances else 0.0,
        }
    # Different-class distances within one domain reveal whether smoothing
    # preserves semantic separation instead of merely contracting all nodes.
    for d in range(mask.shape[1]):
        observed_classes = np.flatnonzero(mask[:, d])
        for a in range(len(observed_classes)):
            for b in range(a + 1, len(observed_classes)):
                ca, cb = observed_classes[a], observed_classes[b]
                raw_different.append(1 - np.dot(raw[ca, d], raw[cb, d]))
                smooth_different.append(1 - np.dot(smooth[ca, d], smooth[cb, d]))
    raw_same = float(np.mean(raw_all)) if raw_all else 0.0
    smooth_same = float(np.mean(smooth_all)) if smooth_all else 0.0
    raw_diff = float(np.mean(raw_different)) if raw_different else 0.0
    smooth_diff = float(np.mean(smooth_different)) if smooth_different else 0.0
    return {
        "mean_raw_cross_domain_distance": raw_same,
        "mean_smoothed_cross_domain_distance": smooth_same,
        "mean_raw_different_class_same_domain_distance": raw_diff,
        "mean_smoothed_different_class_same_domain_distance": smooth_diff,
        "raw_semantic_geometry_gap": raw_diff - raw_same,
        "smoothed_semantic_geometry_gap": smooth_diff - smooth_same,
        "mean_node_displacement": float(np.mean(displacement_all)) if displacement_all else 0.0,
        "per_class": per_class,
    }


def domain_balanced_fedavg_weights(specs: Sequence[ClientSpec]) -> List[float]:
    """Each domain receives equal total FedAvg mass; size weights apply within it."""
    totals = defaultdict(float)
    for spec in specs:
        totals[spec.domain_id] += len(spec.indices)
    return [len(spec.indices) / totals[spec.domain_id] for spec in specs]


def fmcl_distance(signatures: Sequence[Mapping[int, ClassStat]], alpha: float = 1.0,
                  cap: float = 100.0, epsilon: float = 1e-3) -> np.ndarray:
    n = len(signatures)
    matrix = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            overlap = sorted(set(signatures[i]) & set(signatures[j]))
            if not overlap:
                continue
            weights, distances = [], []
            for c in overlap:
                weight = min(signatures[i][c].raw_support, signatures[j][c].raw_support)
                cosine = float(np.dot(signatures[i][c].prototype, signatures[j][c].prototype))
                weights.append(weight)
                distances.append(1.0 - np.clip(cosine, -1, 1))
            omega = sum(weights)
            d_cap = float(np.dot(weights, distances) / (omega + epsilon))
            multiplier = min(max(omega, epsilon) ** (-alpha), cap)
            matrix[i, j] = matrix[j, i] = d_cap * multiplier
    finite = matrix[np.isfinite(matrix) & ~np.eye(n, dtype=bool)]
    if finite.size:
        d_big = min(2 * np.percentile(finite, 95), np.percentile(finite, 99))
    else:
        d_big = 1.0
    matrix[~np.isfinite(matrix)] = d_big
    return matrix


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p + 1e-12, q + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    value = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(value / math.log(2))


def proposed_distance(signatures: Sequence[Mapping[int, ClassStat]],
                      prototype_mix: float, overlap_penalty: float) -> np.ndarray:
    n, classes = len(signatures), len(CLASS_NAMES)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = signatures[i], signatures[j]
            overlap = sorted(set(si) & set(sj))
            pi = np.array([si[c].tempered_support if c in si else 0 for c in range(classes)])
            pj = np.array([sj[c].tempered_support if c in sj else 0 for c in range(classes)])
            support_d = js_divergence(pi, pj)
            if overlap:
                weights = np.array([
                    math.sqrt(si[c].tempered_support * sj[c].tempered_support) for c in overlap
                ])
                cosine_d = np.array([
                    1 - np.clip(np.dot(si[c].prototype, sj[c].prototype), -1, 1)
                    for c in overlap
                ])
                proto_d = float(np.dot(weights, cosine_d) / (weights.sum() + 1e-12))
                coverage = float(np.minimum(pi, pj).sum())
            else:
                proto_d, coverage = 1.0, 0.0
            value = (prototype_mix * proto_d + (1 - prototype_mix) * support_d
                     + overlap_penalty * (1 - coverage))
            matrix[i, j] = matrix[j, i] = value
    return matrix


def hard_cluster(distance: np.ndarray, clusters: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=clusters, metric="precomputed", linkage="average")
    except TypeError:  # scikit-learn < 1.2
        model = AgglomerativeClustering(n_clusters=clusters, affinity="precomputed", linkage="average")
    return model.fit_predict(distance)


def fuzzy_medoids(distance: np.ndarray, clusters: int, fuzziness: float,
                  seed: int, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    hard = hard_cluster(distance, clusters)
    medoids = []
    for k in range(clusters):
        members = np.flatnonzero(hard == k)
        medoids.append(int(members[np.argmin(distance[np.ix_(members, members)].sum(axis=1))]))
    medoids = np.asarray(medoids)
    membership = np.zeros((len(distance), clusters))
    for _ in range(max_iter):
        d = np.maximum(distance[:, medoids], 1e-10)
        zero_rows = np.any(distance[:, medoids] <= 1e-10, axis=1)
        power = 2.0 / (fuzziness - 1.0)
        membership = 1.0 / np.sum((d[:, :, None] / d[:, None, :]) ** power, axis=2)
        for i in np.flatnonzero(zero_rows):
            membership[i] = 0
            membership[i, np.argmin(distance[i, medoids])] = 1
        new_medoids = []
        for k in range(clusters):
            costs = (membership[:, k] ** fuzziness) @ distance
            new_medoids.append(int(np.argmin(costs)))
        new_medoids = np.asarray(new_medoids)
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    return membership, medoids


def clustering_metrics(pred: np.ndarray, semantic: np.ndarray, domain: np.ndarray) -> Dict[str, float]:
    return {
        "semantic_ari": float(adjusted_rand_score(semantic, pred)),
        "semantic_nmi": float(normalized_mutual_info_score(semantic, pred)),
        "domain_ari": float(adjusted_rand_score(domain, pred)),
        "domain_nmi": float(normalized_mutual_info_score(domain, pred)),
    }


def prototype_geometry(signatures: Sequence[Mapping[int, ClassStat]], specs: Sequence[ClientSpec]) -> Dict[str, float]:
    cross_domain_same, same_domain_different = [], []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            if specs[i].domain_id != specs[j].domain_id:
                for c in set(signatures[i]) & set(signatures[j]):
                    cross_domain_same.append(1 - np.dot(signatures[i][c].prototype,
                                                         signatures[j][c].prototype))
            elif specs[i].domain_id == specs[j].domain_id:
                for c in signatures[i]:
                    for d in signatures[j]:
                        if c != d:
                            same_domain_different.append(1 - np.dot(signatures[i][c].prototype,
                                                                    signatures[j][d].prototype))
    same = float(np.mean(cross_domain_same))
    different = float(np.mean(same_domain_different))
    return {
        "same_class_cross_domain_distance": same,
        "different_class_same_domain_distance": different,
        "semantic_geometry_gap": different - same,
    }


def signature_vectors(signatures: Sequence[Mapping[int, ClassStat]]) -> np.ndarray:
    vectors = []
    dim = next(iter(signatures[0].values())).prototype.size
    for signature in signatures:
        proto = np.zeros((len(CLASS_NAMES), dim), dtype=np.float32)
        support = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        for c, stat in signature.items():
            proto[c] = stat.prototype * math.sqrt(stat.tempered_support)
            support[c] = stat.tempered_support
        vectors.append(np.concatenate([proto.ravel(), support]))
    return np.stack(vectors)


def save_plots(output: Path, fmcl: np.ndarray, proposed: np.ndarray,
               signatures: Sequence[Mapping[int, ClassStat]], specs: Sequence[ClientSpec],
    membership: np.ndarray) -> None:
    semantic = np.array([s.profile_id for s in specs])
    domain = np.array([s.domain_id for s in specs])
    order = np.lexsort((domain, semantic))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(fmcl[np.ix_(order, order)], ax=axes[0], cmap="mako")
    axes[0].set_title("FMCL-style distance")
    sns.heatmap(proposed[np.ix_(order, order)], ax=axes[1], cmap="mako")
    axes[1].set_title("Proposed distance")
    for ax in axes:
        ax.set_xlabel("Client (semantic then domain order)")
        ax.set_ylabel("Client")
    fig.tight_layout()
    fig.savefig(output / "distance_heatmaps.png", dpi=180)
    plt.close(fig)

    points = PCA(n_components=2, random_state=0).fit_transform(signature_vectors(signatures))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    colours = ["#2b8cbe", "#e34a33"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, point in enumerate(points):
        ax.scatter(
            *point,
            marker=markers[int(domain[i]) % len(markers)],
            color=colours[int(semantic[i]) % len(colours)],
            s=80,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.annotate(str(i), point, xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("PCA of reliability-tempered client signatures")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(output / "client_pca.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(membership))
    for k in range(membership.shape[1]):
        ax.bar(np.arange(len(membership)), membership[:, k], bottom=bottom, label=f"Cluster {k}")
        bottom += membership[:, k]
    ax.set_ylim(0, 1)
    ax.set_xlabel("Client")
    ax.set_ylabel("Soft membership")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "soft_memberships.png", dpi=180)
    plt.close(fig)


def save_client_class_distributions(
    output: Path, specs: Sequence[ClientSpec], majority_fraction: float,
) -> None:
    """Save one ten-class bar chart per client in domain-by-client order."""
    by_domain: Dict[int, List[ClientSpec]] = defaultdict(list)
    for spec in specs:
        by_domain[spec.domain_id].append(spec)
    for domain_specs in by_domain.values():
        domain_specs.sort(key=lambda spec: spec.client_id)

    clients_per_domain = max(len(domain_specs) for domain_specs in by_domain.values())
    fig, axes = plt.subplots(
        len(DOMAIN_NAMES), clients_per_domain,
        figsize=(3.15 * clients_per_domain, 2.45 * len(DOMAIN_NAMES)),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for domain_id, domain_name in enumerate(DOMAIN_NAMES):
        domain_specs = by_domain.get(domain_id, [])
        for local_position in range(clients_per_domain):
            ax = axes[domain_id, local_position]
            if local_position >= len(domain_specs):
                ax.axis("off")
                continue
            spec = domain_specs[local_position]
            ax.bar(
                np.arange(len(CLASS_NAMES)), spec.class_counts,
                color=PROFILE_COLOURS[spec.profile_id],
                edgecolor="black", linewidth=0.55,
                hatch=DOMAIN_HATCHES[domain_id],
            )
            ax.set_title(
                f"Client {spec.client_id + 1}\n"
                f"{GROUP_NAMES[spec.profile_id]} heavy",
                fontsize=9,
            )
            ax.set_xticks(np.arange(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            if local_position == 0:
                ax.set_ylabel(f"{domain_name}\nSamples", fontsize=9)
            if domain_id == len(DOMAIN_NAMES) - 1:
                ax.set_xlabel("Digit class")

    legend_items = [
        Patch(facecolor=PROFILE_COLOURS[0], edgecolor="black",
              label=f"{majority_fraction:.0%} from {GROUP_NAMES[0]}"),
        Patch(facecolor=PROFILE_COLOURS[1], edgecolor="black",
              label=f"{majority_fraction:.0%} from {GROUP_NAMES[1]}"),
    ]
    legend_items.extend(
        Patch(facecolor="white", edgecolor="black", hatch=DOMAIN_HATCHES[d],
              label=DOMAIN_NAMES[d])
        for d in range(len(DOMAIN_NAMES))
    )
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 1.025),
               ncol=7, frameon=False, fontsize=9)
    fig.suptitle(
        f"Non-IID class distribution across {len(specs)} federated clients",
        fontsize=14, y=1.035,
    )
    fig.savefig(output / "client_class_distributions.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "client_class_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def save_graph_plots(output: Path, bank: GraphPrototypeBank) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    weights = bank.edge_weights.numpy()
    mask = bank.mask.numpy()
    for c, ax in enumerate(axes.flat):
        observed = np.flatnonzero(mask[c])
        if len(observed):
            shown = weights[c][np.ix_(observed, observed)]
            labels = [DOMAIN_NAMES[d] for d in observed]
            sns.heatmap(shown, ax=ax, cmap="viridis", vmin=0, vmax=1,
                        xticklabels=labels, yticklabels=labels, cbar=False,
                        annot=True, fmt=".2f")
        ax.set_title(CLASS_NAMES[c])
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Same-class cross-domain graph edge weights")
    fig.savefig(output / "class_domain_graphs.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def evaluate_model(model: FederatedNet, loaders: Sequence[DataLoader], device: torch.device) -> Dict[str, object]:
    model.eval().to(device)
    per_domain, per_domain_balanced = {}, {}
    confusion = np.zeros((10, 10), dtype=np.int64)
    for domain_id, loader in enumerate(loaders):
        correct, total = 0, 0
        domain_confusion = np.zeros((10, 10), dtype=np.int64)
        for images, labels, _ in loader:
            labels = labels.to(device)
            domains = torch.full((labels.size(0),), domain_id,
                                 dtype=torch.long, device=device)
            logits = model(images.to(device), domains=domains)[1]
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum())
            total += labels.numel()
            for y, p in zip(labels.cpu().numpy(), pred.cpu().numpy()):
                confusion[y, p] += 1
                domain_confusion[y, p] += 1
        per_domain[DOMAIN_NAMES[domain_id]] = correct / max(total, 1)
        domain_recalls = np.diag(domain_confusion) / np.maximum(domain_confusion.sum(axis=1), 1)
        per_domain_balanced[DOMAIN_NAMES[domain_id]] = float(domain_recalls.mean())
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    domain_values = list(per_domain.values())
    return {
        "domain_accuracy": per_domain,
        "domain_balanced_accuracy": per_domain_balanced,
        "worst_domain_accuracy": float(min(domain_values)),
        "domain_accuracy_gap": float(max(domain_values) - min(domain_values)),
        "balanced_accuracy": float(recalls.mean()),
        "worst_class_accuracy": float(recalls.min()),
        "per_class_recall": {CLASS_NAMES[i]: float(recalls[i]) for i in range(10)},
    }


@torch.no_grad()
def evaluate_cluster_validation(
    model: FederatedNet, validation_loaders: Sequence[DataLoader],
    membership: np.ndarray, cluster_id: int, device: torch.device,
    min_membership: float,
) -> Dict[str, float]:
    """Membership-weighted validation without touching the final test set."""
    model.eval().to(device)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.float64)
    for client_id, loader in enumerate(validation_loaders):
        client_weight = float(membership[client_id, cluster_id])
        if client_weight < min_membership:
            continue
        for images, labels, domains in loader:
            images, labels, domains = images.to(device), labels.to(device), domains.to(device)
            predictions = model(images, domains=domains)[1].argmax(dim=1)
            for truth, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                confusion[truth, prediction] += client_weight
    support = confusion.sum(axis=1)
    observed = support > 0
    recalls = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    recalls[observed] = np.diag(confusion)[observed] / support[observed]
    balanced = float(recalls[observed].mean()) if observed.any() else 0.0
    worst = float(recalls[observed].min()) if observed.any() else 0.0
    return {
        "balanced_accuracy": balanced,
        "worst_class_accuracy": worst,
        "monitor": 0.5 * balanced + 0.5 * worst,
    }


def train_cluster_models(
    initial: FederatedNet, client_loaders: Sequence[DataLoader], membership: np.ndarray,
    validation_loaders: Sequence[DataLoader], test_loaders: Sequence[DataLoader],
    device: torch.device, min_rounds: int, max_rounds: int, patience: int,
    min_delta: float, metric_ema: float, monitor_window: int,
    local_epochs: int, encoder_lr: float, head_lr: float, class_gamma: float,
    min_membership: float,
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for k in range(membership.shape[1]):
        model = copy.deepcopy(initial).cpu()
        best_state = copy.deepcopy(model.state_dict())
        best_metric = -float("inf")
        best_round = 0
        bad_rounds = 0
        smoothed_metric: Optional[float] = None
        raw_metrics: List[float] = []
        validation_history: List[Dict[str, object]] = []
        active = [i for i in range(len(client_loaders)) if membership[i, k] >= min_membership]
        for round_id in range(max_rounds):
            states, weights = [], []
            for i in tqdm(active, desc=f"Cluster {k}, round {round_id + 1}", leave=False):
                state, _ = local_train(
                    model, client_loaders[i], device, local_epochs,
                    encoder_lr, head_lr, class_gamma,
                    domain_lambda=0.0, prototype_lambda=0.0,
                    contrastive_lambda=0.0, consistency_lambda=0.0,
                    alignment_lambda=0.0, separation_lambda=0.0,
                    separation_margin=0.5, global_centre_lambda=0.0,
                    global_margin_lambda=0.0, global_margin=0.2,
                    adapter_lambda=0.001,
                    contrastive_temperature=0.07,
                    prototype_bank=None,
                    head_only=False, grl_strength=0.0,
                )
                states.append(state)
                weights.append(len(client_loaders[i].dataset) * float(membership[i, k]))
            if states:
                active_domains = [client_loaders[i].dataset.spec.domain_id for i in active]
                model.load_state_dict(domain_aware_state_average(
                    states, weights, active_domains
                ))
            validation = evaluate_cluster_validation(
                model, validation_loaders, membership, k, device, min_membership
            )
            raw_metric = validation["monitor"]
            raw_metrics.append(raw_metric)
            selection_metric = float(np.mean(raw_metrics[-monitor_window:]))
            smoothed_metric = (
                raw_metric if smoothed_metric is None else
                metric_ema * smoothed_metric + (1.0 - metric_ema) * raw_metric
            )
            validation_history.append({
                "round": round_id + 1,
                **validation,
                "smoothed_monitor": float(smoothed_metric),
                "selection_monitor": selection_metric,
            })
            if selection_metric > best_metric + min_delta:
                best_metric = selection_metric
                best_round = round_id + 1
                best_state = copy.deepcopy(model.state_dict())
                bad_rounds = 0
            else:
                bad_rounds += 1
            print(
                f"Cluster {k}, round {round_id + 1}: "
                f"validation={raw_metric:.4f}, selection={selection_metric:.4f}, "
                f"ema={smoothed_metric:.4f}, "
                f"bad_rounds={bad_rounds}/{patience}"
            )
            if round_id + 1 >= min_rounds and bad_rounds >= patience:
                break
        model.load_state_dict(best_state)
        final_test = evaluate_model(model, test_loaders, device)
        results[f"cluster_{k}"] = {
            **final_test,
            "best_round": best_round,
            "stopped_round": len(validation_history),
            "best_validation_metric": best_metric,
            "active_client_ids": active,
            "validation_history": validation_history,
        }
    return results


def write_manifest(path: Path, specs: Sequence[ClientSpec]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "domain", "profile", "total", *CLASS_NAMES])
        for s in specs:
            writer.writerow([s.client_id, DOMAIN_NAMES[s.domain_id], PROFILE_NAMES[s.profile_id],
                             len(s.indices), *s.class_counts])


def write_membership(path: Path, membership: np.ndarray, specs: Sequence[ClientSpec]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "domain", "profile", *[f"cluster_{k}" for k in range(membership.shape[1])]])
        for row, spec in zip(membership, specs):
            writer.writerow([spec.client_id, DOMAIN_NAMES[spec.domain_id], PROFILE_NAMES[spec.profile_id], *row])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/digits5"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/digits5"))
    p.add_argument("--seed", type=int, default=7)
    # CPU is the safe default because merely probing CUDA can fail when a
    # notebook has an incompatible NVIDIA driver/PyTorch CUDA combination.
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    p.add_argument("--clients-per-domain", type=int, default=6)
    p.add_argument("--samples-per-client", type=int, default=1000)
    p.add_argument("--majority-fraction", type=float, default=0.7,
                   help="fraction assigned to a client's majority digit group")
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--warmup-rounds", type=int, default=12)
    p.add_argument("--head-only-rounds", type=int, default=2)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--encoder-lr", type=float, default=0.001)
    p.add_argument("--head-lr", type=float, default=0.01)
    p.add_argument("--class-gamma", type=float, default=0.5)
    p.add_argument("--domain-lambda", type=float, default=0.0)
    p.add_argument("--prototype-lambda", type=float, default=0.1)
    p.add_argument("--contrastive-lambda", type=float, default=0.1)
    p.add_argument("--consistency-lambda", type=float, default=0.05)
    p.add_argument("--alignment-lambda", type=float, default=0.2,
                   help="same-class cross-domain feature/prototype alignment weight")
    p.add_argument("--separation-lambda", type=float, default=0.05,
                   help="different-class centre margin-loss weight")
    p.add_argument("--separation-margin", type=float, default=0.5,
                   help="minimum cosine distance between batch class centres")
    p.add_argument("--global-centre-lambda", type=float, default=0.20,
                   help="alignment weight to the server global class centroid")
    p.add_argument("--global-margin-lambda", type=float, default=0.10,
                   help="hard-negative global prototype margin-loss weight")
    p.add_argument("--global-margin", type=float, default=0.20,
                   help="required distance advantage over the nearest wrong centroid")
    p.add_argument("--loss-ramp-rounds", type=int, default=5,
                   help="rounds used to ramp representation losses after head-only warm-up")
    p.add_argument("--adapter-bottleneck", type=int, default=64)
    p.add_argument("--adapter-lambda", type=float, default=0.001,
                   help="keeps each residual domain adapter close to identity")
    p.add_argument("--no-domain-adapters", action="store_true",
                   help="ablation: classify directly from the shared representation")
    p.add_argument("--contrastive-temperature", type=float, default=0.07)
    p.add_argument("--tau-count", type=float, default=20.0)
    p.add_argument("--tau-variance", type=float, default=0.0,
                   help="0 avoids automatically suppressing hard/high-dispersion domains")
    p.add_argument("--reliability-floor", type=float, default=0.3)
    p.add_argument("--graph-lambda", type=float, default=0.1,
                   help="Laplacian smoothing strength; 0 is the raw multi-prototype ablation")
    p.add_argument("--graph-eta", type=float, default=0.2,
                   help="Minimum semantic edge strength between same-class domains")
    p.add_argument("--graph-temperature", type=float, default=0.2)
    p.add_argument("--graph-ridge", type=float, default=1e-4)
    p.add_argument("--tempering-beta", type=float, default=0.5)
    p.add_argument("--prototype-mix", type=float, default=0.6)
    p.add_argument("--overlap-penalty", type=float, default=0.1)
    p.add_argument("--clusters", type=int, default=2)
    p.add_argument("--fuzziness", type=float, default=2.0)
    p.add_argument("--cluster-min-rounds", type=int, default=5)
    p.add_argument("--cluster-max-rounds", type=int, default=30)
    p.add_argument("--cluster-patience", type=int, default=5)
    p.add_argument("--cluster-min-delta", type=float, default=0.001)
    p.add_argument("--cluster-metric-ema", type=float, default=0.6)
    p.add_argument("--cluster-monitor-window", type=int, default=3)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--min-membership", type=float, default=0.1)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--skip-cluster-training", action="store_true")
    if argv is not None:
        return p.parse_args(list(argv))
    if "ipykernel" in sys.modules:
        # Jupyter injects `-f <kernel.json>`. Keep valid user arguments while
        # ignoring only notebook-owned extras.
        args, _unknown = p.parse_known_args()
        return args
    return p.parse_args()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not 0.5 < args.majority_fraction < 1:
        raise ValueError("--majority-fraction must lie between 0.5 and 1.")
    if not 0 < args.tempering_beta <= 1:
        raise ValueError("--tempering-beta must lie in (0, 1].")
    if args.fuzziness <= 1:
        raise ValueError("--fuzziness must be greater than 1.")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must lie in (0, 0.5).")
    if not 1 <= args.cluster_min_rounds <= args.cluster_max_rounds:
        raise ValueError("Require 1 <= cluster-min-rounds <= cluster-max-rounds.")
    if args.cluster_patience < 1 or args.cluster_min_delta < 0:
        raise ValueError("Cluster patience must be positive and min-delta non-negative.")
    if not 0 <= args.cluster_metric_ema < 1:
        raise ValueError("--cluster-metric-ema must lie in [0, 1).")
    if args.cluster_monitor_window < 1:
        raise ValueError("--cluster-monitor-window must be positive.")
    if not 0 <= args.head_only_rounds <= args.warmup_rounds:
        raise ValueError("--head-only-rounds must be between 0 and --warmup-rounds.")
    if not 0 <= args.reliability_floor < 1:
        raise ValueError("--reliability-floor must lie in [0, 1).")
    if args.graph_lambda < 0 or not 0 <= args.graph_eta <= 1:
        raise ValueError("Require --graph-lambda >= 0 and --graph-eta in [0, 1].")
    if args.graph_temperature <= 0 or args.graph_ridge <= 0:
        raise ValueError("Graph temperature and ridge must be positive.")
    if args.loss_ramp_rounds < 1 or args.separation_margin <= 0:
        raise ValueError("Loss ramp rounds and separation margin must be positive.")
    if args.global_centre_lambda < 0 or args.global_margin_lambda < 0 or args.global_margin <= 0:
        raise ValueError("Global loss weights must be non-negative and margin positive.")
    if args.adapter_bottleneck < 1 or args.adapter_lambda < 0:
        raise ValueError("Adapter bottleneck must be positive and adapter lambda non-negative.")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is unavailable. Use --device cpu, "
            "or install a PyTorch CUDA build compatible with the NVIDIA driver."
        )

    base_trains = [load_domain_split(args.data_dir, name, "train") for name in DOMAIN_NAMES]
    base_tests = [load_domain_split(args.data_dir, name, "test") for name in DOMAIN_NAMES]
    specs, client_bases = [], []
    for domain_id, base in enumerate(base_trains):
        targets = getattr(base, "targets", getattr(base, "labels", None))
        if targets is None:
            raise AttributeError(f"Dataset {DOMAIN_NAMES[domain_id]} has no targets/labels")
        domain_specs = build_client_specs(
            targets, args.clients_per_domain, args.samples_per_client,
            args.majority_fraction, args.seed + domain_id, domain_id,
        )
        for spec in domain_specs:
            spec.client_id = len(specs)
            specs.append(spec)
            client_bases.append(base)
    write_manifest(args.output_dir / "client_manifest.csv", specs)
    save_client_class_distributions(args.output_dir, specs, args.majority_fraction)

    # Keep all assigned client examples for training. Allocate extra local
    # validation examples from the unused part of each domain's train split.
    train_specs = specs
    validation_per_client = max(1, round(args.samples_per_client * args.validation_fraction))
    validation_specs: List[ClientSpec] = []
    for domain_id, base in enumerate(base_trains):
        domain_specs = [spec for spec in specs if spec.domain_id == domain_id]
        validation_specs.extend(build_extra_validation_specs(
            base, domain_specs, validation_per_client,
            args.seed + 10_000 + domain_id,
        ))
    validation_specs.sort(key=lambda spec: spec.client_id)
    write_manifest(args.output_dir / "client_train_manifest.csv", train_specs)
    write_manifest(args.output_dir / "client_validation_manifest.csv", validation_specs)

    train_sets = [ClientDataset(base, s, args.image_size, train=True, two_views=True)
                  for base, s in zip(client_bases, train_specs)]
    prototype_sets = [ClientDataset(base, s, args.image_size, train=False)
                      for base, s in zip(client_bases, train_specs)]
    validation_sets = [ClientDataset(base, s, args.image_size, train=False)
                       for base, s in zip(client_bases, validation_specs)]
    train_loaders = [DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.workers,
                                pin_memory=device.type == "cuda") for ds in train_sets]
    prototype_loaders = [DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers,
                                    pin_memory=device.type == "cuda") for ds in prototype_sets]
    validation_loaders = [DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers,
                                     pin_memory=device.type == "cuda") for ds in validation_sets]
    test_loaders = [DataLoader(DomainTestDataset(base, d, args.image_size), args.batch_size,
                               shuffle=False, num_workers=args.workers)
                    for d, base in enumerate(base_tests)]

    print("Extracting frozen FMCL-style signatures...")
    frozen = FederatedNet(pretrained=not args.no_pretrained,
                          adapter_bottleneck=args.adapter_bottleneck,
                          use_domain_adapters=not args.no_domain_adapters)
    frozen_signatures = extract_signatures(frozen, prototype_loaders, device,
                                            args.tau_count, args.tau_variance,
                                            args.reliability_floor,
                                            args.tempering_beta)
    frozen_fmcl = fmcl_distance(frozen_signatures)
    frozen.cpu()
    del frozen
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Federated domain-robust warm-up...")
    global_model = FederatedNet(pretrained=not args.no_pretrained,
                                adapter_bottleneck=args.adapter_bottleneck,
                                use_domain_adapters=not args.no_domain_adapters)
    prototype_bank: Optional[GraphPrototypeBank] = None
    history = []
    aggregation_weights = domain_balanced_fedavg_weights(specs)
    for round_id in range(args.warmup_rounds):
        states, losses = [], []
        head_only = round_id < args.head_only_rounds
        active_round = max(0, round_id - args.head_only_rounds + 1)
        loss_ramp = min(1.0, active_round / args.loss_ramp_rounds)
        for client_id, loader in enumerate(tqdm(train_loaders, desc=f"Warm-up round {round_id + 1}")):
            state, loss = local_train(
                global_model, loader, device, args.local_epochs,
                args.encoder_lr, args.head_lr, args.class_gamma,
                args.domain_lambda, loss_ramp * args.prototype_lambda,
                loss_ramp * args.contrastive_lambda,
                loss_ramp * args.consistency_lambda,
                loss_ramp * args.alignment_lambda,
                loss_ramp * args.separation_lambda,
                args.separation_margin,
                loss_ramp * args.global_centre_lambda,
                loss_ramp * args.global_margin_lambda,
                args.global_margin,
                args.adapter_lambda,
                args.contrastive_temperature, prototype_bank,
                head_only=head_only,
                grl_strength=min(1.0, (round_id + 1) / max(args.warmup_rounds, 1)),
            )
            states.append(state)
            losses.append(loss)
        global_model.load_state_dict(domain_aware_state_average(
            states, aggregation_weights, [s.domain_id for s in specs]
        ))
        round_signatures = extract_signatures(global_model, prototype_loaders, device,
                                               args.tau_count, args.tau_variance,
                                               args.reliability_floor,
                                               args.tempering_beta)
        prototype_bank = build_class_domain_graphs(
            round_signatures, specs, global_model.feature_dim,
            args.graph_lambda, args.graph_eta, args.graph_temperature, args.graph_ridge,
        )
        mean_losses = {
            key: float(np.mean([client[key] for client in losses])) for key in losses[0]
        }
        history.append({"round": round_id + 1, "head_only": head_only,
                        "representation_loss_ramp": loss_ramp, **mean_losses})
        print(f"Round {round_id + 1}: loss={mean_losses['loss']:.4f}, head_only={head_only}")

    learned_signatures = extract_signatures(global_model, prototype_loaders, device,
                                             args.tau_count, args.tau_variance,
                                             args.reliability_floor,
                                             args.tempering_beta)
    prototype_bank = build_class_domain_graphs(
        learned_signatures, specs, global_model.feature_dim,
        args.graph_lambda, args.graph_eta, args.graph_temperature, args.graph_ridge,
    )
    learned_fmcl = fmcl_distance(learned_signatures)
    proposed = proposed_distance(learned_signatures, args.prototype_mix, args.overlap_penalty)

    semantic = np.array([s.profile_id for s in specs])
    domain = np.array([s.domain_id for s in specs])
    signature_matrix = signature_vectors(learned_signatures)
    proposed_for_silhouette = np.asarray(proposed, dtype=np.float64).copy()
    proposed_for_silhouette = (
        proposed_for_silhouette + proposed_for_silhouette.T
    ) / 2.0
    proposed_for_silhouette = np.maximum(proposed_for_silhouette, 0.0)
    np.fill_diagonal(proposed_for_silhouette, 0.0)
    signature_silhouette = {
        "semantic": float(silhouette_score(
            signature_matrix, semantic, metric="euclidean"
        )),
        "domain": float(silhouette_score(
            signature_matrix, domain, metric="euclidean"
        )),
    }
    proposed_distance_silhouette = {
        "semantic": float(silhouette_score(
            proposed_for_silhouette, semantic, metric="precomputed"
        )),
        "domain": float(silhouette_score(
            proposed_for_silhouette, domain, metric="precomputed"
        )),
    }
    print("Signature-vector silhouette:", signature_silhouette)
    print("Proposed-distance silhouette:", proposed_distance_silhouette)
    labels_frozen = hard_cluster(frozen_fmcl, args.clusters)
    labels_learned_raw = hard_cluster(learned_fmcl, args.clusters)
    labels_proposed = hard_cluster(proposed, args.clusters)
    membership, medoids = fuzzy_medoids(proposed, args.clusters, args.fuzziness, args.seed)
    labels_soft = membership.argmax(axis=1)

    # Persist everything needed to inspect clustering after main() returns.
    # This is important in notebooks because membership is local to main().
    client_sizes = np.asarray([len(loader.dataset) for loader in train_loaders],
                              dtype=np.int64)
    hard_counts = np.bincount(labels_soft, minlength=args.clusters)
    soft_effective_clients = membership.sum(axis=0)
    effective_samples = membership.T @ client_sizes.astype(np.float64)
    allocation = {
        "hard_client_counts": hard_counts.tolist(),
        "hard_client_percentages": (
            100.0 * hard_counts / max(len(labels_soft), 1)
        ).tolist(),
        "soft_effective_clients": soft_effective_clients.tolist(),
        "soft_effective_client_percentages": (
            100.0 * soft_effective_clients
            / max(float(soft_effective_clients.sum()), 1e-12)
        ).tolist(),
        "effective_samples": effective_samples.tolist(),
        "effective_sample_percentages": (
            100.0 * effective_samples
            / max(float(effective_samples.sum()), 1e-12)
        ).tolist(),
        "hard_client_ids": [
            np.flatnonzero(labels_soft == k).tolist()
            for k in range(args.clusters)
        ],
    }
    print("Cluster allocation:", json.dumps(allocation, indent=2))

    metrics = {
        "fmcl_frozen_hard": clustering_metrics(labels_frozen, semantic, domain),
        "learned_raw_hard": clustering_metrics(labels_learned_raw, semantic, domain),
        "learned_tempered_hard": clustering_metrics(labels_proposed, semantic, domain),
        "learned_tempered_soft": clustering_metrics(labels_soft, semantic, domain),
        "signature_silhouette": signature_silhouette,
        "proposed_distance_silhouette": proposed_distance_silhouette,
        "prototype_geometry": prototype_geometry(learned_signatures, specs),
        "soft_membership_entropy": float(np.mean(-np.sum(membership * np.log(membership + 1e-12), axis=1))),
        "cluster_allocation": allocation,
        "medoid_client_ids": medoids.tolist(),
        "warmup_history": history,
        "warmup_model": evaluate_model(global_model, test_loaders, device),
        "class_domain_graph": graph_diagnostics(prototype_bank),
    }
    np.save(args.output_dir / "fmcl_distance.npy", frozen_fmcl)
    np.save(args.output_dir / "proposed_distance.npy", proposed)
    np.save(args.output_dir / "signature_vectors.npy", signature_matrix)
    np.save(args.output_dir / "soft_membership.npy", membership)
    np.save(args.output_dir / "hard_cluster_labels.npy", labels_soft)
    np.save(args.output_dir / "client_sizes.npy", client_sizes)
    np.save(args.output_dir / "cluster_medoids.npy", medoids)
    with (args.output_dir / "cluster_allocation.json").open("w") as f:
        json.dump(allocation, f, indent=2)
    write_membership(args.output_dir / "soft_memberships.csv", membership, specs)
    save_plots(args.output_dir, frozen_fmcl, proposed, learned_signatures, specs, membership)
    save_graph_plots(args.output_dir, prototype_bank)
    np.savez_compressed(
        args.output_dir / "class_domain_graph_bank.npz",
        raw_nodes=prototype_bank.raw_nodes.numpy(),
        smooth_nodes=prototype_bank.smooth_nodes.numpy(),
        mask=prototype_bank.mask.numpy(),
        reliability=prototype_bank.reliability.numpy(),
        edge_weights=prototype_bank.edge_weights.numpy(),
        centroids=prototype_bank.centroids.numpy(),
    )

    checkpoint = {
        "model": global_model.state_dict(),
        "config": vars(args),
        "class_names": CLASS_NAMES,
        "domain_names": DOMAIN_NAMES,
        "graph_bank": {
            "raw_nodes": prototype_bank.raw_nodes,
            "smooth_nodes": prototype_bank.smooth_nodes,
            "mask": prototype_bank.mask,
            "reliability": prototype_bank.reliability,
            "edge_weights": prototype_bank.edge_weights,
            "centroids": prototype_bank.centroids,
        },
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")

    if not args.skip_cluster_training:
        cluster_results = train_cluster_models(
            global_model, train_loaders, membership, validation_loaders,
            test_loaders, device, args.cluster_min_rounds, args.cluster_max_rounds,
            args.cluster_patience, args.cluster_min_delta, args.cluster_metric_ema,
            args.cluster_monitor_window,
            args.local_epochs, args.encoder_lr, args.head_lr,
            args.class_gamma, args.min_membership,
        )
        with (args.output_dir / "cluster_models.json").open("w") as f:
            json.dump(cluster_results, f, indent=2)
        metrics["cluster_models"] = cluster_results

    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
