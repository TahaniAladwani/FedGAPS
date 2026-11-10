"""Shared, auditable class-wise Dirichlet partitions and dataset adapters."""
import hashlib
import importlib
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from .utils import atomic_json
from .planted import allocate, allocate_finite

REFERENCE = Path(__file__).resolve().parents[1] / 'reference'
sys.path.insert(0, str(REFERENCE))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'icde-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'icde-cache'))


def to_rgb(image):
    return image.convert("RGB")


def reference_core():
    return importlib.import_module('run_digits5')


class Synthetic(Dataset):
    """Deterministic, independent image splits for exercising the real pipeline."""
    def __init__(self, domain, split, classes=4):
        self.targets = np.tile(np.arange(classes), 32 if split == 'train' else 12).tolist()
        self.domain, self.split = domain, split

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        label = self.targets[index]
        rng = np.random.default_rng(700000 + self.domain * 10000 + (self.split == 'test') * 1000 + index)
        image = rng.integers(0, 40, (32, 32, 3), dtype=np.uint8)
        image[label * 7:label * 7 + 5, :, self.domain % 3] = 220
        return Image.fromarray(image), label


class PreparedDigits(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.images = data['images']
            self.targets = data['labels'].astype(np.int64)
        if len(self.images) != len(self.targets) or not len(self.targets):
            raise ValueError(f'Invalid prepared digits file: {path}')
        if self.targets.min() < 0 or self.targets.max() > 9:
            raise ValueError(f'Invalid digit labels: {path}')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.images[index]
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        return Image.fromarray(image), int(self.targets[index])


def load_digits_split(core, root, name, split):
    prepared = root / name / f'{split}.npz'
    if prepared.is_file():
        return PreparedDigits(prepared)
    try:
        return core.load_domain_split(root, name, split)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'{error}\nPrepare the missing domains with: '
            f'python -m icde.prepare_digits5 --root "{root}"'
        ) from error


def load_bases(config):
    core = reference_core()
    # Each run is a separate process; reload prevents cross-dataset globals leaking in tests.
    importlib.reload(core)
    root = Path(config.data_root) / config.dataset
    if config.dataset == 'synthetic':
        core.CLASS_NAMES = ('zero', 'one', 'two', 'three')
        core.DOMAIN_NAMES = ('domain_a', 'domain_b', 'domain_c')
        core.LOW_DIGITS, core.HIGH_DIGITS = (0, 1), (2, 3)
        return core, [Synthetic(d, 'train') for d in range(3)], [Synthetic(d, 'test') for d in range(3)]
    if config.dataset != 'digits5':
        if config.dataset in ('domainnet10', 'domainnet40'):
            importlib.reload(importlib.import_module('run_domainnet10'))
        adapter = importlib.reload(importlib.import_module('run_' + config.dataset))
        if config.dataset == 'domainnet40':
            adapter._configure_core(*adapter.select_domainnet40_classes(root))
        else:
            adapter._configure_core()
    loader = (lambda root, name, split: load_digits_split(core, root, name, split)) if config.dataset == 'digits5' else core.load_domain_split
    train = [loader(root, name, 'train') for name in core.DOMAIN_NAMES]
    test = [loader(root, name, 'test') for name in core.DOMAIN_NAMES]
    for a, b in zip(train, test):
        if hasattr(a, 'samples') and hasattr(b, 'samples'):
            left = [str(Path(p).resolve()) for p, _ in a.samples]
            right = [str(Path(p).resolve()) for p, _ in b.samples]
            if len(set(left)) != len(left) or len(set(right)) != len(right) or set(left) & set(right):
                raise ValueError('Duplicate image paths or train/test overlap in source data')
        if hasattr(a, 'classes') and hasattr(b, 'classes') and a.classes != b.classes:
            raise ValueError('Train/test class mappings differ')
        if isinstance(a, datasets.ImageFolder) and config.dataset in ('digits5', 'office10'):
            expected = [str(i) for i in range(10)] if config.dataset == 'digits5' else list(core.CLASS_NAMES)
            if list(a.classes) != expected:
                raise ValueError(f'Unexpected class folder mapping: {a.classes}; expected {expected}')
    return core, train, test


def labels_of(base):
    values = getattr(base, 'targets', getattr(base, 'labels', None))
    if values is None:
        raise ValueError('Dataset does not expose targets/labels')
    return np.asarray(values, dtype=np.int64)


def dirichlet_indices(labels, clients, alpha, seed, minimum=4, cap=0):
    """Class-wise Dirichlet-multinomial allocation, rejecting tiny clients.

    All selected examples are used exactly once. A cap selects a uniform subset
    before partitioning. No post-hoc rebalancing disguises the requested alpha.
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(len(labels))
    if cap and len(pool) > cap:
        pool = rng.choice(pool, cap, replace=False)
    if len(pool) < clients * minimum:
        raise ValueError('Not enough samples for the requested clients/minimum')
    for attempt in range(2000):
        result = [[] for _ in range(clients)]
        for label in np.unique(labels):
            indices = pool[labels[pool] == label].copy()
            rng.shuffle(indices)
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
            for client, indices_i in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                result[client].extend(indices_i.tolist())
        if min(map(len, result)) >= minimum:
            for indices in result:
                rng.shuffle(indices)
            return result, attempt
    raise ValueError('Dirichlet rejection exhausted; reduce clients/minimum or increase data')


def split_validation(indices, labels, fraction, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    indices = np.asarray(indices)
    for label in np.unique(labels[indices]):
        local = indices[labels[indices] == label].copy()
        rng.shuffle(local)
        n = min(len(local) - 1, max(1, round(len(local) * fraction))) if len(local) > 1 else 0
        validation.extend(local[:n].tolist())
        train.extend(local[n:].tolist())
    if not validation:
        rng.shuffle(train)
        validation.append(train.pop())
    if not train:
        raise ValueError('Empty training split')
    return train, validation


def fingerprint(bases):
    digest = hashlib.sha256()
    for base in bases:
        digest.update(labels_of(base).tobytes())
        for attribute in ('images', 'data'):
            values = getattr(base, attribute, None)
            if isinstance(values, (np.ndarray, torch.Tensor)):
                digest.update(np.ascontiguousarray(values).tobytes())
                break
        for path, label in getattr(base, 'samples', []):
            path = Path(path)
            stat = path.stat()
            digest.update(f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{label}'.encode())
    return digest.hexdigest()


def make_manifest(config, core, train_bases, test_bases):
    classes = len(core.CLASS_NAMES)
    fingerprint_value = fingerprint(train_bases + test_bases)
    meta = dict(dataset=config.dataset, seed=config.seed, alpha=config.alpha,
                clients_per_domain=config.clients_per_domain, cap=config.max_samples_per_domain,
                minimum=config.min_client_samples, validation_fraction=config.validation_fraction,
                classes=list(core.CLASS_NAMES), domains=list(core.DOMAIN_NAMES),
                data_fingerprint=fingerprint_value, partition_version=2,
                partition='planted_70_30', planted_samples=config.planted_samples)
    if config.partition_mode == 'dirichlet':
        meta.update(partition='dirichlet', partition_version=3)
    elif config.planted_sampling == 'finite':
        meta.update(partition='planted_finite_70_30_capacity_v2', partition_version=4)
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(config.output_root) / 'partitions' / f'{config.dataset}-{key}.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['meta'] != meta:
            raise ValueError('Partition metadata mismatch')
        return manifest, path
    clients, attempts = [], []
    allocation_diagnostics = []
    for d, (base, test_base) in enumerate(zip(train_bases, test_bases)):
        y, yt = labels_of(base), labels_of(test_base)
        if min(y.min(), yt.min()) < 0 or max(y.max(), yt.max()) >= classes:
            raise ValueError('Labels outside the shared class mapping')
        if config.partition_mode == 'dirichlet':
            allocations, tries = dirichlet_indices(y, config.clients_per_domain, config.alpha,
                config.seed + d*1009, config.min_client_samples, config.max_samples_per_domain)
            groups = [-1] * config.clients_per_domain
        else:
            if config.planted_sampling == 'finite':
                allocations, groups, diagnostic = allocate_finite(y, classes, config.clients_per_domain,
                    config.planted_samples, d, len(train_bases), config.seed + d*1009,
                    return_diagnostics=True)
                diagnostic['domain'] = str(core.DOMAIN_NAMES[d])
                allocation_diagnostics.append(diagnostic)
                tries = diagnostic['rejected_draws'] + 1
                if diagnostic['small_domain_fallback']:
                    print(f"Small-domain allocation: {diagnostic['domain']}, "
                          f"{diagnostic['samples_per_client']} samples/client before validation; "
                          f"class capacities={diagnostic['available_class_counts']}; "
                          f"rejected draws={diagnostic['rejected_draws']}", flush=True)
            else:
                allocations, groups = allocate(y, classes, config.clients_per_domain,
                    config.planted_samples, d, len(train_bases), config.seed + d*1009)
                tries = 1
        attempts.append(tries)
        local_clients = []
        for indices, planted_group in zip(allocations, groups):
            cid = len(clients)
            train, val = split_validation(indices, y, config.validation_fraction, config.seed + 50000 + cid)
            counts = np.bincount(y[train], minlength=classes)
            low = int(counts[list(core.LOW_DIGITS)].sum())
            high = int(counts[list(core.HIGH_DIGITS)].sum())
            local_clients.append(dict(id=cid, domain=d, train=train, validation=val, test=[],
                counts=counts.tolist(), planted_group=int(planted_group),
                semantic_profile=int(planted_group) if config.partition_mode == 'planted' else (0 if low > high else 1 if high > low else -1)))
            clients.append(local_clients[-1])
        # Disjoint personalized test shards; allocation uses TRAIN histograms only.
        # Labels construct the benchmark, never choose a model at inference.
        rng = np.random.default_rng(config.seed + 90000 + d)
        for c in range(classes):
            indices = np.flatnonzero(yt == c)
            rng.shuffle(indices)
            weights = np.asarray([x['counts'][c] for x in local_clients], dtype=float)
            if weights.sum() == 0:
                weights[:] = 1
            expected = weights / weights.sum() * len(indices)
            sizes = np.floor(expected).astype(int)
            order = np.argsort(-(expected - sizes), kind='stable')
            sizes[order[:len(indices) - sizes.sum()]] += 1
            for client, chunk in zip(local_clients, np.split(indices, np.cumsum(sizes)[:-1])):
                client['test'].extend(chunk.tolist())
    manifest = dict(meta=meta, clients=clients, rejection_attempts=attempts,
        semantic_label_definition='assigned before sampling: G0 70% first half of classes; G1 70% second half; each domain has equal client counts per group')
    if config.partition_mode == 'dirichlet':
        manifest['semantic_label_definition'] = 'No planted labels; semantic_profile is a training-histogram proxy, not used as ground truth.'
    elif config.planted_sampling == 'finite':
        manifest['semantic_label_definition'] = 'Latent 70/30 class-mixture profiles assigned before multinomial sampling; groups balanced within each domain.'
        manifest['allocation_diagnostics'] = allocation_diagnostics
        manifest['generating_profiles'] = [[(.7 if (c < classes//2) == (g==0) else .3)/(classes//2)
                                           for c in range(classes)] for g in range(2)]
    atomic_json(path, manifest)
    return manifest, path


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class View(Dataset):
    def __init__(self, base, indices, domain, transform, two_views=False):
        self.base, self.indices, self.domain = base, indices, domain
        self.transform, self.two_views = transform, two_views

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.base[self.indices[position]]
        first = self.transform(image)
        second = self.transform(image) if self.two_views else first
        return first, second, int(label), self.domain, position


@dataclass
class Data:
    core: object
    train_bases: list
    test_bases: list
    manifest: dict
    manifest_path: Path
    config: object

    @property
    def clients(self):
        return self.manifest['clients']

    def loader(self, cid, split, round_id=0, two_views=False):
        client = self.clients[cid]
        d = client['domain']
        train = split == 'train'
        source_split = 'train' if split == 'prototype' else split
        base = self.test_bases[d] if split == 'test' else self.train_bases[d]
        if self.config.dataset == 'synthetic':
            transform = transforms.Compose([transforms.Resize((self.config.image_size,) * 2), transforms.ToTensor()])
        elif self.config.dataset == 'digits5':
            ops = [transforms.Resize((self.config.image_size,)*2), transforms.Lambda(to_rgb)]
            if train:
                ops.append(transforms.RandomAffine(degrees=8, translate=(.05,.05), scale=(.95,1.05)))
            ops.extend([transforms.ToTensor(), transforms.Normalize((.5,)*3,(.5,)*3)])
            transform = transforms.Compose(ops)
        else:
            transform = self.core.domain_transform(d, self.config.image_size, train=train)
        ds = View(base, client[source_split], d, transform, two_views)
        generator = torch.Generator().manual_seed(self.config.seed + 100003 * round_id + 97 * cid)
        return DataLoader(ds, batch_size=self.config.batch_size, shuffle=train,
            num_workers=self.config.workers, pin_memory=self.config.device == 'cuda',
            worker_init_fn=seed_worker, generator=generator, persistent_workers=False)


def prepare(config):
    core, train, test = load_bases(config)
    manifest, path = make_manifest(config, core, train, test)
    return Data(core, train, test, manifest, path, config)
