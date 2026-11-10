import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def seed_all(seed, deterministic=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class Average:
    """Streaming weighted state average: memory independent of client count."""
    def __init__(self, exclude=()):
        self.exclude = set(exclude)
        self.values = {}
        self.total = 0.0

    def add(self, state, weight):
        if weight <= 0:
            return
        for key, value in state.items():
            if key in self.exclude:
                continue
            value = value.detach().cpu()
            if key not in self.values:
                self.values[key] = value.float().clone() * weight if value.is_floating_point() else value.clone()
            elif value.is_floating_point():
                self.values[key].add_(value.float(), alpha=float(weight))
            else:
                self.values[key] = torch.maximum(self.values[key], value)
        self.total += weight

    def result(self, template):
        if self.total <= 0:
            return {k: v.clone() for k, v in template.items()}
        return {k: (self.values[k] / self.total).to(v.dtype) if k in self.values and v.is_floating_point()
                else self.values.get(k, v).clone() for k, v in template.items()}


def code_hash():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.glob('*.py')) + sorted((root.parent / 'reference').glob('*.py')) + sorted(root.parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
