from dataclasses import dataclass, asdict
import hashlib
import math
import json

BASELINES = ('fedavg', 'fedbn', 'fedproto', 'fedplvm', 'feddap', 'fedrc', 'fpl', 'fedsoft')
ABLATIONS = ('ours_no_graph', 'ours_no_reliability', 'ours_hard', 'ours_no_global_centre',
             'ours_no_margin', 'ours_fixed_rounds')
METHODS = BASELINES + ('ours',) + ABLATIONS
DATASETS = ('digits5', 'office10', 'domainnet10', 'domainnet40', 'synthetic')

@dataclass
class Config:
    dataset: str = 'digits5'
    data_root: str = 'data'
    output_root: str = 'outputs'
    method: str = 'ours'
    seed: int = 42
    alpha: float = 0.1
    clients_per_domain: int = 4
    max_samples_per_domain: int = 0
    min_client_samples: int = 4
    validation_fraction: float = 0.2
    image_size: int = 96
    batch_size: int = 64
    workers: int = 0
    threads: int = 4
    device: str = 'cpu'
    backbone: str = 'resnet18'
    pretrained: bool = True
    amp: bool = False
    deterministic: bool = True
    rounds: int = 50
    warmup_rounds: int = 20
    head_only_rounds: int = 2
    local_epochs: int = 1
    encoder_lr: float = 0.001
    head_lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    clusters: int = 2
    early_stop_min_rounds: int = 1
    early_stop_patience: int = 5
    monitor_window: int = 1
    min_delta: float = 0.0
    min_membership: float = 0.1
    fuzziness: float = 2.0
    class_gamma: float = 0.5
    proto_weight: float = 0.1
    alignment_weight: float = 0.2
    centre_weight: float = 0.2
    margin_weight: float = 0.1
    margin: float = 0.2
    contrastive_weight: float = 0.1
    consistency_weight: float = 0.05
    ramp_rounds: int = 5
    temperature: float = 0.07
    graph_lambda: float = 0.1
    graph_eta: float = 0.2
    graph_temperature: float = 0.2
    graph_ridge: float = 0.0001
    tau_count: float = 20.0
    tau_variance: float = 0.0
    reliability_floor: float = 0.3
    tempering_beta: float = 0.5
    prototype_mix: float = 0.6
    overlap_penalty: float = 0.1
    baseline_proto_weight: float = 1.0
    plvm_power: float = 0.75
    dap_intra_weight: float = 1.0
    dap_cross_weight: float = 1.0
    dap_aggregation_temperature: float = 0.001
    soft_proximal: float = 0.01
    soft_counter_floor: float = 0.001
    soft_update_interval: int = 1
    checkpoint_every: int = 1

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        if self.dataset not in DATASETS or self.method not in METHODS:
            raise ValueError('Unknown dataset or method')
        for key in ('alpha', 'encoder_lr', 'head_lr', 'temperature', 'graph_temperature',
                    'graph_ridge', 'dap_aggregation_temperature', 'plvm_power'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('rounds', 'local_epochs', 'batch_size', 'clients_per_domain', 'threads',
                    'clusters', 'monitor_window', 'early_stop_patience', 'checkpoint_every',
                    'soft_update_interval'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if not 0 < self.validation_fraction < 0.5 or self.min_client_samples < 2:
            raise ValueError('Require 0 < validation_fraction < 0.5 and min_client_samples >= 2')
        if self.method.startswith('ours') and not 0 <= self.head_only_rounds <= self.warmup_rounds < self.rounds:
            raise ValueError('Require head_only_rounds <= warmup_rounds < rounds for Ours')
        if self.method.startswith('ours') and self.early_stop_min_rounds > self.rounds - self.warmup_rounds:
            raise ValueError('early_stop_min_rounds exceeds the cluster round budget')
        if self.fuzziness <= 1 or not 0 <= self.min_membership < 1:
            raise ValueError('Invalid fuzzy membership settings')
        if self.amp and self.device != 'cuda':
            raise ValueError('AMP currently requires CUDA')
        if self.backbone not in ('resnet18', 'tiny'):
            raise ValueError('Unknown backbone')
        if self.image_size < 32 or self.workers < 0 or self.max_samples_per_domain < 0:
            raise ValueError('Invalid image size, workers or data cap')
        if not 0 <= self.reliability_floor <= 1 or not 0 < self.tempering_beta <= 1:
            raise ValueError('Invalid reliability parameters')
        if not 0 <= self.prototype_mix <= 1 or not 0 <= self.graph_eta <= 1:
            raise ValueError('Invalid graph/prototype mixing parameters')
        for key in ('graph_lambda', 'tau_count', 'tau_variance', 'min_delta', 'soft_proximal'):
            if getattr(self, key) < 0:
                raise ValueError(f'{key} cannot be negative')
        if self.ramp_rounds < 1 or self.early_stop_min_rounds < 1:
            raise ValueError('Ramp and minimum cluster rounds must be positive')
        if not 0 < self.plvm_power < 1 or not 0 < self.soft_counter_floor < 1:
            raise ValueError('Require plvm_power and soft_counter_floor in (0,1)')
        for key in ('proto_weight','alignment_weight','centre_weight','margin_weight','contrastive_weight',
                    'consistency_weight','baseline_proto_weight','dap_intra_weight','dap_cross_weight',
                    'weight_decay','class_gamma','overlap_penalty'):
            if getattr(self,key) < 0:
                raise ValueError(f'{key} cannot be negative')
        return self

    def identity(self):
        value = asdict(self)
        value.pop('output_root')
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]
