# FedGAPS

Graph-Aligned Prototypes for Clustered Federated Learning.

![FedGAPS overview](docs/assets/overview.png)

FedGAPS learns a shared discovery encoder, builds same-class cross-domain
prototype graphs, and uses reliability-tempered client signatures for fuzzy
clustering. Cluster models are trained and selected independently using validation
data. Predictions combine cluster probabilities with client memberships.
The memberships and curves in the overview are schematic.

## Installation

Use Python 3.10 or newer. Install PyTorch and torchvision for your CUDA environment
first, then install the project from this directory:

```bash
python -m pip install -e .
python -m icde --help
```

## Data

Datasets are not included. Place data under `data/digits5`, `data/office10`,
`data/domainnet10`, or `data/domainnet40`. Dataset adapters live in `reference/`.
Digits-5 setup is described in [docs/DIGITS5_SETUP.md](docs/DIGITS5_SETUP.md).
See [the Office-10 adapter guide](reference/README_OFFICE10.md) for supported
folder layouts and source splitting.
DomainNet supports official domain split lists and image-folder layouts.
Check the loader paths against your dataset before starting a suite.

## Experiments

The dataset configurations use three seeds (42, 123, 2024), two Dirichlet
concentrations (0.1, 0.5), and four clients per domain. They use ResNet-18,
50 maximum rounds, and validation patience 5. AMP is disabled by default.
Edit the JSON configurations to change device, data paths, or seed lists.

Plan a suite first; add `--execute` to run it:

```bash
python -m icde sweep --config configs/digits5.json --ablations
python -m icde sweep --config configs/digits5.json --ablations --execute
python -m icde sweep --config configs/office10.json --ablations --execute
python -m icde sweep --config configs/domainnet10.json --ablations --execute
```

Each suite contains 90 runnable experiments with ablations (54 without).
CSCWD client-prototype clustering is recorded as pending and is not implemented.
Use `--resume` for interrupted runs with unchanged source and configuration.
For a CPU run, add `--device cpu`. Summary output is written alongside the runs.

```bash
python -m icde summarize --output-root outputs/digits5_patience5
python -m icde smoke --output-root outputs/smoke
```

The smoke command uses synthetic data and a small network to check execution;
it does not reproduce real-data accuracy. Dataset suites use real data only.
The summary completeness flag currently expects the supplied three seeds.

## Protocol and methods

See [PROTOCOL.md](docs/PROTOCOL.md) for partitioning, model selection,
communication accounting, and metric definitions, and
[BASELINES.md](docs/BASELINES.md) for implementation details.

Baselines: FedAvg, FedBN, FedProto, FedPLVM, FedDAP, FedRC, FPL, and FedSoft.
These are independent common-protocol implementations, not official author code.
Six ablations remove graph guidance, reliability weighting, soft memberships,
the global-centre term, the margin term, or cluster-stage early stopping.

The supplied suite uses Dirichlet allocation, without controlled minority
injection. DomainNet-10 and DomainNet-40 are separate dataset options.
The overview illustrates the intended mechanism; it is not evidence that
semantic clusters are recovered in every experiment.

## Repository

- `icde/`: training, evaluation, configurations, and command-line interface.
- `reference/`: dataset adapters and prototype/discovery routines.
- `configs/`: reproducible experiment settings.
- `docs/`: protocol and baseline notes.

This release excludes data, checkpoints, result archives, and executed notebooks.
No license is assigned in this package; choose one after confirming rights to
all included code and figure assets.
