# Baseline implementations and fidelity

These are independent common-protocol implementations, not copies of official
training entrypoints and not claims to reproduce the exact published numbers.
The original eight are runnable; none is an alias of Ours or a stub. The dataset subsets,
architecture, full participation, optimizer settings and validation selection
are shared and differ from some original experiments.

| Method | Implemented behavior | Reference |
|---|---|---|
| FedAvg | Sample-weighted model aggregation, ordinary cross entropy | [AISTATS 2017 paper](https://proceedings.mlr.press/v54/mcmahan17a.html) |
| FedBN | FedAvg excluding all BN affine parameters and running buffers; persistent client BN at training and inference | [Official repository](https://github.com/med-air/FedBN) |
| FedProto | Persistent local models, no model averaging; equal-client per-class prototype aggregation, feature MSE; nearest-global-prototype inference | [Official repository](https://github.com/yuetan031/FedProto) |
| FedPLVM | Local class-wise FINCH, server class-wise FINCH; alpha-powered cosine contrastive term plus paper Eq. 8 correction; model averaging | [Paper](https://arxiv.org/abs/2403.09048) |
| FedDAP | Similarity-weighted class/domain prototypes, own-domain cosine alignment, cross-domain same-class positives and different-class negatives | [Author repository](https://github.com/quanghuy6997/FedDAP_CVPR2026) |
| FedRC | Per-example responsibilities proportional to client mixture weight times class probability divided by cluster label prior; responsibility-weighted local gradients and sample-weighted server updates | [ICML author repository](https://github.com/LINs-lab/FedRC), [equations](https://arxiv.org/html/2301.12379) |
| FPL | Local class means, server FINCH cluster prototypes, multi-positive contrastive loss and MSE toward the unweighted mean of same-class clustered prototypes | [Official repository](https://github.com/WenkeHuang/RethinkFL) |
| FedSoft | Per-example minimum-loss cluster counts, floored importance weights, personalized local proximal updates toward weighted cluster parameters; full-participation weighted aggregation | [AAAI paper and algorithm](https://arxiv.org/html/2112.06053v2) |

The separately requested CSCWD 2024 baseline is **not yet implemented**; its
full algorithm or verified author code is required. See
[the pending baseline record](CSCWD2024_BASELINE.md). It is not counted among
these eight and has no placeholder metrics.

## Source audit

FedDAP's author repository was inspected at commit
`988211eb826d81251bd8b3b8ae53fbdcc7f3dc20` (2026-10-05), including
`models/feddap.py`, `models/fedplvm.py`, `models/fpl.py`, and `utils/finch.py`.
It contains undefined names in FedDAP's constructor/logging, and discrepancies
between some comments, losses and paper summaries. No faulty entrypoint was
vendored. The independent FedDAP loss follows the executable cross-domain
positive/negative construction: **same-class cross-domain nodes are positives**.
Do not describe it as indiscriminately repelling every other-domain prototype.
Attention uses the summed cosine similarities and temperature 0.001 by default.
The incidental prototype perturbation call in that repository is omitted;
this suite does not claim a private/noisy FedDAP reproduction.

For FedPLVM, the loss follows the paper's Eq. 6--8 rather than the FedDAP
repository's altered correction and irregular minibatch accumulation. Local
and global clustering use the last nontrivial FINCH hierarchy. The FINCH
implementation here independently builds exact first-neighbor components and
uses blocked cosine search; it does not depend on or distribute the third-party
FINCH source. Ties, tiny classes and numerical precision can affect partitions.
The sparsity power `plvm_power` is separate from the data Dirichlet `alpha`.

FedRC implements fixed-K vanilla responsibility updates, not the optional
adaptive-cluster deletion or FedRC-Adam variants. Its per-example weights remain
fixed during each local update. Label priors are estimated from global weighted
class counts. FedSoft implements the full-participation case; the original
stochastic cluster-wise client sampling is not enabled. It preserves the
paper's unnormalized counter floor in proximal penalties and server weights.

All prototype baselines extract prototypes after the local optimizer finishes,
using deterministic views, instead of collecting changing-model features from
the final training epoch's minibatches. This makes the extraction convention
consistent and auditable but is a deviation from several public scripts.
FedProto uses a common homogeneous architecture and initialization here; its
original model-heterogeneity experiments are outside this protocol.

Before claiming official reproduction, validate losses and update equations
against the chosen publication version, tune on validation, run the actual
data, and compare an official reference run under its own native settings.
The included tests establish code invariants and execution only.
