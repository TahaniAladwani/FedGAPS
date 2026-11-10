# Experimental protocol

## Partition and semantic-label definition

Within each domain, for each class, draw a symmetric Dirichlet vector with the
requested alpha over the clients and then multinomial class counts. Each image
is assigned exactly once. Reject allocations containing fewer than
`min_client_samples` observations in a client; record the number of retries.
This minimum-size conditioning slightly changes the ideal unconditioned
Dirichlet law and is explicitly part of the protocol. An optional uniform
per-domain data cap is applied before partitioning; cap 0 uses all training data.

Validation examples are withheld from each client's allocation, approximately
stratified by class. Singleton classes normally stay in training; a client with
only singleton classes contributes one example to validation. Consequently,
very small validation sets can be noisy. Use enough data for stable selection.

The two semantic class groups come from the dataset adapters. A client's
profile is the majority group in its **actual training counts**; ties are -1
and are excluded from semantic ARI/NMI/silhouette. These labels are metadata for
evaluation only. Training and clustering never read them. They are not an
independent ground truth for latent concept shifts. For publication, complement
these diagnostics with a separately specified planted semantic-profile experiment
if semantic discovery is the headline claim. The present code does not silently
reinterpret alpha as a 10% or 50% majority fraction.

Test samples are assigned disjointly among same-domain clients in proportion
to per-class training counts (largest remainder rounding; uniform if absent
from every client). All test images are retained. This defines a personalized
benchmark, without using test performance to choose model routing. The server
does not use any validation or test sample for gradient updates or prototypes.

Image-folder datasets are fingerprinted by ordered labels, resolved paths,
file size and modification time; in-memory image datasets also hash their image
arrays. Fingerprints are change detectors, not a guarantee against content-
preserving timestamp manipulation or near-duplicate images. Source file-path
overlap and repeated image paths are rejected. Serialized Digits data retain
support for the supported preprocessed format.

## Training and fairness

All methods share ResNet-18, initialization seed, transforms, data allocation,
SGD momentum/weight decay, learning rates, local epochs, and all-client
participation. FedRC/FedSoft use separate initialized cluster heads to break
symmetry. FedProto keeps models personal. Ours retains its class-balanced loss,
two-view contrastive/consistency losses and equal-total-domain discovery
aggregation. These are method-specific differences, not changes in data.

The domain adapters and adversarial domain head from the exploratory source are
not enabled. This follows its recommended shared-encoder/no-adapter setting.
Ours' local minibatch separation loss is omitted in favor of the global
prototype margin objective. The frozen-FMCL diagnostic is not rerun as
a baseline. Dense class-domain graph solves and distances use the reference
implementation; fuzzy-medoid updates use a consistent squared-distance objective,
unique medoids and memberships recomputed after the last medoid update.

Small memberships below `min_membership` are removed, except that every client
retains its largest membership, and the remaining row is renormalized. These
same memberships determine training weights and inference mixtures. Effective
sample counts are reported after this threshold. Empty unused clusters are
possible under hard argmax summaries; soft effective counts are also reported.

No encoder features are cached across training rounds because the encoder is
changing. Global prototypes are fixed targets within a round. Baseline
prototypes are recomputed after local training on deterministic training views;
Ours recomputes signatures after model aggregation so client prototypes share
a coordinate system. That extra broadcast is charged to Ours' communication.

## Selection and stopping

FedAvg/BN/Proto/PLVM/DAP/FPL/RC/Soft, Ours' warm-up, and Ours' individual
cluster stages all use the same stopping defaults: patience 5, minimum round 1,
window 1, and minimum improvement 0. The score is (balanced accuracy + worst
observed class recall)/2 on validation. Baselines and warm-up pool the validation
confusions; cluster stages weight client validation confusions by memberships.
A tie or decrease counts as a bad round. Any increase resets patience. The
first observation establishes the best checkpoint, so a constant score stops
at round 6. No minimum-round exemption delays that decision.

Ours' warm-up has a maximum of `warmup_rounds`. On patience exhaustion or the
cap, restore its best validation model, recompute signatures, and then cluster.
Each cluster has a maximum of `rounds - warmup_rounds` training rounds, counted
from the **actual** end of warm-up. Unused warm-up rounds do not increase that
budget. All baselines stop by patience or the `rounds` cap and restore their
best state, including any personalized models, prototypes and BN state.

The explicit `ours_fixed_rounds` ablation disables only cluster-stage stopping;
it still restores the best validation checkpoints and uses warm-up patience 5.
Changing `monitor_window` applies consistently to every stage, although the
default plans use raw per-round scores (window 1). For larger windows, the best
checkpoint is the last round of the winning window, not an averaged model.

Checkpoint state includes the counters, best state, actual warm-up end, phase
transition and terminal stop decision. Interrupted transitions resume without
rerunning warm-up or adding an extra round after termination. Final test
inference uses only selected models and is performed after training.

Round budgets alone do not equalize compute: multi-expert methods can perform
K local updates per client, and Ours has two-view discovery. Communication and
elapsed training time are therefore included. Time includes validation,
prototype passes and earlier checkpoint work, but excludes dataset preparation,
model initialization and final test evaluation. It is not a pure GPU timing.

## Metrics

All accuracies are fractions, not percentages. Overall accuracy is micro
accuracy. Balanced accuracy and worst-class recall consider classes observed
in the pooled test set, with absent recalls serialized as null. Worst-domain
and domain gap use observed-domain accuracies. Confusion matrices and supports
make aggregation auditable. The main predictions are personalized-client
predictions, not an oracle choice of the best expert on each test example.

ARI/NMI use argmax native cluster memberships. Ours silhouettes use its
prototype/support distance; FedRC/FedSoft silhouettes use membership-vector
Euclidean distance. `silhouette_space` records this distinction. These spaces
are method-specific diagnostics and must not be described as a controlled
cross-method representation comparison. Allocation balance is entropy of hard
cluster counts divided by log(K); also report soft client and sample totals.

Traffic is uncompressed tensor payload bytes for the simulated training
protocol, including model/BN exclusions, prototype labels, and graph metadata.
Initialization is assumed installed on clients. Actual network framing,
serialization overhead, optimizer states, validation reporting and final model
deployment are excluded. This is a payload accounting estimate, not measured
network traffic. There is no claim of differential privacy or secure aggregation.

Aggregate separately by dataset, alpha, algorithm and every other configuration
field. Report mean +/- sample standard deviation over 42, 123, 2024; include
seed count. Hyperparameters must be selected using validation, ideally a
separate tuning setup, before inspecting final test comparisons. This suite
does not automatically search hyperparameters or establish a statistically
significant advantage for any algorithm.
