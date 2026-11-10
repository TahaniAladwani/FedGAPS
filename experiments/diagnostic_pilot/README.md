# FedGAPS: matched Digits-5 diagnostic pilot

Open `Run_Digits5_Diagnostic_Pilot.ipynb` from this directory, or use:

```bash
python -m pip install -e .
python pilot.py --data-root /path/to/data --device cuda --output outputs/diagnostic_pilot
python plot_diagnostics.py --output outputs/diagnostic_pilot
```

The data root contains `digits5/`. Reuse your existing prepared dataset. CUDA-compatible
PyTorch and torchvision should already be installed. No data is bundled.

## Protocol

Digits-5, Dirichlet alpha 0.1, seed 42, four clients per domain (20 clients).
The three variants are FedGAPS, no graph, and no reliability weighting. All use the
same partition and configuration apart from the method. Defaults match the benchmark:
ResNet-18 pretrained on ImageNet, batch 64, image size 96, one local epoch, at most
20 warm-up rounds plus 30 cluster rounds, validation patience 5, AMP disabled.
This pilot retains early stopping; it is not the fixed-round ablation.

Each variant runs with diagnostics off and on: six executions total. The runner checks
exact equality of the partition, histories, selected parameters, cluster memberships,
distances, test metrics, and per-example test probabilities. It writes a verification
report only for completed successful comparisons. Check that all three entries are
present. Any numerical mismatch raises an error; do not claim verification if it fails.
Wall time is excluded from equality checks because export work adds overhead.

```bash
python pilot.py --smoke --device cpu --output outputs/pilot_smoke
```

The CPU synthetic check passed for all three variants during packaging. Real Digits-5
and CUDA equivalence must be checked on the target machine. The synthetic check is
not an accuracy experiment. Use a new output directory for new settings; `--resume`
continues interrupted runs with unchanged code, data, and configuration. Do not resume
old benchmark checkpoints in this modified package.

## Diagnostic files

Every export-on run contains `diagnostics/initial`, `diagnostics/round_NNN`, and
`diagnostics/selected` JSON/NPZ pairs. Selected means the restored best warm-up
checkpoint, not necessarily the last executed round.

NPZ files contain raw/smoothed class-domain nodes, masks, reliability, edge weights,
centroids, per-client prototypes, counts, dispersion, supports, and client distances.
JSON files contain labels, client/domain IDs, checkpoint rounds, and mean cosine
geometry summaries. The ordinary run outputs contain selected memberships and
semantic/domain clustering metrics.

Initial exports use a copied model and isolated Python/NumPy/PyTorch RNG states.
Warm-up exports reuse already-computed statistics. Selected exports reconstruct a
graph for reporting without replacing training state. Extra diagnostics do not count
as simulated protocol communication; export-on runtime includes their computational
cost and should not be used for efficiency comparisons.

In no-graph runs, saved affinities are diagnostic only: graph guidance is disabled.
The no-graph ablation also disables graph-specific local losses, as in the benchmark;
it is not a smoothing-only intervention. Graph weights are computed from prototypes,
not independent trainable parameters. Missing nodes are identified by the mask.
Small raw-to-smoothed distances alone do not establish better generalization.

## Returning results

Share the verification report and the three export-on run folders' metrics, partitions,
histories, distance/membership arrays, and diagnostics. Checkpoints, raw data, and
per-example test probability files are unnecessary for the diagnostic analysis.

The benchmark implementation notes remain in `docs/`. No license is assigned.
