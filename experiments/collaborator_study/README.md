# Office-10 small-domain fix

This copy preserves the original project and outputs. Open `Run_Collaborator_Study.ipynb` here. Defaults are Office-10, planted only, two seeds, and a new `office10_planted_capacity_v2` output. Do not rerun completed Dirichlet jobs.

The old smallest-class 50% reserve can give fewer than ten samples/client even when a feasible allocation exists. The revised sampler preserves the old allocation when that bound is at least ten. Otherwise, it attempts a complete ten-sample/client multinomial allocation, rejecting every draw that exceeds any class pool. It never duplicates images, drops classes, or clips class counts. Missing classes, fewer than 40 total training examples per domain, or failure after 10,000 attempts remain explicit errors.

The preflight prints real class capacities and checks both seeds before training. `partition.json` records available/selected class counts, actual samples/client, fallback use, and rejected draws. Samples are drawn from the generating profiles **conditioned on available class capacities**. That conditioning can materially affect tiny-domain mixtures; report the realized counts and do not claim unconstrained multinomial sampling. Ten samples is before validation, so some clients will be very small and estimates may be variable.

This changes the finite planted partition protocol, not the Dirichlet construction. Keep old results separately. New source hashes intentionally prevent resuming old training in this copy. Do not copy old checkpoints into the new output.

The preflight may still identify genuinely missing or insufficient data; that should be corrected at the dataset level rather than bypassed.

---

# Matched collaborator-selection study

This package implements two separate experiments. Default dataset: **DomainNet-10**, seeds **42 and 123**. It supports Digits-5 and Office-10 as well. It preserves previous experiment folders.

## Jobs

| Suite | Per seed | Two seeds |
|---|---|---|
| Dirichlet alpha=0.1 | Aligned discovery + no-graph discovery + four downstream comparisons | 12 |
| Planted profiles | One shared discovery + eight grouped downstream comparisons + two FedAvg controls | 22 |
| Both, from scratch | 17 jobs | **34** |

There are **28 final performance results**, plus six discovery jobs. A cluster-model update is not counted as a separate experiment: one downstream experiment may train multiple cluster models.

If the old 20-job planted suite is complete and passes the reuse checks, only its **two histogram-only downstream jobs** are new. Combining this extension with the fresh 12-job Dirichlet suite requires **14 new training jobs**, representing 34 logical jobs including the reused ones. Reuse requires the same dataset, partition, training settings, source version, and both seeds. A Digits-5 checkpoint cannot be used for DomainNet-10.

## Dirichlet suite: four distance comparisons

Keep alpha=0.1, K=2, fuzzy medoids, fuzziness=2, membership threshold=0.1, the partition, optimizer, training budgets, and prediction rule identical. The default uses four clients per domain (24 clients for six-domain DomainNet-10).

1. **counts:** Hellinger distance between normalized training class counts: `||sqrt(p)-sqrt(q)|| / sqrt(2)`. This follows the requested predefined histogram distance, not Jensen–Shannon divergence.
2. **reliability_counts:** the existing Jensen–Shannon support-distance term on reliability-tempered supports, using the original base-2 implementation. Prototype and coverage/overlap terms are excluded. The omitted global coefficient `(1-prototype_mix)` only rescales the whole distance and does not affect fuzzy medoids or its memberships.
3. **unaligned:** full proposed distance from signatures of a separately trained `ours_no_graph` discovery encoder. This retains the original no-graph ablation: it disables graph smoothing/guidance but retains the other discovery losses. It is not a classifier-only encoder.
4. **aligned:** full proposed distance from the graph-guided shared discovery encoder.

Train each discovery encoder for 20 rounds and select its checkpoint using validation only. **All four downstream conditions start from the exact same complete aligned model weights, including the classifier.** The unaligned encoder is used only to compute its signatures. The full distance still uses client prototypes extracted from the selected encoder; it does not substitute the server's smoothed prototype targets for client signatures.

Run 30 downstream rounds with identical settings. Patience is 51 to fix the round budget; validation still selects each cluster's final checkpoint. Prediction mixes cluster probabilities using the actual thresholded/renormalized memberships. The selected discovery round may differ between aligned and no-graph discovery, but their training budgets are identical. Conditional on a common initialization, any downstream difference is due to the resulting assignments, not different initial model weights.

This experiment deliberately uses two signature encoders for the graph ablation. Counts, reliability-weighted counts, and full aligned signatures all use the common aligned discovery information. The unaligned row is the stated exception to the same-signature-encoder condition; its downstream initialization remains shared.

There is no known planted group in a Dirichlet partition. **Planted ARI/NMI and neighbor recovery are not reported for this suite.** Training-histogram proxies are not silently relabeled as ground truth. Domain MI/NMI/ARI can still be reported.

## Planted suite: generating profiles and neighbor recovery

For each domain, assign two clients to G0 and two to G1 before drawing examples. G0's generating distribution assigns 70% probability to the first half of the ordered classes and 30% to the second half, uniformly within each half. G1 reverses these probabilities. Domain and planted group are therefore independent at the client level. Group IDs are used only to evaluate discovery, never to compute distances or train models.

The default `--planted-sampling finite` draws finite multinomial class counts from these profiles and allocates disjoint real images. It does **not** define the group label from the observed histogram. Generating profiles are exported in partition.json. The sampler reserves class capacity; infeasible domain draws are rejected without clipping classes or duplicating images. This is sampling subject to available dataset capacity, not an unconstrained infinite population. Some small clients may lack classes; masks/counts handle this normally.

Client-size targets have a predeclared domain factor ranging from 0.5 to 1.5 in loader order, around `--samples-per-client 500` before validation splitting. The target is capped by available class capacity. All clients within a domain have the same pre-validation size. This controlled quantity skew makes the domain-balanced FedAvg control meaningful. Audit files expose actual sample counts and flag if domain totals end up equal. The quantity skew is not claimed to occur naturally.

The planted suite reuses **one** selected aligned discovery checkpoint per seed for:

- Aligned FedGAPS fuzzy medoids at K=1,2,3,4.
- Aligned agglomerative, spectral, and K-means comparisons at K=2.
- **Counts-only Hellinger distance at K=2**, with exactly the same fuzzy procedure and downstream initialization/settings as aligned FedGAPS K=2.
- Standard FedAvg and domain-balanced FedAvg, each trained for 50 rounds from the same seeded original initialization. These two controls do not reuse the trained discovery checkpoint. Domain-balanced aggregation gives each domain equal total weight, distributing that weight among its clients in proportion to their training sizes.

K-means uses Euclidean distance on the existing reliability-tempered signature vectors. Agglomerative uses average linkage on the proposed distance; spectral uses its RBF affinity with the median positive distance as scale. These are clustering algorithm controls, not official PACFL/FMCL implementations. The fuzzy K=2 histogram/full comparison changes only the client distance.

**Cross-domain precision@3:** exclude every client from the query client's own domain, then measure how many of the three nearest neighbors have the same *generating* group. Boundary ties are averaged to avoid favoring client-ID order; k is fixed and never silently reduced. Each run exports per-client `neighbors.csv`. Clustering ARI/NMI use argmax of the actual thresholded memberships. Domain MI uses natural logs (nats); domain NMI and ARI are also reported. K=1 has zero domain MI but zero planted recovery and must not be presented as successful discovery.

The planted setting is **not Dirichlet alpha=0.1**. Do not combine their scores into a single mean.

## Cost accounting

Every result includes upload and download bytes in these components:

- `discovery_models`: local training uploads/downloads plus the shared-encoder broadcast used to collect each discovery round's signatures.
- `discovery_summaries`: prototype/count/dispersion uploads and graph-target downloads during discovery.
- `selection_models` and `selection_summaries`: selected-checkpoint signature extraction and membership delivery.
- `downstream_models`: **every** active cluster-model download/upload for every client, not one model per client by assumption.
- `auxiliary_discovery_models` and `auxiliary_discovery_summaries`: the additional no-graph discovery required by the unaligned row in this matched experiment.

Fresh runs use an explicit dense summary schema: float32 prototypes, int64 counts, float32 dispersions, and one int64 domain ID per client. Counts-only can reuse counts already uploaded during shared discovery. With the default tau_variance=0, reliability-tempered supports can also be computed from those cached counts. Thus these two comparisons need no additional selection-summary upload; they still pay all shared discovery costs. Full selected signatures use a selected-model broadcast and a new summary upload, matching the extraction procedure. Membership matrices are accounted as float64 downloads.

These are **uncompressed training payload estimates**, not measured network traffic. Validation/test traffic, framing, serialization overhead, and control packets are excluded. Model sizes include the transmitted state tensors. The single-machine simulator does not measure real distributed latency. Elapsed seconds and separate CPU distance/clustering times are provided. Cached full-distance computation is charged its measured construction time, not merely the time to copy the cached matrix.

The unaligned row pays for both the aligned common initialization and the auxiliary no-graph signature encoder. This is the cost of this controlled experiment, not a claim that an independently deployed no-graph method would need both. Shared costs are attributed once to every standalone result; do not sum result-row costs to estimate actual sweep runtime.

This matched experiment tests whether richer summaries improve grouping **after shared discovery**. It does not show savings or accuracy for a counts-only pipeline that skips discovery. Such a pipeline would require another explicitly separate experiment. The package also does not add a client-count scalability sweep; the requested 34 jobs have fixed client counts.

## Running

Use your existing CUDA-enabled PyTorch environment. Do not reinstall PyTorch just for this package. The source uses APIs available in PyTorch 2.5.1; the local execution checks used CPU PyTorch 2.11.0, not your exact GPU environment.

Extract the **entire ZIP**, open `Run_Collaborator_Study.ipynb`, set DATA_ROOT to the parent of the dataset folder, and run cells in order. DomainNet-10 needs all six domains in the existing data layout. For example, DATA_ROOT may be `/home/tahani/Paper4/ICDE_benchmark 6/data`.

```bash
python run_study.py --dataset domainnet10 --suite both --data-root "/path/to/data" --output outputs/domainnet10_study --device cuda --workers 4 --batch-size 32 --amp --resume
```

Use `--suite dirichlet` for only the 12-job experiment, or `--suite planted` for the 22-job planted experiment. Use `--plan-only` to print/save the plan without loading datasets or training. For Digits-5, use `--dataset digits5 --batch-size 64` and a separate output directory. Plans and source hashes prevent unsafe resume after a setting/source change. Resume uses completed results and round checkpoints; an output lock prevents concurrent writers.

## Reusing the previous 20-job planted package

```bash
python run_study.py --dataset digits5 --suite planted --planted-sampling quota --reuse-planted "/path/to/old/outputs/planted" --data-root "/same/data/root/as/before" --output outputs/digits5_histogram_extension --device cuda --workers 4 --batch-size 64 --amp --resume
```

Replace arguments to match the **actual old settings**. Reuse is read-only and requires complete `.pt` discovery/last/selected checkpoints and metrics from the previously delivered package, not just the small result ZIP. It verifies source version, configuration, data fingerprint/manifest, common encoder, and exact memberships before importing downstream predictions/accuracy. It never silently substitutes incompatible results.

The old package used exact 70/30 quotas. Reuse therefore requires `--planted-sampling quota`, keeps those exact old partitions, and does not claim multinomial sampling. Groups were still assigned before construction, but histogram-only can be near-perfect by construction; treat this as a strong sanity control, not an unbiased demonstration that prototypes should win. To evaluate finite-sample profile recovery, use a fresh finite suite instead of reusing old quota results.

Legacy discovery totals are decomposed using the known fixed model-transfer schedule and recorded counters; this provenance is exported. New selection exchanges are costed explicitly. Legacy distance-computation timing is recomputed as a diagnostic, not represented as historical timing. Existing training accuracy/results are unchanged. Imported results are copied into the new report directory; old data, code, and checkpoints are not modified.

## Results to return

- `summary.csv`: all measures, mean and sample SD with actual completed seed counts.
- `per_run.csv`: per-seed results.
- `paired_differences.csv`: differences versus reliability-weighted counts for Dirichlet, and counts-only for planted fuzzy K=2. Accuracy differences are percentage points.
- `communication_per_run.csv`: separate upload/download components and totals in decimal GB.
- `dirichlet_results.tex`, `planted_results.tex`: LaTeX tables (requires booktabs).
- `audits/`: client manifests, partition hashes, realized mixtures, domain sizes.
- Per-run `comparison.json`, `neighbors.csv`, `distance.npy`, and `membership.npy`.

Reports update after completed jobs. Missing planted metrics are blank, not zero. A row with n_seeds=1 is partial and has no sample SD. Two paired seeds support descriptive differences, not a reliable significance claim. Neighbor recovery, ARI/NMI, and downstream accuracy may disagree; report that rather than selecting only favorable measures.

## Smoke check and missing-file recovery

```bash
python -m unittest discover -s tests -v
python run_study.py --dataset synthetic --suite both --device cpu --workers 0 --batch-size 32 --output outputs/smoke --resume
```

Synthetic mode uses a tiny encoder and four total rounds. It validates execution, not research performance. If the upload omitted source files, run `python repair_sources.py` in the extracted project. Different existing sources are backed up; data, outputs, and notebooks are preserved.
