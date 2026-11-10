# Release validation

Checked on 10 November 2026.

- All 200 packaged Python sources parsed successfully.
- All nine notebooks passed Python-cell syntax checks (not real-data execution).
- All 15 main methods completed the CPU synthetic smoke suite: eight baselines, FedGAPS, and six ablations.
- All three sensitivity planners produced 16 unique configurations, with the intended dataset and two seeds.
- Geometry, collaborator-study, and diagnostic-pilot CLI imports passed.
- All ten collaborator-study unit tests passed, including finite sampling, small-domain fallback, Hellinger distance, grouping, and communication ledger checks.
- Packaged Python source hashes match the source manifest.

Runtime checks used Python from the local Anaconda environment with PyTorch 2.11.0 on CPU, with OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. CUDA/PyTorch 2.5.1 and full real-data training were not rerun for this release. Synthetic execution does not verify paper accuracy or equivalence to official baseline implementations.

No datasets, trained checkpoints, or experimental result folders are bundled.
