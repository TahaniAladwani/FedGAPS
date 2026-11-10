# FedGAPS prototype geometry: four runs

Extract the entire ZIP. Open Run_Geometry_4Runs.ipynb and set DATA_ROOT to the existing directory containing digits5. Install requirements.txt in your PyTorch environment if needed.

Exactly four FedGAPS runs: Dirichlet alpha 0.1 and 0.5, each with seeds 42 and 123. No ablation runs or export-off duplicates. The previously verified read-only diagnostic implementation is retained unchanged.

Defaults: 20 clients, ResNet18, 20 discovery rounds, 50 total round budget, cluster early stopping patience 5, batch size 64, workers 4, CUDA mixed precision in the notebook. Each selected discovery checkpoint follows the existing validation selection. Training code is unchanged.

Terminal:

    python run_geometry.py --data-root /path/to/data --output outputs/geometry_4runs --device cuda --workers 4 --batch-size 64 --amp --resume

Plan without training: append --plan-only. Resume uses the same settings; changed settings create different run identities. The four runs are sequential.

Outputs: geometry_per_run.csv (exact per-seed values, pair counts and signed changes), geometry_summary.json (mean and sample SD over two seeds), geometry_table.tex (IEEE one-column table; needs booktabs). To regenerate only the summaries:

    python summarize_geometry.py --output outputs/geometry_4runs

Stages: initial raw prototypes before discovery updates; selected checkpoint raw prototypes before smoothing; same selected checkpoint smoothed prototypes. Pair masks are intersected within each run. Initial and selected checkpoints use the same training partition. Initial does not mean an untrained random encoder: the default encoder is ImageNet pretrained. Graph construction itself does not move prototypes. Changes during discovery are attributable to the complete training procedure, not uniquely to the graph. No increase in different-class distance is assumed or forced. Two-seed SD is descriptive, not a significance test.

Smoke test uses synthetic data, never evidence for Digits-5:

    python run_geometry.py --smoke --device cpu --workers 0 --output outputs/smoke
