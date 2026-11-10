# FedGAPS prototype geometry: four runs

Extract the entire ZIP. Open Run_DomainNet10_Geometry_4Runs.ipynb and set DATA_ROOT to the existing directory containing domainnet10. Install requirements.txt in your PyTorch environment if needed.

Exactly four FedGAPS runs: Dirichlet alpha 0.1 and 0.5, each with seeds 42 and 123. No ablation runs or export-off duplicates. The previously verified read-only diagnostic implementation is retained unchanged.

Images are resized to 224 pixels.

Defaults: 24 clients (4 per domain: Clipart, Infograph, Painting, Quickdraw, Real, Sketch), ResNet18, 20 discovery rounds, 50 total round budget, cluster early stopping patience 5, batch size 32, workers 4, CUDA mixed precision in the notebook. Each selected discovery checkpoint follows the existing validation selection. Training code is unchanged.

Terminal:

    python run_geometry.py --data-root /path/to/data --output outputs/geometry_4runs --device cuda --workers 4 --batch-size 32 --amp --resume

Plan without training: append --plan-only. Resume uses the same settings; changed settings create different run identities. The four runs are sequential.

Outputs: geometry_per_run.csv (exact per-seed values, pair counts and signed changes), geometry_summary.json (mean and sample SD over two seeds), geometry_table.tex (IEEE one-column table; needs booktabs). To regenerate only the summaries:

    python summarize_geometry.py --output outputs/geometry_4runs

Stages: initial raw prototypes before discovery updates; selected checkpoint raw prototypes before smoothing; same selected checkpoint smoothed prototypes. Pair masks are intersected within each run. Initial and selected checkpoints use the same training partition. Initial does not mean an untrained random encoder: the default encoder is ImageNet pretrained. Graph construction itself does not move prototypes. Changes during discovery are attributable to the complete training procedure, not uniquely to the graph. No increase in different-class distance is assumed or forced. Two-seed SD is descriptive, not a significance test.

Smoke test uses synthetic data, never evidence for DomainNet-10:

    python run_geometry.py --smoke --device cpu --workers 0 --output outputs/smoke

Fixed classes: bird, cat, dog, elephant, horse, airplane, bicycle, car, motorbike, truck. This suite uses class-wise Dirichlet allocation, not the controlled profiles described in the standalone reference README.

Set DATA_ROOT to the parent directory containing domainnet10. The dataset layout is DATA_ROOT/domainnet10/<domain>/<class>/images, with official split files, or DATA_ROOT/domainnet10/<domain>/train/<class> and test/<class>. Without split files or split folders, the existing adapter uses a deterministic stratified split.
