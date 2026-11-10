# DomainNet-10 sensitivity: 16 runs

Open Run_DomainNet10_Sensitivity.ipynb after extracting the entire package. Set DATA_ROOT to the directory containing domainnet10. Use your existing PyTorch CUDA environment; requirements.txt lists dependencies.

Fixed: Dirichlet alpha 0.1, seeds 42 and 123, four clients per domain (24 total), ImageNet-pretrained ResNet18, image size 224, batch size 32, workers 4. Notebook enables CUDA AMP. 50 total round budget, 20 discovery rounds, one local epoch, cluster early-stopping patience 5. Cluster count remains 2. Existing engine and checkpoint selection are unchanged.

Graph lambda: 0.01, 0.05, 0.1, 0.2, 0.5 with beta fixed to 0.5. Beta: 0.25, 0.5, 0.75, 1.0 with lambda fixed to 0.1. The shared default is counted once per seed: 8 configurations x 2 seeds = 16 runs. Beta=1 does not disable all reliability weighting. This package does not import previous default runs automatically.

Terminal:

    python run_sensitivity.py --data-root /path/to/data --output outputs/domainnet10_sensitivity --device cuda --workers 4 --batch-size 32 --amp --resume

Append --plan-only to inspect without training. Runs are sequential. Resume continues incomplete runs or reads completed results using the existing engine. Changing settings requires a new output directory to prevent mixed summaries. Diagnostic exports are disabled; no unnecessary export-off/on duplicates. Partitions are checked for equality across configurations within each seed after each run.

Outputs: plan.json, per-run configs/checkpoints/metrics, partition_check.json, sensitivity_per_run.csv updated after every run. Once all 16 finish: sensitivity_summary.csv, sensitivity.png and sensitivity.pdf (mean +/- sample SD across two seeds). The same default appears in both plots, but was trained only once per seed. The default is marked; do not select a new default using test performance.

To regenerate summaries:

    python summarize_sensitivity.py --output outputs/domainnet10_sensitivity

If folder transfer loses files, upload repair_sensitivity_sources.py alone and run:

    python repair_sensitivity_sources.py --project "/path/to/FedGAPS_DomainNet10_Sensitivity_16Runs"

This restores source files, backs up changed source files, and leaves data/results/notebooks untouched.

Synthetic smoke test (not paper results):

    python run_sensitivity.py --smoke --device cpu --workers 0 --output outputs/smoke

DomainNet-10 uses all six domains: clipart, infograph, painting, quickdraw, real, sketch. Fixed classes: bird, cat, dog, elephant, horse, airplane, bicycle, car, motorbike, truck. DATA_ROOT must contain domainnet10. For your existing benchmark data, set DATA_ROOT = Path("/home/tahani/Paper4/ICDE_benchmark 6/data"). Cluster count remains 2.
