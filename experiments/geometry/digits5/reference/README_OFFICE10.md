# Office-Caltech-10 federated prototype experiment
After training, inspect `cluster_allocation.json` in the output directory for
hard client counts, soft effective-client counts, and effective samples per
cluster. The corresponding NumPy arrays are also saved; see `README.md`.
This entry point applies the same federated warm-up, domain-robust prototype
learning, graph smoothing, soft client clustering, and cluster-model training
used by `run_digits5.py` to Office-Caltech-10.

## Dataset layout

Place the four ImageFolder-style domains below `data/office10`:

```text
data/office10/
  amazon/<class>/*.jpg
  caltech/<class>/*.jpg
  dslr/<class>/*.jpg
  webcam/<class>/*.jpg
```

Every domain must contain the same ten class directories. The script creates a
deterministic, class-stratified 70/30 train/test split.

## Recommended first run

Office-Caltech-10 is much smaller than Digits-5, especially DSLR and Webcam.
Start with four clients per domain and 20 images per client. This creates 16
clients and avoids duplicating an original image between clients.

```python
import importlib
import run_office10

run_office10 = importlib.reload(run_office10)
main = run_office10.main

main([
    "--device", "cuda",
    "--data-dir", "data/office10",
    "--output-dir", "outputs/office10_main",
    "--clients-per-domain", "4",
    "--samples-per-client", "20",
    "--majority-fraction", "0.7",
    "--warmup-rounds", "50",
    "--head-only-rounds", "2",
    "--loss-ramp-rounds", "5",
    "--local-epochs", "1",
    "--batch-size", "16",
    "--image-size", "224",
    "--workers", "0",
    "--encoder-lr", "0.0003",
    "--head-lr", "0.003",
    "--prototype-lambda", "0.10",
    "--contrastive-lambda", "0.15",
    "--alignment-lambda", "0.25",
    "--separation-lambda", "0.0",
    "--global-centre-lambda", "0.30",
    "--global-margin-lambda", "0.20",
    "--global-margin", "0.35",
    "--consistency-lambda", "0.05",
    "--no-domain-adapters",
    "--domain-lambda", "0.0",
    "--graph-lambda", "0.10",
    "--graph-eta", "0.20",
    "--tempering-beta", "0.50",
    "--prototype-mix", "0.10",
    "--clusters", "2",
    "--fuzziness", "2.0",
    "--validation-fraction", "0.20",
    "--cluster-min-rounds", "5",
    "--cluster-max-rounds", "30",
    "--cluster-patience", "5",
    "--cluster-min-delta", "0.001",
    "--cluster-metric-ema", "0.60",
    "--cluster-monitor-window", "3",
    "--min-membership", "0.10",
])
```

If a class is too small for the requested non-IID allocation, the script stops
with an explicit error. Reduce `--samples-per-client`; do not duplicate images
across clients. The output includes `client_class_distributions.png` and PDF.
