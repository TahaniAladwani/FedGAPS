# DomainNet-40: 1,000 distinct samples per client

`run_domainnet40.py` extends the semantic client-clustering experiment to 40
classes and all six DomainNet domains. It automatically selects 20
well-supported natural/living classes and 20 well-supported
manufactured/mobility classes from the official training lists.

The default experiment creates 24 clients:

- 6 domains × 4 clients;
- 1,000 distinct training images per client;
- two 70/30 natural-heavy clients and two 30/70 manufactured-heavy clients
  per domain;
- no image shared by two clients;
- pooled availability-aware sampling inside each semantic group;
- 100 additional, disjoint validation images per client.

## Preflight

Run this before training:

```python
from pathlib import Path
import importlib
import run_domainnet40

run_domainnet40 = importlib.reload(run_domainnet40)
data_dir = Path("data/domainnet").resolve()

report = run_domainnet40.preflight(
    data_dir,
    clients_per_domain=4,
    samples_per_client=1000,
    validation_fraction=0.10,
)

print("Natural/living classes:", report["natural_classes"])
print("Manufactured/mobility classes:", report["manufactured_classes"])
for domain, values in report["domains"].items():
    print(domain, values)
print("Feasible:", report["feasible"])
```

Do not start training unless `Feasible: True` is printed.

## Training

```python
main = run_domainnet40.main

main([
    "--device", "cuda",
    "--data-dir", str(data_dir),
    "--output-dir", "outputs/domainnet40_1000_seed7",
    "--seed", "7",
    "--clients-per-domain", "4",
    "--samples-per-client", "1000",
    "--majority-fraction", "0.70",
    "--warmup-rounds", "50",
    "--head-only-rounds", "2",
    "--loss-ramp-rounds", "5",
    "--local-epochs", "1",
    "--batch-size", "32",
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
    "--overlap-penalty", "0.10",
    "--clusters", "2",
    "--fuzziness", "2.0",
    "--min-membership", "0.10",
    "--validation-fraction", "0.10",
    "--cluster-min-rounds", "5",
    "--cluster-max-rounds", "30",
    "--cluster-patience", "5",
    "--cluster-min-delta", "0.001",
    "--cluster-metric-ema", "0.60",
    "--cluster-monitor-window", "3",
])
```

The selected class list is written into `metrics.json`, ensuring the exact
experimental task is recorded for every seed and dataset release.
