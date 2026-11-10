# DomainNet-10 federated prototype experiment

This entry point selects ten fixed DomainNet classes and uses all six domains:
Clipart, Infograph, Painting, Quickdraw, Real, and Sketch.

- Animals: `bird`, `cat`, `dog`, `elephant`, `horse`.
- Vehicles: `airplane`, `bicycle`, `car`, `motorbike`, `truck`.

The first half of the clients in every domain are animal-heavy and the second
half are vehicle-heavy. With four clients per domain, the run creates 24
clients. The desired two clusters must each contain clients from all six
domains.

## Dataset layout

Download the official DomainNet images and split text files. Either of these
layouts is accepted:

```text
data/domainnet/
  clipart/<class>/*
  infograph/<class>/*
  painting/<class>/*
  quickdraw/<class>/*
  real/<class>/*
  sketch/<class>/*
  splits/clipart_train.txt
  splits/clipart_test.txt
  ...
```

or pre-separated folders:

```text
data/domainnet/<domain>/train/<class>/*
data/domainnet/<domain>/test/<class>/*
```

If official split files and explicit train/test folders are both absent, the
loader creates a deterministic class-stratified 80/20 split.

## Recommended first run

```python
import importlib
import run_domainnet10

run_domainnet10 = importlib.reload(run_domainnet10)
main = run_domainnet10.main

main([
    "--device", "cuda",
    "--data-dir", "data/domainnet",
    "--output-dir", "outputs/domainnet10_seed7",
    "--seed", "7",
    "--clients-per-domain", "4",
    "--samples-per-client", "50",
    "--majority-fraction", "0.70",
    "--warmup-rounds", "30",
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

Begin with 100 samples per client because some DomainNet classes are small in
Infograph. The loader raises an explicit capacity error if the requested
partition is not possible; reduce the sample count rather than duplicating
images.
