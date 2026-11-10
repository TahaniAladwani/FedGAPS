# Digits-5 federated prototype experiment
After training, inspect `cluster_allocation.json` in the output directory for
hard client counts, soft effective-client counts, and effective samples per
cluster. The corresponding NumPy arrays are also saved; see `README.md`.
`run_digits5.py` applies the same shared encoder, class-domain graph,
cross-domain alignment, class separation, residual domain adapters, and soft
client clustering used by the CIFAR-10 experiment.

The five domains are MNIST, SVHN, USPS, SynthDigits, and MNIST-M. MNIST, SVHN,
and USPS download automatically through torchvision. Place SynthDigits and
MNIST-M under `data/digits5` in either supported format:

```text
data/digits5/
  synth/
    train/0/*.png ... train/9/*.png
    test/0/*.png  ... test/9/*.png
  mnistm/
    train/0/*.png ... train/9/*.png
    test/0/*.png  ... test/9/*.png
```

Alternatively, each domain directory may contain trusted `train.pt` and
`test.pt` files. Each file must store `(images, labels)` or a dictionary using
`images`/`data` and `labels`/`targets`. Never load untrusted pickle files.

The preprocessed Digits-5 release linked by the FedBN authors is an appropriate
source. Convert its MNIST-M and SynthDigits splits to either layout above if its
download uses a different partition layout.

Each domain is divided into an equal number of low-digit-heavy and
high-digit-heavy clients. The desired client partition therefore follows digit
prevalence rather than dataset domain.

## Notebook usage

```python
from run_digits5 import main

main([
    "--device", "cuda",
    "--data-dir", "data/digits5",
    "--output-dir", "outputs/digits5_main",
    "--clients-per-domain", "6",
    "--samples-per-client", "600",
    "--majority-fraction", "0.7",
    "--warmup-rounds", "50",
    "--head-only-rounds", "2",
    "--loss-ramp-rounds", "5",
    "--local-epochs", "1",
    "--batch-size", "64",
    "--image-size", "96",
    "--workers", "0",
    "--encoder-lr", "0.001",
    "--head-lr", "0.01",
    "--prototype-lambda", "0.10",
    "--contrastive-lambda", "0.15",
    "--alignment-lambda", "0.25",
    "--separation-lambda", "0.0",
    "--separation-margin", "0.50",
    "--global-centre-lambda", "0.30",
    "--global-margin-lambda", "0.20",
    "--global-margin", "0.35",
    "--consistency-lambda", "0.05",
    "--adapter-bottleneck", "64",
    "--adapter-lambda", "0.005",
    "--no-domain-adapters",
    "--graph-lambda", "0.10",
    "--graph-eta", "0.20",
    "--tempering-beta", "0.50",
    "--prototype-mix", "1.0",
    "--validation-fraction", "0.10",
    "--cluster-min-rounds", "5",
    "--cluster-max-rounds", "30",
    "--cluster-patience", "5",
    "--cluster-min-delta", "0.001",
    "--cluster-metric-ema", "0.60",
    "--cluster-monitor-window", "3",
])
```

With five domains and `--clients-per-domain 6`, the experiment creates 30
clients. Within every domain, the first three clients receive 70% of their
samples from digits 0--4 and 30% from digits 5--9; the other three receive the
opposite distribution. The run writes `client_class_distributions.png` and
`client_class_distributions.pdf`, containing one ten-class bar chart per
client. Bar colour identifies the semantic profile and hatch identifies the
source domain.

USPS is the limiting training domain, so begin with 600 samples per client.
The script raises an explicit error if a requested class allocation exceeds a
domain's available examples.

The principal success criteria are a positive raw semantic geometry gap, high
ARI/NMI against low/high-digit prevalence, low ARI/NMI against the five domain
identities, and strong worst-domain accuracy.
