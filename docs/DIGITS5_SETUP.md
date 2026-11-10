# Fix: missing SynthDigits / MNIST-M

The MNIST HTTP 404 is harmless when torchvision successfully downloads from its
S3 fallback. The fatal missing `synth/train` error means the two additional
Digits-5 domains have not been installed.

From the project directory, using the same Python environment as the notebook:

```bash
python -m icde.prepare_digits5 --root data/digits5
```

The updated `Run_Real_digits5.ipynb` runs this command before preflight. It
fetches a ~278 MB archive, checks its SHA256, and converts the SynthDigits and
MNIST-M splits into non-pickled NumPy files. Existing MNIST/SVHN/USPS torchvision
caches are preserved. Training partitions 0–9 are combined before the suite's
own shared Dirichlet partitioning; test data remains separate. This uses all
released training partitions, not FedBN's default 10% subsample.

An offline download can be used with:

```bash
python -m icde.prepare_digits5 --root data/digits5 --archive /path/to/digit_dataset.zip
```

Source: [FedBN author repository](https://github.com/med-air/FedBN), its
[dataset reader](https://github.com/med-air/FedBN/blob/master/utils/data_utils.py),
and the linked [dataset mirror](https://huggingface.co/datasets/Jemary/FedBN_Dataset/blob/main/digit_dataset.zip).
The upstream README currently swaps the dataset/model link filenames; this
command deliberately uses `digit_dataset.zip`, not `digit_model.zip`.
SHA256: `6c006e41ce16404aab520895a5c510166453c8b58ed2e7eb7e23e133e1fa4221`.
Only the verified archive is deserialized; its trusted upstream pickle payloads
are converted to `.npz` files loaded with `allow_pickle=False` during experiments.

To update an existing Linux project, replace the `icde/` code and
`Run_Real_digits5.ipynb` from the new package, retaining `data/` and `outputs/`.
Restart the notebook kernel and run its cells from the beginning. The reported
failure happened before training. Checkpoints from older code versions are
intentionally not accepted by source-hash validation.

Verified release sizes: SynthDigits 7,438 training / 97,791 test; MNIST-M 7,438 training / 14,000 test. Record these preprocessed-release sizes in the experiment protocol; they are not the full original source datasets.
