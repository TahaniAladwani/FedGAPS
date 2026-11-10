"""Install missing domains from the checksum-pinned FedBN Digits release."""
import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

URL = 'https://huggingface.co/datasets/Jemary/FedBN_Dataset/resolve/main/digit_dataset.zip'
SHA256 = '6c006e41ce16404aab520895a5c510166453c8b58ed2e7eb7e23e133e1fa4221'


def unpack_pair(stream):
    # Only called on members of the checksum-verified upstream archive.
    images, labels = np.load(stream, allow_pickle=True)
    images, labels = np.asarray(images), np.asarray(labels).reshape(-1)
    if len(images) != len(labels) or not len(labels):
        raise ValueError('Invalid image/label lengths')
    if not np.all(np.isfinite(labels)) or not np.all(labels == labels.astype(np.int64)):
        raise ValueError('Non-integer labels')
    labels = labels.astype(np.int64)
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError('Labels must be digits 0..9')
    if images.dtype != np.uint8 or images.ndim not in (3, 4):
        raise ValueError('Expected uint8 image array')
    return images, labels



def available(root, domain):
    aliases = ('synth', 'SynthDigits') if domain == 'synth' else ('mnistm', 'MNISTM', 'MNIST_M')
    return all(any(
        (root / alias / split).is_dir() or
        any((root / alias / f'{split}.{suffix}').is_file() for suffix in ('pt', 'pkl', 'pickle')) or
        (root / domain / f'{split}.npz').is_file()
        for alias in aliases) for split in ('train', 'test'))

def install(archive, root):
    archive, root = Path(archive), Path(root)
    with archive.open('rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else digest(source)
    if actual != SHA256:
        raise ValueError('FedBN archive SHA256 mismatch; remove the cached archive and retry')
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for domain, aliases in [('synth', {'synth','synthdigits'}), ('mnistm', {'mnistm'})]:
            if available(root, domain):
                continue
            matches = [n for n in names if n.endswith('/test.pkl') and Path(n).parent.name.lower().replace('_','').replace('-','') in aliases]
            if len(matches) != 1:
                raise ValueError(f'Expected one {domain}/test.pkl in archive; found {matches}')
            prefix = matches[0][:-len('test.pkl')]
            for split in ('train', 'test'):
                destination = root/domain/f'{split}.npz'
                if destination.exists():
                    continue
                members = [prefix+f'partitions/train_part{i}.pkl' for i in range(10)] if split == 'train' else matches
                if any(n not in names for n in members):
                    raise ValueError(f'Incomplete upstream {domain} {split} split')
                pairs = []
                for member in members:
                    with z.open(member) as stream:
                        pairs.append(unpack_pair(stream))
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.tmp.npz')
                np.savez(temporary, images=np.concatenate([p[0] for p in pairs]), labels=np.concatenate([p[1] for p in pairs]))
                temporary.replace(destination)
                print(f'Prepared {destination}', flush=True)


def digest(source):
    h = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024*1024), b''):
        h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/digits5'))
    parser.add_argument('--archive', type=Path, help='Already downloaded official digit_dataset.zip')
    args = parser.parse_args()
    if all(available(args.root, d) for d in ('synth', 'mnistm')):
        print('SynthDigits and MNIST-M are already prepared.')
        return
    archive = args.archive or args.root/'downloads/digit_dataset.zip'
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.part')
        print(f'Downloading FedBN dataset (~278 MB): {URL}', flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as target:
            shutil.copyfileobj(response, target, 1024*1024)
        temporary.replace(archive)
    install(archive, args.root)


if __name__ == '__main__':
    main()
