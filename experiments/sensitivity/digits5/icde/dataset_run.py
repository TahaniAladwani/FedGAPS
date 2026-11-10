"""Dataset-specific launchers; never expand into another dataset's runs."""
import argparse
import json
from pathlib import Path
from .cli import main


def main_for_dataset(dataset, argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs' / f'{dataset}.json'))
    options, remaining = parser.parse_known_args(argv)
    matrix = json.loads(Path(options.config).read_text())
    if matrix.get('datasets') != [dataset]:
        raise ValueError(f'This launcher requires datasets=["{dataset}"]; it cannot run other datasets')
    main(['sweep', '--config', options.config, *remaining])
