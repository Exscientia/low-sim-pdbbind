from pathlib import Path
from typing import Dict

from datasets import DatasetDict

import molflux.datasets


def load_splits_datasets(input_dir: str) -> Dict[str, Dict[str, DatasetDict]]:
    """
    Function to load all datasets it can find in high split and low split dir
    and puts them into nested dictionary
    """
    high_low_dict = {}

    for split in ['higher_split', 'lower_split']:
        split_dir = Path(input_dir) / split
        high_low_dict[split] = {}
        for path in split_dir.iterdir():
            fold = path.parts[-1]
            high_low_dict[split][fold] = molflux.datasets.load_dataset_from_store(str(path), format="parquet")

    return high_low_dict


def write_splits_datasets(high_low_dict: Dict[str, Dict[str, DatasetDict]], output_dir: str) -> None:
    """
    Function to write high and lows split data into a structured directory structure.
    """
    for split, split_data in high_low_dict.items():
        split_output_dir = Path(output_dir) / split
        split_output_dir.mkdir(exist_ok=True, parents=True)

        for fold, data in split_data.items():
            fold_output_dir = split_output_dir / fold
            fold_output_dir.mkdir(exist_ok=True, parents=True)
            molflux.datasets.save_dataset_to_store(data, str(fold_output_dir), format="parquet")
