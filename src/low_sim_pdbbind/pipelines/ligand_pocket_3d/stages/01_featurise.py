import os

import logging

import dvc.api
from datasets import disable_caching

import molflux.datasets
from low_sim_pdbbind.utils.dir import get_pipeline_dir
from low_sim_pdbbind.utils import io_utils

disable_caching()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = dvc.api.params_show()

    dataset_dir = get_pipeline_dir() / "data" / "dataset_filtered.parquet"
    dataset = molflux.datasets.load_dataset_from_store(str(dataset_dir), format="parquet")

    dataset_w_feats_dir = get_pipeline_dir() / "data" / "dataset_with_features.parquet"
    dataset_with_features = molflux.datasets.load_dataset_from_store(str(dataset_w_feats_dir), format="parquet")

    pdb_key_to_idx = {dataset_with_features[idx]["pdb_code"]: idx for idx in range(len(dataset_with_features))}
    new_features = list(set(dataset_with_features.column_names) - set(dataset.column_names))

    def add_features(x, features):
        point = dataset_with_features[pdb_key_to_idx[x["pdb_code"]]]
        for feat in features:
            x[feat] = point[feat]
        return x


    log.info("Loading split dataset from disk")

    input_dir = str(get_pipeline_dir() / "data/dataset_split")
    output_dir = str(get_pipeline_dir() / "data/dataset_processed")

    # Load datasets from high and low split directories
    high_low_dict = io_utils.load_splits_datasets(input_dir)

    # Featurise high and low splits.
    for split, split_data in high_low_dict.items():
        for fold, data in split_data.items():
            data = data.map(lambda x: add_features(x, new_features))
            high_low_dict[split][fold] = data

    # Save featurised datasets
    io_utils.write_splits_datasets(high_low_dict, output_dir)


if __name__ == "__main__":
    main()
