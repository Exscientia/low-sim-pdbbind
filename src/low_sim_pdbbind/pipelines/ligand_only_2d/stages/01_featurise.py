import os

import logging

import dvc.api
from datasets import disable_caching

from low_sim_pdbbind.utils.dir import get_pipeline_dir
from molflux.core import featurise_dataset

from low_sim_pdbbind.utils import io_utils
from low_sim_pdbbind.utils.featurisation import normalization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
disable_caching()

def fill_featurisation_metadata_config(featurisation_metadata, train_data):
    # fills featurisation metadata with train samples required for dataset level featurisation

    for config in featurisation_metadata["config"]:
        for sub_config in config["representations"]:
            if sub_config["config"].get("train_samples", False):
                sub_config["config"]["train_samples"] = train_data[config["column"]]
    return featurisation_metadata


def main() -> None:
    cfg = dvc.api.params_show()
    logger.info("Loading dataset from disk")

    input_dir = str(get_pipeline_dir() / "data/dataset_split")
    output_dir = str(get_pipeline_dir() / "data/dataset_processed")

    # Load datasets from high and low split directories
    high_low_dict = io_utils.load_splits_datasets(input_dir)

    # Featurise high and low splits.
    for split, split_data in high_low_dict.items():
        for fold, data in split_data.items():
            # fill in featurisation_metadata if needed (for dataset level features)
            featurisation_metadata = fill_featurisation_metadata_config(
                featurisation_metadata=cfg["featurisation"]["featurisation_metadata"],
                train_data=data["train"],
            )
            ds_feat = featurise_dataset(
                data,
                featurisation_metadata=featurisation_metadata,
                num_proc=cfg["featurisation"]["num_proc"],
                batch_size=cfg["featurisation"]["batch_size"],
            )
            high_low_dict[split][fold] = ds_feat

    # Normalize and save datasets
    for split, split_data in high_low_dict.items():
        for fold, data in split_data.items():
            if cfg["featurisation"]["standard_scale"]:
                logger.info("Scaling x_features")
                x_features = cfg["featurisation"]["normalize"]
                data, x_features_norm = normalization.normalize_dataset_dict(
                    data, features=x_features
                )
            high_low_dict[split][fold] = data

    # Save featurised datasets
    io_utils.write_splits_datasets(high_low_dict, output_dir)


if __name__ == "__main__":
    main()
