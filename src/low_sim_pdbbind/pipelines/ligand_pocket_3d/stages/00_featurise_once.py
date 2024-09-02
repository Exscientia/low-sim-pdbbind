import os

import logging

import dvc.api
from datasets import disable_caching

import molflux.datasets
from low_sim_pdbbind.utils.dir import get_pipeline_dir
from molflux.core import featurise_dataset
from process_structs import add_bytes, suppress_hydrogens

disable_caching()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = dvc.api.params_show()

    dataset_dir = get_pipeline_dir() / "data" / "dataset_filtered.parquet"
    dataset = molflux.datasets.load_dataset_from_store(str(dataset_dir), format="parquet")

    dataset = dataset.map(
        lambda x: add_bytes(
            x,
            cut_off=cfg["featurisation"]["cut_off"],
            ligand_mol_column=cfg["dataset"]["ligand_mol_column"],
            receptor_path_column=cfg["dataset"]["receptor_path_column"],
        ),
        num_proc=cfg["featurisation"]["num_proc"],
    )

    dataset = dataset.map(
        lambda x: suppress_hydrogens(x, which_hydrogens=cfg["featurisation"]["which_hydrogens"]),
        num_proc=cfg["featurisation"]["num_proc"],
    )
    dataset = featurise_dataset(
        dataset,
        featurisation_metadata=cfg["featurisation"]["featurisation_metadata"],
        num_proc=cfg["featurisation"]["num_proc"],
        batch_size=cfg["featurisation"]["batch_size"],
    )

    dataset_w_feats_dir = (
        get_pipeline_dir() / "data" / "dataset_with_features.parquet"
    )
    molflux.datasets.save_dataset_to_store(dataset, str(dataset_w_feats_dir), format="parquet")


if __name__ == "__main__":
    main()
