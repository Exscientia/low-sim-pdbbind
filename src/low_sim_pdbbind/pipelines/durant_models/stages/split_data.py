import os

import logging

import dvc.api
from datasets import disable_caching

import molflux.datasets
from low_sim_pdbbind.utils.dir import get_pipeline_dir

disable_caching()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = dvc.api.params_show()

    log.info("Loading dataset")

    dataset_dir = get_pipeline_dir() / "data" / "dataset.parquet"
    dataset = molflux.datasets.load_dataset_from_store(dataset_dir)

    data_dir = get_pipeline_dir() / "data" / "data_dump"

    df = dataset.to_pandas()

    df_train = df[df[cfg["higher_split"]["presets"]["column"]] == "train"]
    df_test = df[df[cfg["higher_split"]["presets"]["column"]] == "test"]

    df_train = df_train[[cfg["key"], cfg["dataset"]["y_features"][0], cfg["dataset"]["receptor_path_column"]]]
    df_train = df_train.rename(columns={
        cfg["key"]: "key",
        cfg["dataset"]["y_features"][0]: "pk",
        cfg["dataset"]["receptor_path_column"]: "protein",
    })
    df_train["ligand"] = str(data_dir / "ligands") + "/" + df_train["key"].astype(str) + ".sdf"
    df_train["protein"] = str(data_dir / "proteins") + "/" + df_train["key"].astype(str) + ".pdb"

    df_test = df_test[[cfg["key"], cfg["dataset"]["y_features"][0], cfg["dataset"]["receptor_path_column"]]]
    df_test = df_test.rename(columns={
        cfg["key"]: "key",
        cfg["dataset"]["y_features"][0]: "pk",
        cfg["dataset"]["receptor_path_column"]: "protein",
    })
    df_test["ligand"] = str(data_dir / "ligands") + "/" + df_test["key"].astype(str) + ".sdf"
    df_test["protein"] = str(data_dir / "proteins") + "/" + df_test["key"].astype(str) + ".pdb"

    df_train.to_csv(get_pipeline_dir() / "data" / "train.csv")
    df_test.to_csv(get_pipeline_dir() / "data" / "test.csv")

if __name__ == "__main__":
    main()
