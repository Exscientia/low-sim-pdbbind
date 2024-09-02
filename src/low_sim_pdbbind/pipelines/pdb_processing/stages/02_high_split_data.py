"""
Script to apply a high-split
strategy (train split is used subsequently as a dev dataset).
"""

import logging
import dvc.api
from datasets import disable_caching
import numpy as np
import pandas as pd
import molflux.datasets
import molflux.splits
import molflux.features
from low_sim_pdbbind.utils.dir import get_pipeline_dir

from low_sim_pdbbind.utils.splits.by_column_split import ByColumnSplit
from low_sim_pdbbind.utils.splits.tanimoto_split import Tanimoto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

disable_caching()

def main() -> None:
    cfg = dvc.api.params_show()

    dataset_dir = get_pipeline_dir() / "data" / "dataset_processed.parquet"
    dataset = molflux.datasets.load_dataset_from_store(str(dataset_dir), format="parquet")

    # select the test data
    test_dataset = dataset.filter(lambda x: x["uniprot_id"] in cfg["uniprot_ids_benchmarking"])
    train_dataset = dataset.filter(lambda x: x["uniprot_id"] not in cfg["uniprot_ids_benchmarking"])

    # filter any point in train with > 0.5 sim to test ligands

    # compute fps
    ecfp_rep = molflux.features.load_representation("circular")
    test_ecfps = np.array(ecfp_rep.featurise(test_dataset["canonical_smiles"])["circular"])
    mod_test_ecfps = test_ecfps.sum(-1)
    def filter_by_ligand_sim(x):
        x_ecfp = np.array(ecfp_rep.featurise(x["canonical_smiles"])["circular"])
        intersection = (x_ecfp * test_ecfps).sum(-1)
        mod_B = x_ecfp.sum()

        sim = float(max(intersection / (mod_B + mod_test_ecfps - intersection)))
        return sim < cfg["SIM_THRESH"]

    train_dataset = train_dataset.filter(filter_by_ligand_sim, num_proc=10)

    # filter any point in train with > SIM_THRESH sim to test proteins
    # similarity computed using foldseek
    logger.info("loading sim dataframe")
    df = pd.read_csv(
        cfg["path_to_foldseek_aln"],
        delimiter="\t",
        names="query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,lddt".split(",")
    )
    logger.info("adding pdb codes")
    df["query_pdb"] = df["query"].str.slice(0, 4)
    df["target_pdb"] = df["target"].str.slice(0, 4)

    # pdb contains multiple chains, pick chain with max similarity
    logger.info("pick max similarity")
    df = df[["query_pdb", "target_pdb", "lddt", "fident"]]
    df = df.groupby(["query_pdb", "target_pdb"]).max().reset_index()

    # pick pdbs with similarity < SIM_THRESH
    logger.info("get low sim pdbs and filter")
    low_sim_pdbs = set(df[(df["lddt"] < cfg["SIM_THRESH"]) & (df["fident"] < cfg["SIM_THRESH"])]["target_pdb"])
    train_dataset = train_dataset.filter(lambda x: x["pdb_code"] in low_sim_pdbs)

    # add lp splits with 5%, 10%, 30%, 50%, 80% scaffold split
    split_config = {
        "name": "tanimoto",
        "presets": {
            "validation_fraction": 0.0,
            "n_splits": 3,
        }
    }
    def add_split(x, split_name, train_pdb_codes, test_pdb_codes):
        if x["pdb_code"] in train_pdb_codes:
            x[split_name] = "train"
        elif x["pdb_code"] in test_pdb_codes:
            x[split_name] = "test"
        else:
            x[split_name] = None
        return x

    train_pdb_codes = train_dataset["pdb_code"]
    test_pdb_codes = test_dataset["pdb_code"]

    # split 0 (add 3 identical folds with all data in test
    for i in range(3):
        dataset = dataset.map(
            lambda x: add_split(
                x, f"by_bespoke_0_fold_{i}", train_pdb_codes, test_pdb_codes
            )
        )

    split_idxs = {}
    for train_frac in [5, 30, 80]:
        split_idxs[train_frac] = {
            "train": {},
            "test": {},
        }

        split_config["presets"]["train_fraction"] = train_frac / 100
        split_config["presets"]["test_fraction"] = 1 - (train_frac / 100)

        splitting_strat = molflux.splits.load_from_dict(split_config)

        # for each uniprot, split dataset, collect in split_idxs, (combining over split folds)
        for uniprot in cfg["uniprot_ids_benchmarking"]:
            uniprot_test_dataset = test_dataset.filter(lambda x: x["uniprot_id"] == uniprot)
            for idx, data_split in enumerate(molflux.datasets.split_dataset(
                uniprot_test_dataset, splitting_strat, target_column="canonical_smiles"
            )):
                if idx in split_idxs[train_frac]["train"]:
                    split_idxs[train_frac]["train"][idx] += data_split["train"]["pdb_code"]
                    split_idxs[train_frac]["test"][idx] += data_split["test"]["pdb_code"]
                else:
                    split_idxs[train_frac]["train"][idx] = train_pdb_codes + data_split["train"]["pdb_code"]
                    split_idxs[train_frac]["test"][idx] = data_split["test"]["pdb_code"]

    for train_frac, split_dict in split_idxs.items():
        for idx in split_dict["train"]:
            dataset = dataset.map(
                lambda x: add_split(
                    x, f"by_bespoke_{train_frac}_fold_{idx}", split_dict["train"][idx], split_dict["test"][idx]
                )
            )

    dataset_dir = get_pipeline_dir() / "data" / "dataset_high_split.parquet"
    molflux.datasets.save_dataset_to_store(dataset, str(dataset_dir), format="parquet")


if __name__ == "__main__":
    main()
