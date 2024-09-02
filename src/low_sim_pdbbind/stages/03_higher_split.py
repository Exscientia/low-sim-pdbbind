import logging

import dvc.api
from datasets import disable_caching

import molflux.datasets
import molflux.splits
from low_sim_pdbbind.utils.dir import get_pipeline_dir
from low_sim_pdbbind.utils.splits.by_column_split import ByColumnSplit
from low_sim_pdbbind.utils.splits.tanimoto_split import Tanimoto

disable_caching()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = dvc.api.params_show()

    log.info("Loading dataset from disk")

    dataset_dir = get_pipeline_dir() / "data" / "dataset_filtered.parquet"
    dataset = molflux.datasets.load_dataset_from_store(str(dataset_dir), format="parquet")

    # specify config
    log.info("Splitting dataset")
    split_config = {
        "name": cfg["higher_split"]["name"],
        "presets": cfg["higher_split"]["presets"],
    }

    # load splitting strategy
    splitting_strategy = molflux.splits.load_from_dict(split_config)

    # split the dataset
    dataset_split = molflux.datasets.split_dataset(
        dataset,
        splitting_strategy,
        groups_column=cfg["higher_split"]["groups_column"],
        target_column=cfg["higher_split"]["target_column"],
    )

    for idx, data_fold in enumerate(dataset_split):

        # remove empty keys or None key
        data_fold_keys = list(data_fold.keys())
        for k in data_fold_keys:
            if (k is None) or (len(data_fold[k]) == 0):
                del data_fold[k]

        log.info("Saving split dataset")
        dataset_split_dir = get_pipeline_dir() / "data" / "dataset_split" / "higher_split" / f"fold_{idx:02}"
        dataset_split_dir.mkdir(exist_ok=True, parents=True)
        molflux.datasets.save_dataset_to_store(data_fold, str(dataset_split_dir), format="parquet")


if __name__ == "__main__":
    main()
