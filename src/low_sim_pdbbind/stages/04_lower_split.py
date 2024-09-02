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

    dataset_split_dir = get_pipeline_dir() / "data" / "dataset_split" / "higher_split"

    for data_fold_dir in dataset_split_dir.iterdir():
        higher_idx = int(data_fold_dir.parts[-1][-2:])

        dataset_split = molflux.datasets.load_dataset_from_store(
            str(data_fold_dir), format="parquet"
        )

        # specify config
        log.info("Splitting dataset")

        split_config = {
            "name": cfg["lower_split"]["name"],
            "presets": cfg["lower_split"]["presets"],
        }

        # load splitting strategy
        splitting_strategy = molflux.splits.load_from_dict(split_config)

        # split the dataset
        dataset_split_lower_folds = molflux.datasets.split_dataset(
            dataset_split["train"],
            splitting_strategy,
            groups_column=cfg["lower_split"]["groups_column"],
            target_column=cfg["lower_split"]["target_column"],
        )

        for lower_idx, dataset_split_lower in enumerate(dataset_split_lower_folds):
            log.info("Saving split dataset")
            dataset_split_lower_dir = (
                get_pipeline_dir()
                / "data"
                / "dataset_split"
                / "lower_split"
                / f"fold_{higher_idx:02}_{lower_idx:02}"
            )

            # remove empty keys
            data_fold_keys = list(dataset_split_lower.keys())
            for k in data_fold_keys:
                if len(dataset_split_lower[k]) == 0:
                    del dataset_split_lower[k]

            dataset_split_lower_dir.mkdir(exist_ok=True, parents=True)
            molflux.datasets.save_dataset_to_store(
                dataset_split_lower, str(dataset_split_lower_dir), format="parquet"
            )


if __name__ == "__main__":
    main()
