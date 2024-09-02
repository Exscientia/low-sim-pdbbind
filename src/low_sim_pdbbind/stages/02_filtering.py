import logging

import dvc.api
import molflux.datasets
from datasets import disable_caching
from low_sim_pdbbind.utils.dir import get_pipeline_dir

disable_caching()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = dvc.api.params_show()

    log.info("Loading dataset from disk")
    dataset_path = get_pipeline_dir() / "data" / "dataset.parquet"
    dataset = molflux.datasets.load_dataset_from_store(dataset_path, format="parquet")

    ori_count = len(dataset)
    for filter in cfg["filtering"]:
        dataset = dataset.filter(lambda x: x[filter["column_name"]] == filter["value"])
    log.info(f"Original row count {ori_count} filtered down to {len(dataset)}")

    for none_filter in cfg["filtering_nones"]:
        dataset = dataset.filter(lambda x: x[none_filter["column_name"]] is not None)
    log.info(f"after removing Nones: {len(dataset)}")

    assert len(dataset) > 0, f"filter removed all points in dataset"

    dataset_split_dir = (
        get_pipeline_dir() / "data" / "dataset_filtered.parquet"
    )
    molflux.datasets.save_dataset_to_store(dataset, str(dataset_split_dir), format="parquet")


if __name__ == "__main__":
    main()
