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

    dataset = molflux.datasets.load_dataset_from_store(cfg["dataset"]["path"])

    log.info("saving fetched dataset")
    dataset_dir = get_pipeline_dir() / "data" / "dataset.parquet"
    molflux.datasets.save_dataset_to_store(dataset, str(dataset_dir), format="parquet")


if __name__ == "__main__":
    main()
