import logging

import dvc.api
from datasets import disable_caching

import molflux.core
import molflux.modelzoo
from low_sim_pdbbind.utils.dir import get_pipeline_dir
from low_sim_pdbbind.utils import io_utils
from low_sim_pdbbind.utils.splits.by_column_split import ByColumnSplit
from low_sim_pdbbind.utils.splits.tanimoto_split import Tanimoto

disable_caching()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = dvc.api.params_show()

    log.info("Loading featurised dataset from disk")

    dataset_dir = get_pipeline_dir() / "data" / "dataset_processed"
    high_low_dict = io_utils.load_splits_datasets(dataset_dir)

    for split, data_split in high_low_dict.items():
        for fold, data in data_split.items():

            # specify config
            log.info("Loading model")
            model = molflux.modelzoo.load_from_dict(cfg["train"]["model_config"])

            # train
            log.info(f"Training model: {split}, {fold}")
            model.train(
                train_data=data["train"],
                **cfg["train"]["model_config"].get("presets", {})
            )

            log.info(f"Saving model: {split}, {fold}")
            model_path = get_pipeline_dir() / "models" / split / fold
            model_path.mkdir(exist_ok=True, parents=True)
            molflux.core.save_model(
                model,
                path=str(model_path),
                featurisation_metadata=cfg["featurisation"]["featurisation_metadata"],
            )

if __name__ == "__main__":
    main()
