import logging
import json
import dvc.api
from datasets import disable_caching

import molflux.metrics
from low_sim_pdbbind.utils.dir import get_pipeline_dir
from low_sim_pdbbind.utils.metrics.metric_computation import inference_and_score
from low_sim_pdbbind.utils import io_utils

disable_caching()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = dvc.api.params_show()

    logger.info("Loading processed dataset from disk")
    dataset_dir = get_pipeline_dir() / "data" / "dataset_processed"
    high_low_dict = io_utils.load_splits_datasets(dataset_dir)

    metric_suite = molflux.metrics.load_suite(cfg["metrics"])

    for split, data_split in high_low_dict.items():
        fold_scores = {}
        for fold, data in data_split.items():
            logger.info(f"Computing metrics for: {split}, {fold}")
            fold_scores[fold] = inference_and_score(
                model_dir=str(get_pipeline_dir() / "models" / split / fold),
                ds_dict=data,
                metric_suite=metric_suite
            )

        score_output_dir = get_pipeline_dir() / "metrics" / split
        score_output_dir.mkdir(exist_ok=True, parents=True)
        total_score_file = score_output_dir / "scores.json"

        with open(total_score_file, "w") as f:
            json.dump(fold_scores, f)

if __name__ == "__main__":
    main()
