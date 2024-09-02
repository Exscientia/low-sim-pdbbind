import logging

import molflux.modelzoo
from molflux.core.models import get_references, predict
from molflux.core.scoring import compute_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def score_model(
    model,
    fold,
    metrics,
    prediction_kwargs=None,
    scoring_kwargs=None,
):
    prediction_method = "predict"

    references = get_references(model=model, fold=fold)
    predictions = predict(
        model=model,
        fold=fold,
        prediction_method=prediction_method,
        prediction_kwargs=prediction_kwargs,
    )
    pred_keys = list(predictions.keys())
    for k in pred_keys:
        if len(predictions[k]) < 2:
            del predictions[k]
            del references[k]

    scores = compute_scores(
        predictions=predictions,
        references=references,
        metrics=metrics,
        scoring_kwargs=scoring_kwargs,
    )

    return (
        scores,
        predictions,
        references,
    )

def inference_and_score(
    model_dir, ds_dict, metric_suite,
):
    model = molflux.modelzoo.load_from_store(str(model_dir))
    fold_scores, predictions, references = score_model(model, fold=ds_dict, metrics=metric_suite)
    logger.info(fold_scores)

    for split, res_dic_all in fold_scores.items():
        for task, res_dic in res_dic_all.items():
            # convert lists to strings: dvc exp show does not parse key: value pairs with
            # list values, only numbers/strings
            res_dic["references"] = str(references[split][task])
            res_dic["predictions"] = str(predictions[split][f"{model.tag}::{task}"])
            if "unique_index" in ds_dict[split].column_names:
                res_dic["unique_index"] = str(ds_dict[split]["unique_index"])

            if "display_name" in ds_dict[split].column_names:
                res_dic["display_name"] = str(ds_dict[split]["display_name"])

            if "uniprot_id" in ds_dict[split].column_names:
                res_dic["uniprot_id"] = str(ds_dict[split]["uniprot_id"])

    return fold_scores
