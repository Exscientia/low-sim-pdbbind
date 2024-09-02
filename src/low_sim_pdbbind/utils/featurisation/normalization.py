from typing import List, Tuple

import numpy as np
from datasets import DatasetDict
from sklearn.preprocessing import StandardScaler


def _norm(x, col, scaler):
    X = np.array(x[col]).reshape(1, -1)  # Single sample
    X_norm = np.squeeze(scaler.transform(X))
    x[f"{col}::normalized"] = X_norm
    return x


def normalize_dataset_dict(
    ds_dict: DatasetDict, features: List[str]
) -> Tuple[DatasetDict, List[str]]:
    """
    Normalize dataset columns based on given features. Standardize features by removing the mean and
    scaling to unit variance. Computes the mean and std of the train-dataset
    and applies it to all datasets in the dataset-dict.

    Args:
        ds_dict (DatasetDict): Huggingface Dataset Dictionary containing datasets.
        features (List[str]): List of features to normalize.

    Returns:
        Tuple[DatasetDict, List[str]]: Tuple of Normalized Dataset Dictionary and list of new feature names.
    """

    new_features = []
    for feature in features:
        scaler = StandardScaler()
        X = np.array(ds_dict["train"][feature])
        # Reshape to 2D array if X is 1D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        scaler.fit(X)

        for split, ds in ds_dict.items():
            if len(ds) == 0:
                # Necessary for downstream-functionality (requires consistent features across
                # datasets, even if they are empty).
                ds_dict[split] = ds.add_column(f"{feature}::normalized", [])
            else:
                ds_dict[split] = ds.map(lambda x: _norm(x, feature, scaler))
        new_features.append(f"{feature}::normalized")

    return ds_dict, new_features
