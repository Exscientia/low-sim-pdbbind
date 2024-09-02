"""Tanimoto similarity splitting strategy.
"""

import logging
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

import molflux.features
from tqdm.auto import tqdm
from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable
from molflux.splits.catalogue import register_splitting_strategy

logger = logging.getLogger(__name__)

_DESCRIPTION = """
"""

@register_splitting_strategy("core", "tanimoto")
class Tanimoto(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION,
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        n_splits: int = 1,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        batch_size: int = 1000,
        random_int: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        """

        if y is None:
            raise ValueError("""y parameter should be provided for tanimoto splits.""")

        np.testing.assert_almost_equal(
            train_fraction + validation_fraction + test_fraction, 1.0
        )

        ecfp_rep = molflux.features.load_representation("circular")
        fingerprints = ecfp_rep.featurise(y)["circular"]
        sim_mat = compute_similarity_mat(np.array(fingerprints), batch_size=batch_size)

        for idx in range(n_splits):
            # Split into two groups: training set and everything else.

            train_size = int(train_fraction * len(dataset))
            validation_size = int(validation_fraction * len(dataset))
            test_size = len(dataset) - train_size - validation_size

            train_indices, test_validation_indices = _split_fingerprints(
                sim_mat, train_size, validation_size + test_size, random_int
            )

            # Split the second group into validation and test sets.

            if validation_size == 0:
                validation_indices = []
                test_indices = test_validation_indices
            elif test_size == 0:
                test_indices = []
                validation_indices = test_validation_indices
            else:
                test_valid_fps = [fingerprints[i] for i in test_validation_indices]
                sim_mat = compute_similarity_mat(np.array(test_valid_fps), batch_size=batch_size)
                test_indices, validation_indices = _split_fingerprints(
                    sim_mat, test_size, validation_size, random_int
                )
                test_indices = [test_validation_indices[i] for i in test_indices]
                validation_indices = [
                    test_validation_indices[i] for i in validation_indices
                ]
            yield train_indices, validation_indices, test_indices

def batch_sim(fp1, fps, batch_size=100):
    sum_fp1 = fp1.sum()
    fp1 = np.expand_dims(fp1, 0)
    sims = []
    for i in range(1 + len(fps) // batch_size):
        fp_chunk = fps[i * batch_size:(i + 1) * batch_size]

        intersection = (fp1 * fp_chunk).sum(-1)

        sims.append(intersection / (sum_fp1 + fp_chunk.sum(-1) - intersection))
    return np.hstack(sims)

def compute_similarity_mat(fps, batch_size=1000):
    sim_mat = np.zeros((len(fps), len(fps)))

    for i in tqdm(range(len(fps))):
        sims_i = batch_sim(fps[i], fps[i + 1:], batch_size=batch_size)
        sims_i = np.hstack([np.array([0.0] * i + [0.5]), sims_i])
        sim_mat[i] = sims_i

    sim_mat = (sim_mat.T + sim_mat)
    return sim_mat

def _split_fingerprints(
    sim_mat: np.ndarray, size0: int, size1: int, random_int
) -> Tuple[List[int], List[int]]:
    """Divides a list of fingerprints into two groups."""

    # Begin by assigning the first molecule to the first group.
    if random_int is not None:
        random_start_idx = random_int
    else:
        random_start_idx = np.random.randint(sim_mat.shape[0])

    indices_in_group = [[random_start_idx], []]
    remaining_indices = list(range(0, sim_mat.shape[0]))

    # remove row from sim mat
    sim_mat = np.delete(sim_mat, random_start_idx, axis=0)
    # remove index from indices
    del remaining_indices[random_start_idx]

    pbar = tqdm(total=sim_mat.shape[0])
    while len(sim_mat) > 0:

        # Decide which group to assign the next molecule to.
        # goes in smaller group
        group = 0 if len(indices_in_group[0]) / size0 <= len(indices_in_group[1]) / size1 else 1

        # Identify the unassigned molecule that is least similar to everything in
        # the other group.
        indices_of_other_group = indices_in_group[1 - group]
        max_sims_to_other_group = sim_mat[:, indices_of_other_group].max(-1)
        i = np.argmin(max_sims_to_other_group)

        # Add it to the group.
        indices_in_group[group].append(remaining_indices[i])

        # Update the sim mat
        sim_mat = np.delete(sim_mat, i, axis=0)
        # update remaining indices
        remaining_indices = np.delete(remaining_indices, i, axis=0)

        pbar.update(1)
    pbar.close()

    return indices_in_group
