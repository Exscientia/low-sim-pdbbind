from typing import Any, Iterator, Optional, List

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable
from molflux.splits.catalogue import register_splitting_strategy

_DESCRIPTION = """
"""

@register_splitting_strategy("custom", "by_column")
class ByColumnSplit(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION,
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        *,
        columns: Optional[List] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:

        for col in columns:
            train_indices = []
            validation_indices = []
            test_indices = []
            for idx, label in enumerate(dataset[col]):
                if label == "train":
                    train_indices.append(idx)
                elif label == "validation":
                    validation_indices.append(idx)
                elif label == "test":
                    test_indices.append(idx)
            yield train_indices, validation_indices, test_indices
