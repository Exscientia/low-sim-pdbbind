import warnings
from typing import Union

from loguru import logger


def get_stamped_version() -> Union[str, None]:
    try:
        from ._version import version as stamped_version

        return stamped_version
    except ImportError:  # pragma: no cover
        warnings.warn(
            f"could not determine {__name__} package version - have you run `pip"
            " install -e .` ?"
        )
        return None


__version__ = get_stamped_version()

# logging configuration should be left to the users of your library
logger.disable(__name__)
