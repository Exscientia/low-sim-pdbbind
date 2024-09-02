import os
from pathlib import Path

def _find_parent_dir_containing(path: str | os.PathLike | None, subdir: str) -> Path:
    """Finds the deepest dir in the hierarchy of `path` that also contains directory `subdir`."""
    if path is None:
        path = os.getcwd()

    current_path = Path(path).resolve()
    while current_path != current_path.parent:
        if (current_path / subdir).is_dir():
            return current_path
        current_path = current_path.parent

    raise ValueError(f"No directory containing '{subdir}' found in the hierarchy.")

def get_pipeline_dir(path: str | os.PathLike | None = None) -> Path:
    """Returns the path to the DVC pipeline directory containing `path`."""
    return _find_parent_dir_containing(path, ".dvc")
