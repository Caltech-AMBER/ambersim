from pathlib import Path
from typing import Union

from ambersim import ROOT


def _check_filepath(filepath: Union[str, Path]) -> str:
    """Checks validity of a filepath for model loading."""
    assert isinstance(filepath, (str, Path))

    # checking whether file exists
    if isinstance(filepath, str):
        filepath = Path(filepath)  # global/local
    if not filepath.exists():
        filepath = ROOT / filepath  # repo root
        if not filepath.exists():
            raise ValueError("The model file doesn't exist at the specified path!")
    filepath = str(filepath)
    return filepath


def _rmtree(f: Path):
    """Recursively deletes a directory using pathlib.

    See: https://stackoverflow.com/a/66552066
    """
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            _rmtree(child)
        f.rmdir()
