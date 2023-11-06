from pathlib import Path
from typing import Tuple, Union

import mujoco as mj
from mujoco import mjx

from ambersim import ROOT


def load_mjx_model_from_file(filepath: Union[str, Path]) -> Tuple[mjx.Model, mjx.Data]:
    """Loads a mjx model/data pair from a filepath.

    Args:
            filepath: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.

    Returns:
            mjx_model: A mjx Model.
            mjx_data: A mjx Data struct.
    """
    assert isinstance(filepath, (str, Path))

    # checking whether file exists
    if isinstance(filepath, str):
        filepath = Path(filepath)  # global/local
    if not filepath.exists():
        filepath = ROOT / filepath  # repo root
        if not filepath.exists():
            raise ValueError("The model file doesn't exist at the specified path!")
    filepath = str(filepath)

    # loading the model and data
    _model = mj.MjModel.from_xml_path(filepath)
    mjx_model = mjx.device_put(_model)
    mjx_data = mjx.make_data(_model)
    return mjx_model, mjx_data
