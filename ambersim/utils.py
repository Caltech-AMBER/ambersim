from pathlib import Path
from typing import Tuple, Union

import mujoco as mj
from mujoco import mjx

from ambersim import ROOT

# ############## #
# INTERNAL UTILS #
# ############## #


def _filepath_check_util(filepath: Union[str, Path]) -> None:
    """Checks validity of a filepath for model loading."""
    assert isinstance(filepath, (str, Path))

    # checking whether file exists
    if isinstance(filepath, str):
        filepath = Path(filepath)  # global/local
    if not filepath.exists():
        filepath = ROOT / filepath  # repo root
        if not filepath.exists():
            raise ValueError("The model file doesn't exist at the specified path!")


# ############ #
# PUBLIC UTILS #
# ############ #


def load_mjx_model_from_file(filepath: Union[str, Path]) -> Tuple[mjx.Model, mjx.Data]:
    """Loads a mjx model/data pair from a filepath.

    Args:
        filepath: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.

    Returns:
        mjx_model: A mjx Model.
        mjx_data: A mjx Data struct.
    """
    _filepath_check_util(filepath)
    filepath = str(filepath)

    # loading the model and data
    _model = mj.MjModel.from_xml_path(filepath)
    mjx_model = mjx.device_put(_model)
    mjx_data = mjx.make_data(_model)
    return mjx_model, mjx_data


def save_model_xml(filepath: Union[str, Path]) -> None:
    """Loads a model and saves it to a mujoco-compliant XML.

    Will save the file to the directory where this util is called.

    Args:
        filepath: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.
    """
    # loading model and saving XML
    _filepath_check_util(filepath)
    filepath = str(filepath)
    _model = mj.MjModel.from_xml_path(filepath)
    output_name = filepath.split("/")[-1].split(".")[0]
    mj.mj_saveLastXML(f"{output_name}.xml", _model)

    # reporting save path for clarity
    output_path = Path.cwd() / Path(str(output_name) + ".xml")
    print(f"XML file saved to {output_path}!")
