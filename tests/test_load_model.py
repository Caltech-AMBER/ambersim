from pathlib import Path

import mujoco as mj
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils import load_mjx_model_from_file


def _rmtree(f: Path):
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            _rmtree(child)
        f.rmdir()


def test_load_model():
    """Tests model loading."""
    global_path = ROOT + "/models/pendulum/pendulum.urdf"
    local_path = "_test_local/pendulum.urdf"
    repo_path = "models/pendulum/pendulum.urdf"

    local_dir = Path("_test_local")
    local_dir.mkdir(parents=True, exist_ok=False)
    with Path(local_path).open("w", encoding="utf-8") as f:
        f.write(Path(global_path).read_text())

    # string paths
    assert load_mjx_model_from_file(global_path)
    assert load_mjx_model_from_file(local_path)
    assert load_mjx_model_from_file(repo_path)

    # Path paths
    assert load_mjx_model_from_file(Path(global_path))
    assert load_mjx_model_from_file(Path(local_path))
    assert load_mjx_model_from_file(Path(repo_path))
    _rmtree(local_dir)
