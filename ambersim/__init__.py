import os
import subprocess
from pathlib import Path

import mujoco  # if we don't import mujoco before certain functions in ambersim, we may get segfaults

# package root
ROOT = str(Path(__file__).parent.absolute())

# configure MuJoCo to use the EGL rendering backend (requires GPU)
try:
    # checks whether an nvidia GPU is available
    subprocess.check_output("nvidia-smi")

    # Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
    # This is usually installed as part of an Nvidia driver package, but the Colab
    # kernel doesn't install its driver via APT, and as a result the ICD is missing.
    # (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write(
                """{
                "file_format_version" : "1.0.0",
                "ICD" : {
                    "library_path" : "libEGL_nvidia.so.0"
                }
            }
            """
            )

    # set a mujoco environment variable
    os.environ["MUJOCO_GL"] = "egl"

except Exception:
    pass
