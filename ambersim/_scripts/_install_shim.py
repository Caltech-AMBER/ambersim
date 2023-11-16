import argparse
import subprocess

from ambersim import ROOT


def entrypoint():
    """Installs mujoco from source with the installation script."""
    # parses script args
    parser = argparse.ArgumentParser(description="Options for the installation of mujoco from source.")
    parser.add_argument("--hash", dest="h", type=str, default="", help="An optional commit hash to pull from source.")
    parser.add_argument(
        "--mujoco-dir",
        dest="mujoco_dir",
        default="",
        help="An optional path to a local mujoco repo. If not supplied, defaults to the directory containing ambersim.",
    )
    args = parser.parse_args()

    # executing the installation of mujoco from source
    subprocess.call([f"{ROOT}/_scripts/install_mj_source.sh", "-h", args.h, "--mujoco-dir", args.mujoco_dir])


if __name__ == "__main__":
    entrypoint()
