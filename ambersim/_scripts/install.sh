#!/bin/bash
# instructions for installing from source are taken from two places:
# (1) https://mujoco.readthedocs.io/en/latest/programming/#building-from-source
# (2) https://mujoco.readthedocs.io/en/latest/python.html#building-from-source

set -e

# Define usage function
usage() {
    echo "Usage: $0 [-d] [-s] [-h <mujoco_hash>] [--disable-apt] [--mujoco_dir </path/ending/in/mujoco>]"
    exit 1
}

# Parse options and check for getopt errors
OPTIONS=$(getopt -o 'dsh:' --long disable-apt,mujoco-dir: -- "$@")
if [ $? -ne 0 ]; then
  usage
fi

eval set -- "$OPTIONS"

# defaults
dev=false
source=false
apt_dependencies=true

# process inputs
while true; do
  case "$1" in
    -d)
      dev=true
      shift
      ;;
    -s)
      source=true
      shift
      ;;
    -h)
      if [[ -n "$2" ]] && [[ "$2" != "" ]] && [[ "${2:0:1}" != "-" ]]; then
        hash="$2"
        shift 2
      elif [[ "$2" == "" ]]; then
        hash=""
        shift 2
      else
        hash=""
        shift
      fi
      ;;
    --disable-apt)
      apt_dependencies=false
      shift
      ;;
    --mujoco-dir)
      if [[ -n "$2" ]] && [[ "$2" != "" ]] && [[ "${2:0:1}" != "-" ]]; then
        mujoco_dir="$2"
        shift 2
      elif [[ "$2" == "" ]]; then
        mujoco_dir=""
        shift 2
      else
        mujoco_dir=""
        shift
      fi
      ;;
    --mujoco-dir=*)
      mujoco_dir="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      ;;
  esac
done

# check if mujoco_dir ends with "mujoco"
if [[ -n "$mujoco_dir" ]] && ! [[ "$mujoco_dir" == */mujoco ]]; then
    echo "The mujoco directory must end in 'mujoco'!"
    exit 1
fi

# checking whether to install apt dependencies
if [ $apt_dependencies = true ] ; then
    echo "[NOTE] Installing apt dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        libgl1-mesa-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxrandr-dev \
        libxi-dev \
        ninja-build
fi

# Install regular or development dependencies
if [ "$dev" = true ] ; then
    echo "[NOTE] Installing development dependencies..."
    pip install -e .[all] --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --find-links https://download.pytorch.org/whl/cu118
    pre-commit autoupdate
    pre-commit install
else
    echo "[NOTE] Installing non-developer dependencies..."
    pip install -e . --default-timeout=100 future --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --find-links https://download.pytorch.org/whl/cu118
fi

# Checking whether to install mujoco from source
if [ "$source" = true ] ; then
    script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    bash "$script_dir/install_mj_source.sh" -h "$hash" --mujoco-dir "$mujoco_dir"
fi
