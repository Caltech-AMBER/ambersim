#!/bin/bash
# instructions for installing from source are taken from two places:
# (1) https://mujoco.readthedocs.io/en/latest/programming/#building-from-source
# (2) https://mujoco.readthedocs.io/en/latest/python.html#building-from-source

set -e

dev=false
source=false
while getopts ds flag
do
    case "${flag}" in
        d) dev=true;;
        s) source=true;;
    esac
done

# Install regular or development dependencies
if [ "$dev" = true ] ; then
    echo "[NOTE] Installing development dependencies..."
    pip install --upgrade --upgrade-strategy eager -e .[all] --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --find-links https://download.pytorch.org/whl/cu118
    pre-commit autoupdate
    pre-commit install
else
    echo "[NOTE] Installing non-developer dependencies..."
    pip install --upgrade --upgrade-strategy -e . --default-timeout=100 future --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --find-links https://download.pytorch.org/whl/cu118
fi

# Checking whether to install mujoco from source
if [[ "$source" = true ]] ; then
    echo -e "\n[NOTE] Installing mujoco from source..."

    # Check if CMake and a compiler are installed
    if ! command -v cmake &> /dev/null; then
        echo "CMake is not installed. Please install CMake before proceeding."
        exit 1
    fi

    if ! command -v g++ &> /dev/null; then
        echo "C++ compiler is not available. Please install a C++ compiler before proceeding."
        exit 1
    fi

    # Clone the Mujoco repo to the directory above where this script is located
    # If it exists already, then just git pull new changes
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    mujoco_dir="$script_dir/../mujoco"
    if [ -d "$mujoco_dir" ]; then
        echo "Mujoco exists already - running git pull to update it!"
        (cd "$mujoco_dir" && git pull origin main)
    else
        git clone https://github.com/google-deepmind/mujoco.git "$mujoco_dir"
    fi

    # Configure and build
    export MAKEFLAGS="-j$(nproc)"  # allow the max number of processors when building
    cd "$mujoco_dir"
    cmake .
    cmake --build .

    # Install Mujoco
    cmake . -DCMAKE_INSTALL_PREFIX="./mujoco_install"
    cmake --install .

    # Generate source distribution required for Python bindings
    cd "$mujoco_dir/python"
    ./make_sdist.sh
    tar_path=$(find "$mujoco_dir/python/dist" -name 'mujoco-*.tar.gz' 2>/dev/null)

    # Installing mujoco and mjx from source
    MUJOCO_PATH="$mujoco_dir/mujoco_install" MUJOCO_PLUGIN_PATH="$mujoco_dir/plugin" pip install "$tar_path"
    cd "$mujoco_dir/mjx"
    pip install .
fi
