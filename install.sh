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
    bash install_mj_source.sh
fi
