#!/bin/bash
# instructions are taken from two places:
# (1) https://mujoco.readthedocs.io/en/latest/programming/#building-from-source
# (2) https://mujoco.readthedocs.io/en/latest/python.html#building-from-source

set -e

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
MUJOCO_PATH="$mujoco_dir/mujoco_install" MUJOCO_PLUGIN_PATH="$mujoco_dir/plugin" pip install "$tar_path"
