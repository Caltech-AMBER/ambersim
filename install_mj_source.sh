#!/bin/bash
echo -e "\n[NOTE] Installing mujoco from source..."

# the script directory and mujoco directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mujoco_dir="$script_dir/../mujoco"

# check whether we already have the most recent release cached to save build time
LATEST_GIT_HASH=$(git ls-remote https://github.com/google-deepmind/mujoco HEAD | awk '{ print $1}')
if [ -d "$mujoco_dir/python/dist" ]; then
    tar_path=$(find "$mujoco_dir/python/dist" -name "mujoco-*${LATEST_GIT_HASH}.tar.gz" 2>/dev/null)
    whl_path=$(find "$mujoco_dir/python/dist" -name "mujoco-*.whl" 2>/dev/null)
    if [ -f "$tar_path" ]; then
        echo -e "[NOTE] Found cached mujoco tar.gz file..."
        # if wheel exists, install from wheel
        if [ -f "$whl_path" ]; then
            echo -e "[NOTE] Wheel found! Installing from wheel..."
            cd "$mujoco_dir/python/dist"

            # [Nov 10, 2023] we install with --no-deps because this would upgrade numpy to a version incompatible with cmeel-boost 1.82.0
            MUJOCO_PATH="$mujoco_dir/mujoco_install" MUJOCO_PLUGIN_PATH="$mujoco_dir/plugin" pip install --no-deps --force-reinstall "$whl_path"
            cd "$mujoco_dir/mjx"
            pip install --no-deps --force-reinstall .
            exit 0
        else
            echo -e "[NOTE] No wheel found! Building it and installing..."
            cd "$mujoco_dir/python/dist"
            MUJOCO_PATH="$mujoco_dir/mujoco_install" MUJOCO_PLUGIN_PATH="$mujoco_dir/plugin" pip wheel -w $(dirname "whl_path") "$tar_path"

            # [Nov 10, 2023] we install with --no-deps because this would upgrade numpy to a version incompatible with cmeel-boost 1.82.0
            MUJOCO_PATH="$mujoco_dir/mujoco_install" MUJOCO_PLUGIN_PATH="$mujoco_dir/plugin" pip install --no-deps --force-reinstall "$whl_path"
            cd "$mujoco_dir/mjx"
            pip install --no-deps --force-reinstall .
            exit 0
        fi
    fi
fi

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
if [ -d "$mujoco_dir" ]; then
    echo "Mujoco exists already - running git pull to update it!"
    (cd "$mujoco_dir" && git pull origin main)
else
    git clone https://github.com/google-deepmind/mujoco.git "$mujoco_dir"
fi

# Configure and build
export MAKEFLAGS="-j$(nproc)"  # allow the max number of processors when building
cd "$mujoco_dir"
SAVED_GIT_HASH=$(git rev-parse HEAD)  # getting the git hash
cmake .
cmake --build .

# Install Mujoco
cmake . -DCMAKE_INSTALL_PREFIX="./mujoco_install"
cmake --install .

# Generate source distribution required for Python bindings
cd "$mujoco_dir/python"
./make_sdist.sh
tar_path=$(find "$mujoco_dir/python/dist" -name 'mujoco-*.tar.gz' 2>/dev/null)

# Renaming the tar file to append the commit hash
old_name=$(basename "$tar_path")
old_prefix="${old_name%%.tar.gz}"
new_prefix="${old_prefix}-${SAVED_GIT_HASH}"
new_name="${new_prefix}.tar.gz"
new_tar_path="$(dirname "$tar_path")/$new_name"
mv "$tar_path" "$new_tar_path"  # renames the tar file to have the git hash in it

# manually building wheel so we can cache it
cd "$mujoco_dir/python/dist"
MUJOCO_PATH="$mujoco_dir/mujoco_install" MUJOCO_PLUGIN_PATH="$mujoco_dir/plugin" pip wheel -w $(dirname "new_tar_path") "$new_tar_path"

# installing mujoco from wheel and then finally mjx
whl_path=$(find "$mujoco_dir/python/dist" -name "mujoco-*.whl" 2>/dev/null)
MUJOCO_PATH="$mujoco_dir/mujoco_install" MUJOCO_PLUGIN_PATH="$mujoco_dir/plugin" pip install --no-deps --force-reinstall "$whl_path"
cd "$mujoco_dir/mjx"
pip install --no-deps --force-reinstall .