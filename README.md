# AMBER Simulation Tools
This repository houses tools built on the GPU-accelerated simulation capabilities of MuJoCo 3. The goal is to act as a centralized location for
* common robot models shared across the lab,
* shared interfaces for control architectures, and
* massively-parallelized simulation.

## Installation
Clone this repository and run the following commands in the repository root to create and activate a conda environment with Cuda 11.8 support:
```
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```

TL;DR: installation commands are here. This will ask you for your password to install system-wide dependencies. For details, see below.

For non-developers installing `mujoco` from source:
```
./install.sh -s
```

For developers installing `mujoco` from source:
```
# no path to the mujoco repo specified
./install.sh -s -d

# specifying a path to the mujoco repo
./install.sh -s -d --mujoco-dir /path/ending/in/mujoco
```

Installation of this package is done via the above `bash` script. There are a few flags for configuring the installation:
* `-d` controls whether to use the heavier _development_ dependencies, which include linting and testing dependencies;
* `-s` controls whether to install the most recent `mujoco` version from source. We recommend doing this, since the development version usually has important bugfixes.
* `--disable-apt` specifies whether to disable the system-wide dependencies installed by `apt` which are required to install `mujoco` from source. They are enabled by default. The packages are:
    * `libgl1-mesa-dev`
    * `libxinerama-dev`
    * `libxcursor-dev`
    * `libxrandr-dev`
    * `libxi-dev`
    * `ninja-build`
* `--mujoco-dir` specifies the directory of the local `mujoco` repo, which must end in the directory `mujoco`. If one doesn't exist, it will be pulled to this directory. If this isn't specified, `mujoco` will be created as a sibling directory of `ambersim`.

If the following line of code runs without error, then the installation of `mujoco` from source was successful:
```
python -c "import mujoco; from mujoco import mjx"
```
Further, you can examine the latest minor version using `pip`:
```
pip show mujoco
pip show mujoco-mjx
```

## Development Details

### Abridged Dev Guidelines
Development on this code will be controlled via code review. To facilitate this, please follow these guidelines:
* keep your pull requests small so that it's practical to human review them;
* try to create _draft pull requests_ instead of regular ones and request reviews from relevant people only when ready - we rebuild `mujoco` from source when this happens;
* write tests as you go (and if you are reviewing, suggest missing tests);
* write docstrings for public classes and methods, even if it's just a one-liner;
* before committing, make sure you locally pass all tests by running `pytest` in the repo root;
* pass static type checking by running `pyright` in the repo root before committing;
* when typing, use conventions compatible with `python3.8`, e.g., instead of typing with `tuple[obj1, obj2]`, use `from typing import Tuple; ... Tuple[obj1, obj2]` and so on.

### Dependencies
Python dependencies are specified using a `pyproject.toml` file. Non-python dependencies are specified using the `environment.yml` file, which can create a compliant `conda` environment.

Major versioning decisions:
* `python=3.11.5`. `torch`, `jax`, and `mujoco` all support it and there are major reported speed improvements over `python` 3.10.
* `cuda==11.8`. Both `torch` and `jax` support `cuda12`; however, they annoyingly support different minor versions which makes them [incompatible in the same environment](https://github.com/google/jax/issues/18032). Once this is resolved, we will upgrade to `cuda-12.2` or later. It seems most likely that `torch` will support `cuda-12.3` once they do upgrade, since that is the most recent release.

### Tooling
We use various tools to ensure code quality.

| Tooling       | Support                                           |
| ------------- | ------------------------------------------------- |
| Style         | [flake8](https://flake8.pycqa.org/en/latest/)     |
| Formatting    | [black](https://black.readthedocs.io/en/stable/)  |
| Imports       | [isort](https://pycqa.github.io/isort/)           |
| Tests         | [pytest](https://docs.pytest.org/en/stable/)      |
| Type Checking | [pyright](https://microsoft.github.io/pyright/#/) |

We follow [Google code style conventions](https://google.github.io/styleguide/pyguide.html) (in particular, for docstrings).

### Dev Contact
In alphabetical order, the members of the lab working on this are:
* Adrian (aghansah@caltech.edu)
* Albert (alberthli@caltech.edu)
* Amy (kli5@caltech.edu)
* Gary (lzyang@caltech.edu)
* Ivan (ivan.jimenez@caltech.edu)
* Noel (noelcs@caltech.edu)
* Preston (pculbert@caltech.edu)
* Will (wcompton@caltech.edu)
* Vince (vkurtz@caltech.edu)
* Zach (zolkin@caltech.edu)
