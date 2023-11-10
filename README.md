# AMBER Simulation Tools
This repository houses tools built on the GPU-accelerated simulation capabilities of MuJoCo 3. The goal is to act as a centralized location for
* common robot models shared across the lab,
* shared interfaces for control architectures, and
* massively-parallelized simulation.

## Quickstart
Create a conda environment with Cuda 11.8 support:
```
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```
Install the project. Note the two different options depending on whether you want to develop on the repo or not.
```
# OPTION 1: NON-DEVELOPERS
pip install . --default-timeout=100 future --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --find-links https://download.pytorch.org/whl/cu118

# OPTION 2: DEVELOPERS
pip install -e .[all] --default-timeout=100 future --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --find-links https://download.pytorch.org/whl/cu118
```
For developers, install pre-commit hooks by running the following in the repo root:
```
pre-commit autoupdate
pre-commit install
```
To install the latest and greatest version of `mujoco` from source, ensure that your system has the right build dependencies:
```
sudo apt-get update -y
sudo apt-get install -y \
    libgl1-mesa-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    ninja-build
```
Then, run the installation `bash` script, which will update `mujoco` to the latest version built from source:
```
sudo chmod +x install.sh
./install.sh
```
If the following line of code runs without error, then the installation was successful:
```
python -c "import mujoco"
```

## Development Details

### Abridged Dev Guidelines
Development on this code will be controlled via code review. To facilitate this, please follow these guidelines:
* keep your pull requests small so that it's practical to human review them;
* write tests as you go (and if you are reviewing, suggest missing tests);
* write docstrings for public classes and methods, even if it's just a one-liner;
* before committing, make sure you locally pass all tests by running `pytest` in the repo root;
* pass static type checking by running `pyright` in the repo root before committing;
* when typing, use conventions compatible with `python3.8`, e.g., instead of typing with `tuple[obj1, obj2]`, use `from typing import Tuple; ... Tuple[obj1, obj2]` and so on.

### Dependencies
Python dependencies are specified using a `pyproject.toml` file. Non-python dependencies are specified using the `environment.yml` file, which can create a compliant `conda` environment.

Major versioning decisions:
* `python=3.11.5`. `torch`, `jax`, and `mujoco` all support it and there are major reported speed improvements over `python` 3.10.
* `cuda==11.8`. Both `torch` and `jax` support `cuda12`; however, they annoyingly support different minor versions which makes them incompatible in the same environment [https://github.com/google/jax/issues/18032](#18032). Once this is resolved, we will upgrade to `cuda-12.2` or later. It seems most likely that `torch` will support `cuda-12.3` once they do upgrade, since that is the most recent release.

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
