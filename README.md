# AMBER Simulation Tools
This repository houses tools built on the GPU-accelerated simulation capabilities of MuJoCo 3. The goal is to act as a centralized location for
* common robot models shared across the lab,
* shared interfaces for control architectures, and
* massively-parallelized simulation.

## Quickstart

### Non-developers
Create a conda environment with Cuda 11.8 support:
```
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```
To locally install this package, clone the repository and in the repo root, run
```
pip install . --default-timeout=100 future --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --find-links https://download.pytorch.org/whl/cu118
```

### Developers
Create a conda environment with Cuda 11.8 support:
```
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```
Install the project with the editable flag and development dependencies:
```
pip install -e .[all] --default-timeout=100 future --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --find-links https://download.pytorch.org/whl/cu118
```
Then, install pre-commit hooks by running the following in the repo root:
```
pre-commit autoupdate
pre-commit install
```

## Development Details

### Abridged Dev Guidelines
Development on this code will be controlled via code review. To facilitate this, please follow these guidelines:
* keep your pull requests small so that it's practical to human review them,
* write tests as you go (and if you are reviewing, suggest missing tests),
* write docstrings for public classes and methods, even if it's just a one-liner,
* before committing, make sure you locally pass all tests by running `pytest` in the repo root.

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
