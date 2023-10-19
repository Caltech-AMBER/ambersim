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
pip install . --default-timeout=100 future
```

### Developers
Create a conda environment with Cuda 11.8 support:
```
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```
Install the project with the editable flag and development dependencies:
```
pip install -e .[dev] --default-timeout=100 future
```
Then, install pre-commit hooks by running the following in the repo root:
```
pre-commit autoupdate
pre-commit install
```

## Development

### Abridged Dev Guidelines
Development on this code will be controlled via code review. To facilitate this, please follow these guidelines:
* keep your pull requests small so that it's practical to human review them,
* write tests as you go (and if you are reviewing, suggest missing tests),
* write docstrings for public classes and methods, even if it's just a one-liner.

### Dependency Management
Dependencies are managed using `conda`. 

### Code Cleanliness
Code will be automatically checked using `pre-commit`. We use the following tools to enforce code cleanliness:
* Code style: `flake8`
* Code formatter: `black`
* Imports: `isort`
* Tests: `pytest`
* Type checking: `pyright`

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
