# AMBER Simulation Tools
This repository houses tools built on the GPU-accelerated simulation capabilities of MuJoCo 3. The goal is to act as a centralized location for
* common robot models shared across the lab,
* shared interfaces for control architectures, and
* massively-parallelized simulation.

## Quickstart
[TODO]

## Development
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

### Dependency Management
Dependencies are managed using `conda`. 

### Code Cleanliness
Code will be automatically checked using `pre-commit`. We use the following tools to enforce code cleanliness:
* Code style: `flake8`
* Imports: `isort`
* Type checking: `pyright`

### Tests
Tests are located in the `tests` directory and are run by `pytest`. To manually run all tests, run the following in the repository root:
```
[TODO]
```
CI will automatically run all tests, so you shouldn't need to do this.
