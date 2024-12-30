# AHAC README

## Setup

Create docker env:
```
mkdir cuda118-docker && cd cuda118-docker
touch Dockerfile
# copy-paste contents of ambersim_cut/dockerfile
```
Build and run docker image
```
docker build -t cuda118_docker .
docker run --gpus all -it -v {$HOME_DIR}/ambersim_cut:/workspace/ambersim cuda118-docker bash
```
Setup conda environment:
```
cd ambersim
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```

## Code

### Core files:

- ambersim/utils/ahac.py                  :   main training loop
- ambersim/utils/ahac_utils_loss.py       :   actor/Critic loss functions
- ambersim/utils/ahac_training_utils.py   :   policy initialization (hyperparameters etc.)
- ambersim/utils/ahac_utils_common.py     :   gradient function
- ambersim/envs/exo_base.py               :   environment def
- examples/exo/exo_ahac.py                :   training script
- examples/exo/exo_ahac_load.py           :   inference script

### Training/Inference

Run training:

```examples/exo/exo_ahac.py```

Rollout inference from checkpoint:

```examples/exo/exo_ahac_load.py --ckpt_path {$CKPT_PATH}```