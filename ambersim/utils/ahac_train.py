import hydra, os, wandb, yaml
from IPython.core import ultratb
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from gym import wrappers
from omegaconf import OmegaConf, open_dict


from brax import envs
from brax.io import model
from ambersim.envs.exo_base import Exo

import jax

cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, "cfg")


@hydra.main(config_path="cfg", config_name="ahac_config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    # if cfg.general.run_wandb:
    #     create_wandb_run(cfg.wandb, cfg_full)

    # patch code to make jobs log in the correct directory when doing multirun
    logdir = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(logdir, cfg.general.logdir)

    # TODO-DONE : replace with rngkey
    # seeding(cfg.general.seed)
    prng_key = jax.random.PRNGKey(seed=0)

    if "_target_" in cfg.alg:
        # cfg.env.config.no_grad = not cfg.general.train
        env_name = "exo"
        envs.register_environment("exo", Exo)
        env = envs.get_environment(env_name)
        algo = instantiate(cfg.alg, env=env, logdir=logdir)

        if cfg.general.checkpoint:
            algo.load(cfg.general.checkpoint)
        if cfg.general.train:
            algo.train()
        else:
            # algo.run(cfg.env.player.games_num)
            raise NotImplementedError
    else:
        raise NotImplementedError

if __name__ == "__main__":
    train()