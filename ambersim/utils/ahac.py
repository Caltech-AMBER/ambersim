import sys, os
from typing import Optional, List, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate
from tensorboardX import SummaryWriter
import torch
import time
from collections import deque
from ambersim.utils.ahac_utils_loss import *
from ambersim.utils.ahac_utils_dataset import CriticDataset
from ambersim.utils.ahac_utils_model import *
from ambersim.utils.ahac_utils_average_meter import AverageMeter


class AHAC:
    def __init__(
        self,
        # env_config: DictConfig,
        env,
        actor_config: DictConfig,
        critic_config: DictConfig,
        steps_min: int,  # minimum horizon
        steps_max: int,  # maximum horizon
        max_epochs: int,  # number of short rollouts to do (i.e. epochs)
        train: bool,  # if False, we only eval the policy
        logdir: str,
        grad_norm: Optional[float] = None,  # clip actor and ciritc grad norms
        critic_grad_norm: Optional[float] = None,
        contact_threshold: float = 500,  # for cutting horizons
        accumulate_jacobians: bool = False,  # if true clip gradients by accumulation
        actor_lr: float = 2e-3,
        critic_lr: float = 2e-3,
        lambd_lr: float = 1e-4,
        betas: Tuple[float, float] = (0.7, 0.95),
        lr_schedule: str = "linear",
        gamma: float = 0.99,
        lam: float = 0.95,
        rew_scale: float = 1.0,
        critic_iterations: Optional[int] = None,  # if None, we do early stop
        critic_batches: int = 4,
        critic_method: str = "one-step",
        save_interval: int = 500,  # how often to save policy
        score_keys: List[str] = [],
        eval_runs: int = 12,
        log_jacobians: bool = False,  # expensive and messes up wandb
        device: str = "cuda",

        num_envs: int=1, num_actions: int=12, num_obs: int=12, episode_length: int=50
    ):
        # sanity check parameters
        assert steps_max > steps_min > 0
        assert max_epochs > 0
        assert actor_lr > 0; assert critic_lr > 0; assert lambd_lr > 0; assert lr_schedule in ["linear", "constant"]
        assert 0 < gamma <= 1; assert 0 < lam <= 1
        assert rew_scale > 0.0
        assert critic_iterations is None or critic_iterations > 0; assert critic_batches > 0; assert critic_method in ["one-step", "td-lambda"]
        assert save_interval > 0
        assert eval_runs >= 0

        # Create environment
        self.env = env
        # print("num_envs = ", self.env.num_envs)
        # print("num_actions = ", self.env.num_actions)
        # print("num_obs = ", self.env.num_obs)

        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.max_episode_length = episode_length
        # self.device = torch.device(device)
        self.device="cpu"

        self.steps_min = steps_min
        self.steps_max = steps_max
        self.H = torch.tensor(steps_min, dtype=torch.float32)
        self.lambd = torch.tensor([0.0]*steps_min, dtype=torch.float32)
        self.C = contact_threshold
        self.max_epochs = max_epochs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_schedule = lr_schedule

        self.gamma = gamma
        self.lam = lam
        self.rew_scale = rew_scale

        self.critic_method = critic_method
        self.critic_iterations = critic_iterations
        self.critic_batches = critic_batches
        self.critic_batch_size = self.num_envs * self.steps_max // critic_batches
        self.obs_rms=None
        self.ret_rms=None
        env_name = self.env.__class__.__name__
        self.name = self.__class__.__name__ + "_" + env_name

        self.grad_norm = grad_norm
        self.critic_grad_norm = critic_grad_norm
        # self.stochastic_evaluation = stochastic_eval
        self.save_interval = save_interval

        if train:
            self.log_dir = logdir
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_dir, "log"))

        # Create actor and critic
        self.actor = ActorDeterministicMLP(
            obs_dim=self.num_obs,
            action_dim=self.num_actions,
            units=[64,64], 
            activation='elu', 
            device=self.device,
        )

        self.critic = DoubleCriticMLP(
            obs_dim=self.num_obs,
            units=[64,64],
            activation='elu',
            device=self.device)

        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())

        # for logging purposes
        self.jac_buffer = []
        self.jacs = []
        self.early_terms = []
        self.conatct_truncs = []
        self.horizon_truncs = []
        self.episode_ends = []
        self.episode = 0

        if train:
            self.save("init_policy")

        # TODO: Change optimizers to jax
        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            self.actor_lr,
            betas,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            self.critic_lr,
            betas,
        )
        self.lambd_lr = lambd_lr

        # replay buffer
        self.init_buffers()

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32
        )
        self.episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32
        )
        self.episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32
        )
        # NOTE: do not need for single env
        self.episode_length = torch.zeros(self.num_envs, dtype=int)
        self.done_buf = torch.zeros(self.num_envs, dtype=bool)
        self.best_policy_loss = torch.inf
        self.actor_loss = torch.inf
        self.value_loss = torch.inf
        self.grad_norm_before_clip = torch.inf
        self.grad_norm_after_clip = torch.inf
        self.early_termination = 0
        self.episode_end = 0
        self.contact_trunc = 0
        self.horizon_trunc = 0
        self.acc_jacobians = accumulate_jacobians
        self.log_jacobians = log_jacobians
        self.eval_runs = eval_runs
        self.last_steps = 0
        self.last_log_steps = 0

        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.horizon_length_meter = AverageMeter(1, 100).to(self.device)
        self.score_keys = score_keys
        self.episode_scores_meter_map = {
            key + "_final": AverageMeter(1, 100).to(self.device)
            for key in self.score_keys
        }

        print("AHAC init finished.")

    
    def train(self): 
        
        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch} started.")
            time_start_epoch = time.time()

            lr = self.actor_lr
            lambd_lr = self.lambd_lr

            # train actor
            # self.time_report.start_timer("actor training")
            self.actor_optimizer.step(self.actor_closure)    # TODO 1: Actor closure function
            # self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            # self.time_report.start_timer("prepare critic dataset")
            compute_target_values(self)
            critic_batch_size = (
                self.num_envs * self.steps_num // self.critic_batches
            )
            # import ipdb; ipdb.set_trace()
            dataset = CriticDataset(                # TODO 2: Critic dataset creation; or some sort of history
                critic_batch_size,
                self.obs_buf,
                self.target_values,
                drop_last=False,
            )
            # self.time_report.end_timer("prepare critic dataset")

            # self.time_report.start_timer("critic training")
            self.value_loss = 0.0
            last_losses = deque(maxlen=5)
            iterations = self.critic_iterations if self.critic_iterations else 64
            for j in range(iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = compute_critic_loss(self, batch_sample)
                    # training_critic_loss = 0.0
                    training_critic_loss.backward()

                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    # if self.critic_grad_norm:
                    #     clip_grad_norm_(self.critic.parameters(), self.critic_grad_norm) #TODO 3: Replace function

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                total_critic_loss /= batch_cnt
                #### what is this ... START
                if self.critic_iterations is None and len(last_losses) == 5:
                    # import ipdb; ipdb.set_trace()
                    diff = abs(torch.diff(torch.tensor(last_losses)).mean())
                    if diff < 2e-1:
                        iterations = j + 1
                        break
                last_losses.append(total_critic_loss.item())
                #### END ############

                self.value_loss = total_critic_loss
                print("value iter {}/{}, loss = {:7.6f}".format(j + 1, iterations, self.value_loss), end="\r",)

            # self.time_report.end_timer("critic training")

            last_steps = self.steps_num

            # Train horizon
            self.lambd -= lambd_lr * (self.C - self.cfs.mean(-1))
            self.H += lambd_lr * self.lambd.sum()
            self.H = torch.clip(self.H, self.steps_min, self.steps_max)
            print(f"H={self.H.item():.2f}, lambda={self.lambd.mean().item():.2f}")

            # reset buffers correctly for next iteration
            self.init_buffers()

            self.iter_count += 1

            time_end_epoch = time.time()

            fps = last_steps * self.num_envs / (time_end_epoch - time_start_epoch)

            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = (
                    self.episode_discounted_loss_meter.get_mean()
                )

                if mean_policy_loss < self.best_policy_loss:
                    print("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss
            else:
                mean_policy_loss = torch.inf
                mean_policy_discounted_loss = torch.inf
                mean_episode_length = 0
            print(f"Epoch {epoch} finished.")
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + "policy_iter{}_reward{:.3f}".format(self.iter_count, -mean_policy_loss))
    
    def init_buffers(self):
            self.obs_buf = torch.zeros(
                (self.steps_num, self.num_envs, self.num_obs),
                dtype=torch.float32,
                device=self.device,
            )
            self.rew_buf = torch.zeros(
                (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
            )
            self.done_mask = torch.zeros(
                (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
            )
            self.next_values = torch.zeros(
                (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
            )
            self.target_values = torch.zeros(
                (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
            )
            self.ret = torch.zeros(
                (self.num_envs), dtype=torch.float32, device=self.device
            )
            self.cfs = torch.zeros(
                (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
            )
            self.lambd = self.lambd[0].repeat(self.steps_num)

    def save(self, filename=None):
        if filename is None:
            filename = "best_policy"
        torch.save(
            [self.actor, self.critic, self.obs_rms, self.ret_rms],
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path, actor=True):
        print("Loading policy from", path)
        checkpoint = torch.load(path)
        if actor:
            self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.obs_rms = checkpoint[2].to(self.device)
        self.ret_rms = (
            checkpoint[3].to(self.device)
            if checkpoint[3] is not None
            else checkpoint[3]
        )

    def log_scalar(self, scalar, value):
        """Helper method for consistent logging"""
        self.writer.add_scalar(f"{scalar}", value, self.iter_count)

    def close(self):
        self.writer.close()





    @property
    def steps_num(self):
        return round(self.H.item())

  
    def actor_closure(self):
        self.actor_optimizer.zero_grad() 
        # self.time_report.start_timer("compute actor loss") 
        # self.time_report.start_timer("forward simulation")
        actor_loss = compute_actor_loss(self)
        # self.time_report.end_timer("forward simulation") 
        # self.time_report.start_timer("backward simulation")
        actor_loss.backward()
        # self.time_report.end_timer("backward simulation") 
        with torch.no_grad():
            self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
            # if self.grad_norm:
            #     clip_grad_norm_(self.actor.parameters(), self.grad_norm)
            self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 
            # sanity check
            if (
                torch.isnan(torch.tensor(self.grad_norm_before_clip))
                or self.grad_norm_before_clip > 1e6
            ):
                print("NaN gradient")
                raise ValueError 
        # self.time_report.end_timer("compute actor loss") 
        return actor_loss
