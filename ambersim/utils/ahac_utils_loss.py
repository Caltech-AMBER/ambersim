import os
import torch
import numpy as np
import copy
import jax.numpy as jnp
import ambersim.utils.ahac_utils_torch_utils as tu
from ambersim.utils.ahac_utils_common import *
from ambersim.utils.ahac_utils_loss import *
import jax

def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros(
            (self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device
        )
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros(
            (self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device
        )

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)

            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.reset(rng=jax.random.PRNGKey(0))
        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                self.obs_rms.update(obs)
            # normalize the current obs
            obs = obs_rms.normalize(obs)

        # keeps track of the current length of the rollout
        rollout_len = torch.zeros((self.num_envs,), device=self.device)
        # Start short horizon rollout
        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad(): 
                obs_tensor=torch.tensor(np.array(obs.info['obs_history']))[-self.num_obs:]
                self.obs_buf[i] = obs_tensor

            # act in environment
            actions = self.actor(obs_tensor, deterministic=deterministic)
            # import ipdb; ipdb.set_trace()

            state = self.env.step(state=obs, action=jnp.tanh(actions.detach().numpy()))
            obs=state
            rew=torch.tensor(np.array(state.reward))
            info=state.info
            term = torch.tensor(np.array(state.done))
            trunc = torch.tensor(np.array(state.done))

            with torch.no_grad():
                raw_rew = copy.deepcopy(rew)

            # scale the reward
            rew = rew * self.rew_scale

            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)

            if self.ret_rms is not None:
                # update ret rms
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)

                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1
            rollout_len += 1

            # contact truncation
            # defaults to jacobian truncation if they are available, otherwise
            # uses contact forces since they are always available
            cfs = torch.tensor(np.array(info["contact_forces"]))
            cfs = torch.gradient(cfs)[0]
            acc = torch.tensor(np.array(info["accelerations"])) # TODO: Check indices match contacts
            
            acc[acc >= 0] = torch.maximum(acc[acc>=0], torch.ones_like(acc[acc>=0]))
            acc[acc < 0] = torch.maximum(acc[acc<0], torch.ones_like(acc[acc<0]))
            # cfs_normalised = torch.where(acc != 0.0, cfs / acc, torch.zeros_like(cfs))
            cfs_normalised = cfs / acc
            # self.cfs[i] = torch.norm(cfs_normalised, dim=(1, 2))
            self.cfs[i] = torch.norm(cfs_normalised, dim=(0))

            if self.log_jacobians:
                jac_norm = (
                    np.linalg.norm(info["jacobian"]) if "jacobian" in info else None
                )
                k = self.step_count + int(torch.sum(rollout_len).item())
                if jac_norm:
                    self.writer.add_scalar("jacobian", jac_norm, k)
                self.writer.add_scalar("contact_forces", cfs_normalised, k)

            # real_obs = info["obs_before_reset"]
            real_obs = obs_tensor
            # sanity check
            if (~torch.isfinite(real_obs)).sum() > 0:
                print("Got inf obs")
                # raise ValueError # it's ok to have this for humanoid

            if self.obs_rms is not None:
                real_obs = obs_rms.normalize(real_obs)

            next_values[i + 1] = self.critic(real_obs).squeeze(-1)

            # handle terminated environments which stopped for some bad reason
            # since the reason is bad we set their value to 0
            term_env_ids = term.nonzero(as_tuple=False).squeeze(-1)
            for id in term_env_ids:
                next_values[i + 1, id] = 0.0

            # sanity check
            if (next_values > 1e6).sum() > 0 or (next_values < -1e6).sum() > 0:
                print("next value error")
                raise ValueError

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            self.early_terms.append(torch.all(term).item())
            self.horizon_truncs.append(i == self.steps_num - 1)
            self.episode_ends.append(torch.all(trunc).item())

            done = torch.tensor(np.array(state.done))
            done_env_ids = done.nonzero(as_tuple=False).flatten()

            self.early_termination += torch.sum(term).item()
            self.episode_end += torch.sum(trunc).item()

            if i < self.steps_num - 1:
                # first terminate all rollouts which are 'done'
                retrn = (
                    -rew_acc[i + 1, done_env_ids]
                    - self.gamma
                    * gamma[done_env_ids]
                    * next_values[i + 1, done_env_ids]
                )
                actor_loss += retrn.sum()
                with torch.no_grad():
                    self.ret[done_env_ids] += retrn
            else:
                # terminate all envs because we reached the end of our rollout
                retrn = -rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]
                actor_loss += retrn.sum()
                with torch.no_grad():
                    self.ret += retrn

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(
                        self.episode_discounted_loss[done_env_ids]
                    )
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    self.horizon_length_meter.update(rollout_len[done_env_ids])
                    rollout_len[done_env_ids] = 0
                    for k, v in filter(lambda x: x[0] in self.score_keys, info.items()):
                        self.episode_scores_meter_map[k + "_final"].update(
                            v[done_env_ids]
                        )
                    import ipdb; ipdb.set_trace()
                    for id in done_env_ids:
                        if self.episode_loss[id] > 1e6 or self.episode_loss[id] < -1e6:
                            print("ep loss error")
                            raise ValueError

                        self.episode_loss_his.append(self.episode_loss[id].item())
                        self.episode_discounted_loss_his.append(
                            self.episode_discounted_loss[id].item()
                        )
                        self.episode_length_his.append(self.episode_length[id].item())
                        self.episode_loss[id] = 0.0
                        self.episode_discounted_loss[id] = 0.0
                        self.episode_length[id] = 0
                        self.episode_gamma[id] = 1.0

        self.horizon_length_meter.update(rollout_len)

        actor_loss /= self.steps_num * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * np.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().item()

        self.step_count += self.steps_num * self.num_envs

        if (
            self.log_jacobians
            and self.step_count - self.last_log_steps > 1000 * self.num_envs
        ):
            np.savez(
                os.path.join(self.log_dir, f"truncation_analysis_{self.episode}"),
                contact_forces=self.cfs,
                early_termination=self.early_terms,
                horizon_truncation=self.horizon_truncs,
                episode_ends=self.episode_ends,
            )
            self.early_terms = []
            self.horizon_truncs = []
            self.episode_ends = []
            self.episode += 1
            self.last_log_steps = self.step_count
        print("Actor loss computed.")
        return actor_loss



def compute_target_values(self):
    if self.critic_method == "one-step":
        self.target_values = self.rew_buf + self.gamma * self.next_values


def compute_critic_loss(self, batch_sample):
    predicted_values = self.critic.predict(batch_sample["obs"]).squeeze(-2)
    target_values = batch_sample["target_values"]
    critic_loss = ((predicted_values - target_values) ** 2).mean()
    print("Critic loss computed.")
    return critic_loss